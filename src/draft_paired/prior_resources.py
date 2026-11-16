"""Versioned annotation resources and bounded HTTP range reads for motif tracks."""

from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.request
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

RESOURCE_CATALOG = {
    "hg38": {
        "annotation_version": "GENCODE_44",
        "annotation_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz",
    },
    "mm10": {
        "annotation_version": "GENCODE_M25",
        "annotation_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz",
    },
    "dm6": {
        "annotation_version": "Ensembl_112_BDGP6.46",
        "annotation_url": "https://ftp.ensembl.org/pub/release-112/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.46.112.gtf.gz",
    },
}
for _genome, _resource in RESOURCE_CATALOG.items():
    _resource["motif_version"] = "JASPAR_2024"
    _resource["motif_url"] = (
        f"https://mencius-pub.uio.no/JASPAR/JASPAR_TFBSs/2024/JASPAR2024_{_genome}.bb"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_cached(url: str, cache: Path) -> tuple[Path, dict]:
    """Download one pinned annotation, retaining and checking its content digest."""
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    target, metadata = cache / f"{key}.gtf.gz", cache / f"{key}.json"
    if target.exists() and metadata.exists():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("url") == url and sha256(target) == record.get("sha256"):
            return target, record
        raise ValueError(
            f"Cached annotation changed: {target}; remove it to download again."
        )
    temporary = cache / f".{uuid4().hex}.download"
    request = urllib.request.Request(
        url, headers={"User-Agent": "DRAFT prior preparation"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as source,
            temporary.open("wb") as out,
        ):
            length = source.headers.get("Content-Length")
            for block in iter(lambda: source.read(1024 * 1024), b""):
                out.write(block)
        if length is not None and temporary.stat().st_size != int(length):
            raise OSError(f"Incomplete annotation download: {url}")
        with temporary.open("rb") as handle:
            if handle.read(2) != b"\x1f\x8b":
                raise OSError(
                    f"The annotation source did not return a gzip-compressed GTF: {url}"
                )
        record = {
            "url": url,
            "bytes": temporary.stat().st_size,
            "sha256": sha256(temporary),
        }
        temporary.replace(target)
        metadata.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return target, record
    finally:
        temporary.unlink(missing_ok=True)


class HTTPRangeReader(io.RawIOBase):
    """Seekable HTTP byte ranges with bounded memory use.

    Cache at most ``cache_blocks * block_size`` bytes and check Content-Range
    and resource identity on every request.
    """

    def __init__(self, url: str, *, block_size=256 * 1024, cache_blocks=64, timeout=30):
        super().__init__()
        self.url, self.block_size = url, int(block_size)
        self.cache_blocks, self.timeout = int(cache_blocks), timeout
        self.position, self.length = 0, None
        self.identity = None
        self.blocks = OrderedDict()
        self.bytes_downloaded = 0
        self._block(0)

    def _block(self, index):
        if index in self.blocks:
            self.blocks.move_to_end(index)
            return self.blocks[index]
        start = index * self.block_size
        end = start + self.block_size - 1
        if self.length is not None:
            end = min(end, self.length - 1)
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "DRAFT prior preparation",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if response.status != 206 or match is None:
                raise OSError(
                    f"The motif server must support HTTP byte ranges: {self.url}"
                )
            first, last, total = map(int, match.groups())
            if first != start or last != min(end, total - 1):
                raise OSError(
                    "The motif server returned a different byte range than requested."
                )
            identity = response.headers.get("ETag") or response.headers.get(
                "Last-Modified"
            )
            if self.length is not None and (
                total != self.length or identity != self.identity
            ):
                raise OSError("The remote motif resource changed during extraction.")
            self.length, self.identity = total, identity
            data = response.read(last - first + 2)
            if len(data) != last - first + 1:
                raise OSError("Incomplete motif byte-range response.")
        self.bytes_downloaded += len(data)
        self.blocks[index] = data
        if len(self.blocks) > self.cache_blocks:
            self.blocks.popitem(last=False)
        return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.length + offset
        else:
            raise ValueError("Invalid seek origin.")
        if position < 0:
            raise ValueError("Cannot seek before the start of a resource.")
        self.position = position
        return position

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Cannot read a closed motif resource.")
        if size is None or size < 0:
            if self.length - self.position > 128 * 1024 * 1024:
                raise io.UnsupportedOperation(
                    "Whole-genome motif downloads are disabled; use bounded reads."
                )
            size = self.length - self.position
        size = max(0, min(size, self.length - self.position))
        pieces = []
        while size:
            index, offset = divmod(self.position, self.block_size)
            block = self._block(index)
            take = min(size, len(block) - offset)
            pieces.append(block[offset : offset + take])
            self.position += take
            size -= take
        return b"".join(pieces)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self):
        self.blocks.clear()
        super().close()


@contextmanager
def open_motif_track(source: str):
    try:
        import pybigtools
    except ImportError as exc:
        raise RuntimeError(
            "Automatic motif retrieval needs pybigtools; install draft[prior] or draft[full]."
        ) from exc
    if source.startswith(("https://", "http://")):
        with HTTPRangeReader(source) as stream, pybigtools.open(stream) as reader:
            yield reader, stream
    else:
        with open(source, "rb") as stream, pybigtools.open(stream) as reader:
            yield reader, None
