"""Shared configuration, resource and path helpers."""

from __future__ import annotations

from pathlib import Path
from importlib.resources import files as resource_files
import re
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is declared in pyproject
    yaml = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_base_dir(config: dict[str, Any]) -> Path:
    return Path(str(config.get("base_dir", "."))).expanduser().resolve()


def resolve_config_path(config_or_base: dict[str, Any] | str | Path, path: str | Path) -> Path:
    if isinstance(config_or_base, dict):
        base_dir = config_base_dir(config_or_base)
    else:
        base_dir = Path(config_or_base).expanduser().resolve()
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = base_dir / target
    return target.resolve()


def ensure_directory(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resource_traversable(package: str, relative_path: str):
    resource = resource_files(package)
    for part in Path(relative_path).parts:
        resource = resource.joinpath(part)
    return resource


def read_packaged_text(package: str, relative_path: str) -> str:
    resource = _resource_traversable(package, relative_path)
    return resource.read_text(encoding="utf-8")


def _normalize_text_for_compare(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def write_packaged_text(
    package: str,
    relative_path: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    content = read_packaged_text(package, relative_path)
    return write_config_text(content, output_path, overwrite=overwrite)


def write_config_text(content: str, output_path: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(output_path).expanduser()
    if target.exists() and not overwrite:
        try:
            existing = target.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing is not None and _normalize_text_for_compare(existing) == _normalize_text_for_compare(content):
            return target.resolve()
        raise FileExistsError(f"{target} already exists. Use --force to overwrite it.")
    ensure_directory(target.parent)
    target.write_text(content, encoding="utf-8")
    return target.resolve()


_CONFIG_TOKEN_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _is_scalar_for_context(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _replace_config_tokens(value: str, context: dict[str, Any]) -> str:
    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            return match.group(0)
        return str(context[key])

    return _CONFIG_TOKEN_RE.sub(repl, value)


def _expand_top_level_context(data: dict[str, Any]) -> dict[str, Any]:
    context = {key: value for key, value in data.items() if _is_scalar_for_context(value)}
    for _ in range(8):
        changed = False
        for key, value in list(context.items()):
            if not isinstance(value, str):
                continue
            expanded = _replace_config_tokens(value, context)
            if expanded != value:
                context[key] = expanded
                changed = True
        if not changed:
            break
    return context


def _expand_config_values(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, dict):
        return {key: _expand_config_values(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_config_values(item, context) for item in value]
    if isinstance(value, str):
        return _replace_config_tokens(value, context)
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files.")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must be a mapping.")
    context = _expand_top_level_context(data)
    return _expand_config_values(data, context)


def emit_progress(status: str, task: str, detail: str | None = None) -> None:
    line = f"[{status}] {task}"
    if detail:
        line = f"{line} | {detail}"
    print(line)
