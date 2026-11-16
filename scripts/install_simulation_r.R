# One-time setup, for example: Rscript scripts/install_simulation_r.R .R-library
args <- commandArgs(trailingOnly=TRUE)
if (getRversion() < '4.4.0') stop('Bioconductor scMultiSim 1.2.0 requires R >= 4.4.0')
lib <- if (length(args)) args[[1]] else '.R-library'
dir.create(lib, recursive=TRUE, showWarnings=FALSE)
lib <- normalizePath(lib, winslash='/', mustWork=TRUE)
.libPaths(c(lib, .libPaths()))
cran <- 'https://cloud.r-project.org'
for (p in c('ape', 'digest')) {
  if (!requireNamespace(p, quietly=TRUE)) install.packages(p, lib=lib, repos=cran)
}
script_path <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
checker <- normalizePath(file.path(dirname(script_path),'../src/draft_paired/R/check_simulator.R'),winslash='/')
source(checker)
# Verify the package in an isolated R process.
check_script <- tempfile(fileext='.R')
writeLines(c(paste0('.libPaths(c(',encodeString(lib,quote='"'),',.libPaths()))'),
             paste0('source(',encodeString(checker,quote='"'),')'),
             'verify_scmultisim_implementation()'),check_script)
rscript <- file.path(R.home('bin'),if (.Platform$OS.type=='windows') 'Rscript.exe' else 'Rscript')
check_ready <- function() {
  output <- suppressWarnings(system2(rscript,c('--vanilla',shQuote(check_script)),stdout=TRUE,stderr=TRUE))
  is.null(attr(output,'status')) || attr(output,'status')==0L
}
if (!check_ready()) {
  for (p in c('BiocManager','remotes')) {
    if (!requireNamespace(p,quietly=TRUE)) install.packages(p,lib=lib,repos=cran)
  }
  options(repos=BiocManager::repositories())
  BiocManager::install(c('SummarizedExperiment','BiocParallel'), lib=lib, ask=FALSE, update=FALSE)
  archive <- tempfile(fileext='.tar.gz')
  download.file('https://bioconductor.posit.co/packages/3.20/bioc/src/contrib/scMultiSim_1.2.0.tar.gz',
                archive,mode='wb',quiet=TRUE)
  if (digest::digest(archive,algo='sha256',file=TRUE) != scmultisim_source_sha256) {
    stop('Official scMultiSim source checksum mismatch; installation stopped.')
  }
  remotes::install_local(archive,lib=lib,dependencies=NA,upgrade='never',force=TRUE)
  unlink(archive)
  if (!check_ready()) stop('Installed scMultiSim implementation failed verification.')
}
unlink(check_script)
cat('Ready. Pass --r-library', shQuote(lib), 'to draft_paired simulation.\n')
