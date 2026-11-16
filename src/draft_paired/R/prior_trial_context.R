# Use fixed seeds for the native simulation stages in a local function copy.
# Consume the stage-seed draw on every call to keep the RNG sequence aligned.
# Native code computes the weighted GRN for each snapshot.
make_pinned_snapshot_runner <- function(native, master_seed, rng_kind) {
  force(native)
  native_body <- as.list(body(native))
  is_seed_assignment <- function(expr) {
    is.call(expr) && identical(expr[[1]], as.name('<-')) &&
      identical(expr[[2]], as.name('seed')) && is.call(expr[[3]]) &&
      as.character(expr[[3]][[1]]) %in% c('sample', 'sample.int')
  }
  positions <- which(vapply(native_body, is_seed_assignment, logical(1)))
  if (length(positions) != 1L) stop('Unsupported scMultiSim native stage seed layout')
  position <- positions[[1]]
  sampler <- native_body[[position]][[3]]
  do.call(RNGkind, as.list(rng_kind))
  set.seed(master_seed)
  stage_seeds <- eval(sampler, envir=environment(native))
  if (length(stage_seeds) != 9L || anyNA(stage_seeds)) stop('Expected nine native stage seeds')
  native_body[[position]][[3]] <- substitute({DRAW; FIXED}, list(DRAW=sampler, FIXED=stage_seeds))
  pinned <- native
  body(pinned) <- as.call(native_body)
  list(run=pinned, seeds=stage_seeds)
}
