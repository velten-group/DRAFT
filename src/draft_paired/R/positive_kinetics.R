# Rejection-sample nonpositive kinetic rates within the current R process.
# Retain positive draws and use the configured mean for replacement draws.
positive_kinetic_draws <- function(values, center) {
  if (!is.numeric(center) || length(center) != 1L || !is.finite(center) || center <= 0 || any(!is.finite(values))) {
    stop('Invalid native kinetic-rate mean or nonfinite rate')
  }
  corrected <- values
  rounds <- 0L
  while (any(corrected <= 0)) {
    bad <- which(corrected <= 0)
    corrected[bad] <- rnorm(length(bad), mean=center, sd=.1)
    rounds <- rounds+1L
    if (rounds > 1000L) stop('Positive kinetic-rate rejection sampler did not finish')
  }
  list(values=corrected, rounds=rounds)
}

install_positive_kinetic_rates <- function(audit) {
  native_edge <- get('.rnaSimEdge', asNamespace('scMultiSim'))
  checked <- FALSE
  assignInNamespace('.rnaSimEdge', function(sim, ...) {
    if (!checked) {
      rejected <- data.frame(gene=integer(), role=character(), original=double(), replacement=double(), mean=double(), sd=double(), rounds=integer())
      for (role in c('beta','d')) {
        field <- paste0(role,'_genes')
        before <- sim[[field]]
        result <- positive_kinetic_draws(before, sim$options[[role]])
        sim[[field]] <- result$values
        bad <- which(before <= 0)
        if (length(bad)) {
          rejected <- rbind(rejected, data.frame(gene=bad,role=role,
            original=before[bad],replacement=result$values[bad],
            mean=sim$options[[role]],sd=.1,rounds=result$rounds))
        }
      }
      rates <- data.frame(gene=seq_along(sim$beta_genes), beta=sim$beta_genes,d=sim$d_genes)
      if (any(rates$beta <= 0) || any(rates$d <= 0)) stop('Nonpositive kinetic rates remain')
      write.table(rates,file.path(audit,'kinetic_rates.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
      write.table(rejected,file.path(audit,'kinetic_rate_resampling.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
      checked <<- TRUE
    }
    native_edge(sim, ...)
  },ns='scMultiSim')
}
