# Native scMultiSim generation from a user-supplied candidate GRN.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 11) stop('Expected folder plus ten binary trajectory parameters')
folder <- args[[1]]
seed <- as.integer(args[[2]])
ntf <- as.integer(args[[3]])
ngene <- as.integer(args[[4]])
cells <- as.integer(args[[5]])
times <- as.integer(args[[6]])
intrinsic <- as.numeric(args[[7]])
depth <- as.numeric(args[[8]])
diff_cif <- as.numeric(args[[9]])
atac_effect <- as.numeric(args[[10]])
cif_sigma <- as.numeric(args[[11]])
suppressPackageStartupMessages({library(scMultiSim); library(ape)})
if (as.character(packageVersion('scMultiSim')) != '1.2.0') stop('Requires scMultiSim 1.2.0')
script_path <- sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1])
source(file.path(dirname(script_path), 'prior_trial_context.R'))
source(file.path(dirname(script_path), 'positive_kinetics.R'))
source(file.path(dirname(script_path), 'check_simulator.R'))
audit <- file.path(folder, 'generator_only')
implementation <- verify_scmultisim_implementation(audit)
install_positive_kinetic_rates(audit)
read_matrix <- function(name) as.matrix(read.csv(file.path(audit, name), row.names=1, check.names=FALSE))
motif <- read_matrix('potential_motif.csv')
distance <- read_matrix('potential_distance.csv')
events <- read.delim(file.path(audit, 'transitions.tsv'))
planned <- lapply(0:(times-1), function(t) {
  as.matrix(read.csv(file.path(folder, sprintf('grn_time_%d.csv', t)), row.names=1, check.names=FALSE))[(ntf+1):ngene, 1:ntf, drop=FALSE]
})
initial <- planned[[1]]
# Each cell uses the binary graph of its declared segment.
# The kinetic RNA simulator carries expression between segments.
weights_at <- function(position) {
  stage <- min(times-1L, max(0L, as.integer(floor(position+.5))))
  planned[[stage+1L]]
}
potential <- (t(distance) %*% motif) > 0
if (any(initial > potential) || times < 2L) stop('Invalid potential graph/stage count')
context <- new.env()

# Supply an equal-size cell grid per segment to the native Brownian sampler.
native_sample_edge <- get('SampleEdge', asNamespace('scMultiSim'))
assignInNamespace('SampleEdge', function(edge, depth, anc_state, edges, ncells, step_size, t_sample=NA) {
  if (is.na(t_sample[1])) {
    count <- edge[4]*ncells/sum(edges[,4])
    if (abs(count-round(count)) > 1e-10 || count < 2) stop('Trajectory requires an integral grid per segment')
    t_sample <- depth + seq_len(as.integer(round(count))) * edge[4]/count
  }
  native_sample_edge(edge, depth, anc_state, edges, ncells, step_size, t_sample)
}, ns='scMultiSim')

# Use the declared graph schedule with native GRN normalization.
# .rnaSimEdge calls update immediately before simulating each cell.
assignInNamespace('.CreateDynGRN', function(grn, opts) {
  if (!identical(as.integer(grn$regulators), seq_len(ntf)) ||
      !identical(as.integer(grn$targets), (ntf+1L):ngene)) stop('Native TF/TG reindexing changed')
  dyngrn <- get('.dynGRN', asNamespace('scMultiSim'))$new(
    randseed=seed, involved_genes=seq_len(ngene), geff=grn$geff,
    params=grn$params, regulators=grn$regulators, targets=grn$targets,
    n_tgt=grn$n_tgt, n_reg=grn$n_reg, name_map=grn$name_map,
    del_edges=NULL, gen_edges=NULL, has_tf_edges=FALSE,
    max_steps=1L, remaining_steps=1L, remaining_cells=cells,
    cell_per_step=cells, n_edges=sum(initial), n_changing_edges=nrow(events)/(times-1),
    weight_mean=1, weight_sd=0, history=list(), version=1L)
  env <- as.environment(dyngrn)
  update <- function() {
    rank <- as.integer(env$version)
    if (rank > cells*times) stop('Unexpected extra native GRN update')
    w <- weights_at(context$position[context$order[rank]])
    g <- env$geff
    g[(ntf+1):ngene, 1:ntf] <- w
    env$geff <- g
    h <- env$history
    h[[rank]] <- g
    env$history <- h
    env$version <- rank+1L
    rank
  }
  if (exists('update', env, inherits=FALSE) && bindingIsLocked('update', env)) unlockBinding('update', env)
  assign('update', update, envir=env)
  dyngrn
}, ns='scMultiSim')
numbers <- get('.getNumbers', asNamespace('scMultiSim'))
assignInNamespace('.getNumbers', function(GRN, options) {
  n <- numbers(GRN, options); n$region <- nrow(motif); n
}, ns='scMultiSim')
assignInNamespace('.regionToGeneMatrix', function(seed, N, options) {
  result <- matrix(0, nrow(motif), N$gene)
  result[, (ntf+1):ngene] <- distance
  result
}, ns='scMultiSim')
assignInNamespace('.regionToTFMatrix', function(GRN, region_to_gene) {
  motif[, as.character(GRN$regulators), drop=FALSE]
}, ns='scMultiSim')
native_atac <- get('.atacSeq', asNamespace('scMultiSim'))
assignInNamespace('.atacSeq', function(seed, sim) {
  native_atac(seed, sim)
  # Group cells by the native .rnaSeq tree traversal.
  edges <- get('.tree_info', asNamespace('scMultiSim'))(sim$options$tree)$edges
  neutral <- sim$CIF_all$neutral[seq_len(sim$N$cell), , drop=FALSE]
  groups <- lapply(seq_len(nrow(edges)), function(i) {
    which(neutral[, 'from'] == edges[i,'from'] & neutral[, 'to'] == edges[i,'to'])
  })
  if (length(groups) != times || any(lengths(groups) != cells)) stop('Unexpected balanced tree segment sizes')
  order <- unlist(groups, use.names=FALSE)
  if (!identical(sort(order), seq_len(cells*times))) stop('Native traversal is not a cell partition')
  context$order <- order
  context$time <- integer(length(order))
  context$time[order] <- rep(0:(times-1), lengths(groups))
  context$position <- numeric(length(order))
  context$position[order] <- unlist(lapply(seq_along(groups), function(t) {
    (t-1L) + (seq_along(groups[[t]])-.5)/length(groups[[t]])-.5
  }), use.names=FALSE)
  # Evaluate potential RE groups with each cell's stage-specific binary GRN.
  # This also contributes to the native RNA kon calculation.
  activity <- t(vapply(context$position, function(position) {
    rowSums((distance %*% weights_at(position)) * motif) / rowSums(motif)
  }, numeric(nrow(motif))))
  sim$atac_data <- sim$atac_data * (.2 + .8*activity)
  stage_seeds <- get('seed', envir=parent.frame(), inherits=FALSE)
  write.table(data.frame(stage=seq_along(stage_seeds), seed=stage_seeds),
              file.path(audit,'stage_seeds.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
}, ns='scMultiSim')

idx <- which(potential, arr.ind=TRUE)
w0 <- weights_at(-.5)
grn <- data.frame(regulated.gene=idx[,1]+ntf, regulator.gene=idx[,2], regulator.effect=w0[idx])
node <- paste0('S', times-1L, ':1')
for (i in rev(seq_len(times-1L))) node <- paste0('(', node, ')S', i-1L, ':1')
tree <- read.tree(text=paste0('(', node, ');'))
# Define the stage tree for the five-stage simulation design.
if (times == 5L) tree <- read.tree(text='(((((A:1)B:1)C:1)D:1)E:1);')
options <- list(rand.seed=seed, GRN=grn, num.cells=cells*times, num.cifs=55,
  tree=tree, diff.cif.fraction=diff_cif, cif.sigma=cif_sigma,
  intrinsic.noise=intrinsic, do.velocity=TRUE, grn.effect=1,
  atac.effect=atac_effect, atac.p_zero=.3,
  dynamic.GRN=list(cell.per.step=cells, num.changing.edges=1, num.steps=1,
                   weight.mean=1, weight.sd=0, create.tf.edges=FALSE))
runner <- make_pinned_snapshot_runner(get('sim_true_counts', asNamespace('scMultiSim')), seed, RNGkind())
set.seed(seed)
result <- runner$run(options)
order <- context$order
if (length(result$cell_specific_grn) != cells*times || any(!is.finite(result$cell_time))) stop('Missing native kinetic GRNs/time')
rows <- lapply(seq_along(order), function(rank) {
  i <- order[rank]
  position <- context$position[i]
  actual <- as.matrix(result$cell_specific_grn[[i]])
  expected <- weights_at(position)
  observed <- actual[(ntf+1):ngene, 1:ntf, drop=FALSE]
  state <- context$time[i]
  outside <- actual
  outside[(ntf+1):ngene, 1:ntf] <- 0
  if (any(outside != 0)) stop('Native GRN acquired non-TF-TG effects')
  data.frame(cell=paste0('cell', i), time=state, traversal_rank=rank,
    trajectory_position=position, max_weight_error=max(abs(observed-expected)),
    matches_planned_strong_state=all((observed>.5) == planned[[state+1]]),
    nonbinary_weights=sum(!observed %in% c(0,1)),
    native_grn_sha256=digest::digest(actual, algo='sha256'))
})
checks <- do.call(rbind, rows)
if (any(checks$max_weight_error > 1e-12) || any(checks$nonbinary_weights != 0) || !all(checks$matches_planned_strong_state)) stop('Native generating GRNs differ from planned trajectory')
write.table(checks, file.path(audit,'per_cell_network_audit.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
metadata <- data.frame(cell=paste0('cell',order), time=context$time[order],
  traversal_rank=seq_along(order), trajectory_position=context$position[order],
  native_pop=as.character(result$cell_meta$pop[order]), cell_time=result$cell_time[order])
if (any(diff(metadata$cell_time) <= 0) || length(unique(metadata$native_pop)) != times) stop('Native developmental time/populations are not sequential')
write.table(metadata, file.path(folder,'cell_metadata.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
clean_rna <- as.matrix(result$counts)
clean_atac <- as.matrix(result$atac_counts)
if (nrow(clean_rna) < ngene || nrow(clean_atac) != nrow(motif) ||
    ncol(clean_rna) != cells*times || ncol(clean_atac) != cells*times) stop('Unexpected native assay axes')
set.seed(seed+10000L)
rna <- matrix(rpois(length(clean_rna), clean_rna*depth), nrow=nrow(clean_rna))
atac <- matrix(rpois(length(clean_atac), clean_atac*depth), nrow=nrow(clean_atac))
rownames(rna) <- c(as.character(seq_len(ngene)), paste0('background_', seq_len(nrow(rna)-ngene)))
rownames(atac) <- rownames(motif)
colnames(rna) <- colnames(atac) <- paste0('cell', seq_len(cells*times))
summary <- list()
for (t in 0:(times-1)) {
  take <- order[context$time[order] == t]
  write.csv(rna[,take,drop=FALSE], file.path(folder,sprintf('counts_time_%d.csv',t)), quote=FALSE)
  write.csv(atac[,take,drop=FALSE], file.path(folder,'multiome',sprintf('atac_time_%d.csv',t)), quote=FALSE)
  summary[[t+1]] <- data.frame(time=t, cells=length(take), native_pop=unique(metadata$native_pop[metadata$time==t]),
    RNA_mean=mean(rna[,take]), ATAC_mean=mean(atac[,take]), strong_edges=sum(planned[[t+1]]),
    kept_from_previous=if (t == 0) NA_real_ else sum(planned[[t+1]] & planned[[t]])/sum(planned[[t]]))
  cat(sprintf('[GENERATED] T%d RNA=%dx%d ATAC=%dx%d developmental_population=%s\n',
      t,nrow(rna),length(take),nrow(atac),length(take),summary[[t+1]]$native_pop))
}
write.table(do.call(rbind,summary),file.path(audit,'assay_summary.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
writeLines(c('RNA/ATAC counts generated with native scMultiSim kinetic samplers.',
  'Stage-specific candidate networks and region activities follow the configured trajectory.',
  'Training inputs contain observed counts and candidate regulatory relations.'), file.path(audit,'PROVENANCE.txt'))
writeLines(capture.output(sessionInfo()),file.path(audit,'session_info.txt'))

write.table(rbind(data.frame(package=c('scMultiSim','R'), version=c(as.character(packageVersion('scMultiSim')), as.character(getRversion()))), implementation), file.path(audit,'versions.tsv'), sep='\t', row.names=FALSE, quote=FALSE)
