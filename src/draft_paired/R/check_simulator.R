# Verify scMultiSim function fingerprints against the Bioconductor 3.20 source.
scmultisim_source_sha256 <- '15ddbe83503bc1af1b408ddc3e506662888712c0bf3cdec6d1a5cfa60bf91052'
scmultisim_expected_functions <- c(
  ".default" = "24a3d3d1af8a208dc3c71439031428ed42b85a58e6a375e0a15188d47bc34b3e",
  ".should.be.int.between" = "90fa4113156319eeafc27ff927e982b81e6b08bad8562d270aa515d7e51c41c3",
  ".should.be.num.between" = "8ce26b6d21a9840968ecb796738e830e4abadca4cfa778ee220f5a014c72ab16",
  ".choose_from" = "e0d11338b1116a169e26c74ec737998c80cd2a4e04d746dc5519a2347f988c98",
  ".opt_list" = "8b9f880d910c245931b90f6eb2cd9e38d1ce613ad936ba355db2894f0d352199",
  ".check_opt" = "f300b7881f446ce16ceaaee008c8ce5a29440df955abc111da36f2184c4f9242",
  ".split_long_string" = "31675116ff1437813dc670dcf67202c619b52f310307e18099ad0a119d56ecb7",
  ".print_opt" = "176f35eb51213adcaa12a86c69c30abc186b4dcd56ebf59fd79add517a9941d7",
  "OP" = "254cbef323f90ef9fb36d9a819d044cd777f0151a1e74c2159e19ff5e0590f0c",
  ".dynamic_grn_default_params" = "2ce62a3fbdbe5eb15e3f2323939e2fbf95ebaaf8e09b569a70a47ac5a33f05e1",
  ".continuousCIFParam" = "57cfbbaf92ede813c0a7586da1b02b054e4f8f1d1c222a795a65d77a3b742bea",
  ".continuousCIFParamSpatial" = "fbea5292bcf1ae6f40d422ca7d1bcb43b55c47f2fa0a5554720fd41838836999",
  ".continuousCIFParamNormal" = "712078000d3739d34bf7ddfc694f1a6fb4049c42271283090c1d70fb389e214f",
  ".discreteCIFSpatial" = "b707a443ecbee5cb951063608a90ed8bf404167149a2ee4c11681626a657df83",
  "SampleEdge" = "db35094434c41d4638eb9c5b66784d0bf5a29b1a7b086dfbd238f79b89e6ba33",
  "SampleSubtree" = "be6cdb295459f1586794255a82f30aa0bd1ed1856894fde13b84520ae2583480",
  "sim_true_counts" = "0dc7047b362342d2a7384cba67a53f97757900734e0ee40a8d84f2249d9b6db4",
  ".getResult" = "69e38b8252483dd4d043b416baacad4a5ca102ac5f2472f4f4cac7e9b3e47832",
  ".renameGenes" = "bba07961174283c58ac799c185966b91f96de16becc31390b5cd2032a5981a9c",
  ".normalizeGRNParams" = "e8cc02cf0e5f86ea317becc04ae231174f750d7cb1cb9c8a3c3959ef270e8c4c",
  ".getNumbers" = "b14501de62bb52f3339120f5769729a32aff6a280aeedb70a4761b06b049b9bf",
  ".discreteCIF" = "7e073400c85dde4c9f53270e37dbf9a12a70059fbdb095c3d54b846b152402d6",
  ".continuousCIF" = "325246e25fedcd893abfa77c562b5ebd2c310ece686ad1bc1a8049e5f04f173d",
  ".geneEffectsByRegulator" = "45d047fc0a9f2beed077aade54fde641615f0daf3fe01452a46b8d036d575bbd",
  ".identityVectors" = "d270d6b8b7fb4f83abc04bdec1bc11d67078ca40b0e4e11071144310f5e3cedd",
  ".geneIdentifyVectors" = "9c74b5a585280a774fb6f77456479dc1db744d8b04b55060915e3e1c048e3159",
  ".regionIdentityVectors" = "1a49007b34526fd80150b77bc8aeff41e07e4571a05e78ace44a4449078df63a",
  ".regionToGeneMatrix" = "c5d103835544186d7d7d38b0e53e1ffd3b72b1b35d3d49994c525c918889fa02",
  ".summarizeExp" = "c8638508c57cc7486af2365b9331e6528ce423aeaecdb0094d27a844ed3ba149",
  ".matchParamsDen" = "22f2dbbe13e0161507931d4645c080358dcbabe776bb495061b4fd3540f7ebc1",
  ".getParams" = "d2b703f8ef897d1831294b89fab49d6bd5e83155ff3578da6b2a897d8761931d",
  ".prepareHGE" = "d02644825b3198599f4534a522fec1a1381db31b6ad5cd1072490f202f9d8bdf",
  ".atacSeq" = "8eb165444e4cc37e36ced435c0fe341a755b641a0bd44f0c88cb14ce6417a02f",
  ".rnaSeq" = "cdbf1af7db300775bacc080067dc244cbbfc03590b90d2e743ba381597607d06",
  ".rnaSimEdge" = "4b24de51c138a163115cadd70fcf92f0e6a84e66d48178e7833554d2408985f1",
  ".setSpatialFinalCellType" = "a31e64a7700d40e75d012cb26b16056ad0c226bfac30def2a517b05ce98ef654",
  ".getExtraCIF" = "c6e7af1180ab0c68e60b7326d128cc8f2e0361880e0abb6e4fc205d1d3ff9fc0",
  ".rnaSeqSpatial" = "967c2996c69d9be600c753c22c79164d5cbbb1437fdeb65b61ac12f50b66703d",
  "gen_1branch" = "ff96db99a796045e279cecf5b91b66c9519351d88593b0084c89bf9de6d3b359",
  ".betaPoisson" = "3805615714826b57ef164014b3749d3a05e0edcfda3c64a01eaccef9248b9569",
  ".betaPoisson.vec" = "88eed0afb165d911daaed76ecb5a43bd76bbb77585936e37053518d13d0bee5e",
  ".atacIntrNoise" = "580c3666653e8ed016762420c93b316ca958021521c4252aba8345c3d154793e",
  ".parseSpatialParams" = "8754c7da7b082270392a95e08673c06f13e5edc7a9198a62c913b2d2f052983b",
  ".getPaths" = "1edcc46937dedf818ddb0d40c4ed3012eff619749694b2b8164a30826da783bf",
  ".getPathLen" = "a1feb9473dc57c52aea28dcf3614ddb176b6a2042d2342f6cdb9a762bc683eaf",
  ".cellTypeParamToMatrix" = "3d6df3d0f59f5838b46ca9fec25dc0389831d8d1b11e48e9b37138ddab08674c",
  "cci_cell_type_params" = "efbebc5c82ec020c744373a891d0b2de662db2a3ab9412fe9c7ff8bd76ca649a",
  "gen_clutter" = "1f85f18cd3a54f6cc15e27e12f25725b04488fccc92fc871a923d265892d5403",
  ".samplePointsGaussianKernel" = "e3d8ec7c9f2963faf297adb84437ba1d71ef4c7215b82408805e77150c6c36c8",
  "CreateSpatialGrid" = "eec6b61f1f93ed32fbf4f7e87d74ad86978904621f945aabbe601e1fd1e8cc11",
  ".CreateDynGRN" = "6abfb4aa2d2c020ef824a1137da4f48cbe3582514afe9c2a88885d06110fd00e",
  ".getDynGRNOpts" = "b1a6e36ce19c30de682f320c74930c93101370d4f0d544a6f90a4e7fc199471e",
  "add_expr_noise" = "dbf828cf66b3d7721026caba1add562fcb6a6660988f4077f79ed8c4e816fb38",
  "divide_batches" = "9889312f6387167df3799484d3bc16943da374b934209906e840f0a9e782931a",
  ".divideBatchesImpl" = "8c79e7d10e82e227a25fee914246255da3bd8da85b76532711184488944566ae",
  ".amplifyOneCell" = "3d62d27a96805c038f783fcf6b246471439847503045a13d05b1cc165d488b59",
  "True2ObservedCounts" = "b0d086304f24adc14895bf2cbfbcffd498db548b9e13b4af9189811e4d02dd62",
  "True2ObservedATAC" = "9b062af4ba17d21d73276a7e78a6da7dcd1a3886c5fccf540283e82afcd4cdac",
  ".calAmpBias" = "fd1e67c3ee238e66f7533d51fcd324628c848998d5d68e276b49e5c6401082c3",
  ".expandToBinary" = "074c9dff017cb96b2dd4badf16ebc07a14953afa414cc739fb95837b2b44b88d",
  ".rnormTrunc" = "7df68d0127cf7204991077539afcb1dc9b05ff3f3dd91505d7857dc24f2a3c4c",
  ".getProb" = "2124b36d1b2826fc46f13466012cf7463c6abdb64f252469b0f9078239584414",
  "add_outliers" = "749e05782bf5618929fcd3bec14a9a8a37b0b9e5743a4335d37bed06770f9013",
  ".getResultsFromGlobal" = "4a089dfc5763e65cb4e57a6dc2a09b5481c7313992ebb75fb4e4d88b4777864a",
  "plot_phyla" = "053430cf2c84af3401645ce521c7931c84994343addd39eb5e4d2e6451dfc9ee",
  ".plotHist" = "74c748992aa3c4f30ece94cec8a0faacb842f8eb9a67735178dbcea6da2e554c",
  "plot_tsne" = "cf9ec00f2712abce06028e2b1e8c2edfdc257932daa14870f6c2d6bdb2775f32",
  "plot_grid" = "2bc0c0da49a0081ac7f32e0cbb16061e166a5e1f6d4c232e046814076c2fe143",
  ".getGeneModuleColors" = "d8f1cb43647e0012116d068428e8c9cb9b0cb6fc84a941743ac95af8b3e1e403",
  "plot_gene_module_cor_heatmap" = "2be62dd2f9c8676d06e9af630cb8dfec0d9dde678da2599056da52d7c50a8066",
  "plot_cell_loc" = "80741e900f889789f622ac02c1e898b8d1bcfbee78c5b94552fbdc1edf533566",
  "plot_grn" = "9458a695e64bfc36937262021096436698dfb7cfe1206b958e6a892ceeb12f22",
  "gene_corr_regulator" = "41307eb86818920ae75fd4037b2d2ea342d2901c6625b3ce337e329bd8d04a67",
  ".geneCorrGRN" = "9d191fa5a3841317b3a0d3e51fa199829fcfdbcbe61075f807e9ef1011bf1b93",
  ".geneCorr" = "75e28f0a911687dc1f1b758477b90db116064aeff0bb445a17ad28469336399d",
  "gene_corr_cci" = "ffaeb581f35cf0af7297ab77fbca5ff30c56f2ffe64eff383aaabc0e5317aecd",
  "gene_coexpr_cci" = "658bfee3c7619a9673ebeb5ba4038b22eb0e681863639ad34071980517cfb9c4",
  ".processVelocity" = "2dadb514f6f9fbab8a956ca1bfe74d6ed7ce824fda0313bd531ec040e0023a75",
  "plot_rna_velocity" = "aa89f06f59a3a65ac06731620e6f4c53a2448352966eba20da0ceef1ea3d1af6",
  ".cosineSim" = "b214024f2a90bda85c27eee0a1bf15238f71e45f40f205f7e33b34070ea0ebc7",
  ".rnaVelocityCosine" = "3bc204d940b5614d40464f335dfdbb9a7672ab3d36a07fbe29810a8dfaba4b48",
  ".getCountCorrMatrix" = "fbf5970a69846c710b538722e0a97165340f1d7b8647b6053509a534136d66e6",
  "Get_1region_ATAC_correlation" = "3c5eeb0c366caebf47f34eed5a76f1f4c80ee4a44a8a627758d51f1d5f20e2a7",
  "Get_ATAC_correlation" = "0009d9d335204ed4af272dfee1ce7033b984ed88ca41c1d4ef684849eabf7a82",
  "%+%" = "74f09f3942be6b933eb76ada34c9e3e5479e940b4cba82cba3960564a31e2d64",
  ".defaultArgs" = "5e8c80484cee0321b10946ccbe9fc17cbbf22ab823cf263e0150cbe46032e922",
  ".regionToTFMatrix" = "904af2b6f0e19ca4e15e14e63cc53ee5b8851d13364e7a42f76256c2ed676d69",
  "SampleDen" = "3342e74365a410e321d5ad59f49f6ec83a25d6fff387a83a33f4e15376e6a5cb",
  "Phyla5" = "767709e0b65cc3088448a5d379a992f651d2ae3c6eeed339589cb53561b33ee3",
  "Phyla3" = "0d4a5f8029135a1d74c9e6103923342344520f425fb5d4c887ef0a099679717c",
  "Phyla1" = "bee419070c63d42505abbb9956d52e7415f0012954238128d9e61546baecdaf6",
  ".tree_info" = "531a91e0605bbdc1a94e8cb0d0fd54d664e4bc9c22c8d6801c9f1e4c582f70d3",
  ".print_param_summary" = "a43d27d4b20aa67a119395b1235e980c42465a56a65df358df31a7e475c94e18",
  ".print_matrix_dim" = "fb702682afca23fa0c7aabceb0dc273a7eb38a0860a5e7a653a6ca296fc47e84",
  ".print_time" = "f1e92bf72812903674a7d38edeef58adbd820b1c58215b0fe8ec8b3fac77c69f",
  ".print_gene_in_grn" = "f5728ba3a219b1918ba64603f3d15106b96719f74b072f08362dc9e1b2cea8c4",
  "sim_example" = "35082fcfe2906169aef47c0d30143cdc6f7b93b91334fd4dcac14dc2186ce850",
  "sim_example_spatial" = "b5b34c4b2d0134fff75f9f4e1216da1a05b8642d51f9f28e1d0ef362cc6fd9a9",
  "atac_dens_nonzero" = "bb6a57db45a50476505e7fbae4ca2230e55b20a41e7cd4a3ac7e19ecaece2488",
  "run_shiny" = "83baca992c06ee3282b0e02717b94b7b53c183733567874a0de0937188cbac00",
  "generateSpatialLoc" = "797b07764ecb295b82e99f0429f9b39f3ed2a11160a3428f0cd9e635a01d2b11",
  "scmultisim_help" = "fabff23c3768875a33985349f5be57374334dffccb475683cfc2146e307dc8bf",
  ".cci_help" = "86b958283e8a538e028a6d6c5679912685e83e5c04bc235c518b98e8b7960d50",
  "rna_velo_knn" = "add3331e018f391cc9dedcfc8eb1b17d5aef74fb78593316651edd87c8628899"
)

verify_scmultisim_implementation <- function(audit=NULL) {
  if (length(scmultisim_expected_functions) != 104L) stop('Damaged scMultiSim function fingerprint manifest.')
  if (!requireNamespace('scMultiSim',quietly=TRUE) ||
      as.character(packageVersion('scMultiSim')) != '1.2.0') {
    stop('Requires Bioconductor 3.20 scMultiSim 1.2.0; run scripts/install_simulation_r.R.')
  }
  ns <- asNamespace('scMultiSim')
  hashes <- vapply(names(scmultisim_expected_functions),function(name) {
    if (!exists(name,ns,inherits=FALSE) || !is.function(get(name,ns))) return('missing')
    code <- paste(deparse(get(name,ns),width.cutoff=500L),collapse='\n')
    digest::digest(code,algo='sha256',serialize=FALSE)
  },character(1))
  checks <- data.frame(name=names(hashes),sha256=unname(hashes),
                       expected_sha256=unname(scmultisim_expected_functions),
                       matches=unname(hashes==scmultisim_expected_functions))
  if (any(!checks$matches)) {
    stop(paste0('scMultiSim implementation mismatch (',sum(!checks$matches),
      ' native functions differ). Requires the official Bioconductor 3.20 scMultiSim 1.2.0 source; ',
      'the GitHub implementation also reports 1.2.0 but uses a different random-number process. ',
      'Run scripts/install_simulation_r.R into the selected R library.'))
  }
  if (!is.null(audit)) {
    write.table(checks,file.path(audit,'native_function_checks.tsv'),sep='\t',row.names=FALSE,quote=FALSE)
    writeLines(c(paste('library',find.package('scMultiSim')),
                 paste('RNGkind',paste(RNGkind(),collapse='/')),
                 capture.output(packageDescription('scMultiSim'))),
               file.path(audit,'simulator_implementation.txt'))
  }
  data.frame(package=c('scMultiSim_source','scMultiSim_source_sha256'),
             version=c('Bioconductor_3.20/1.2.0',scmultisim_source_sha256))
}
