#' TRexSelector: the T-Rex selector, its T-solvers and ML utilities
#'
#' @description
#' FDR-controlled variable selection with the Terminating-Random Experiments
#' (T-Rex) selector, the terminating solvers it runs on, and the machine
#' learning utilities around them. The exported symbols form four groups,
#' recognizable by their prefix (functions and constant lists) or their
#' leading group word (R6 classes):
#'
#' @section T-Rex selectors and controls (`TRex*Selector`, `trex_*`):
#' The R6 selectors [TRexSelector], [TRexDASelector], [TRexGVSSelector],
#' [TRexScreeningSelector], [TRexBiobankScreeningSelector], [TRexSPCASelector]
#' and [TRexTikhonovSelector]; their control lists [trex_control],
#' [trex_da_control], [trex_gvs_control], [trex_screen_control],
#' [trex_biobank_control], [trex_spca_control], [trex_tikhonov_control] and
#' [trex_dummy_distribution]; the helpers [trex_normalize_design] and
#' [trex_gamma_to_k]; the evaluation metrics [trex_fdp], [trex_tpp],
#' [trex_precision], [trex_recall], [trex_fdp_dense] and [trex_tpp_dense].
#'
#' @section T-solvers (`T*_Solver`, `tsolver_*`):
#' The terminating solvers as R6 classes, constructed as
#' `$new(X, D, y, [groups | tikhonov_K,] options)` with [tsolver_options]:
#' the LARS family ([TLARS_Solver], [TLASSO_Solver], [TSTEPWISE_Solver],
#' [TSTAGEWISE_Solver], [TENET_Solver], [TENETAug_Solver], [TIENET_Solver],
#' [TIENETAug_Solver], [TIKNET_Solver]), the OMP family ([TOMP_Solver],
#' [TGP_Solver], [TACGP_Solver], [TMP_Solver], [TOOLS_Solver],
#' [TNCGMP_Solver]), the AFS solver [TAFS_Solver], and the
#' coordinate-descent family ([TCDLASSO_Solver], [TCDENET_Solver],
#' [TCDIENET_Solver], [TCDTIKNET_Solver]). The solvers work on the caller's
#' matrices without copying and may preprocess them in place; the
#' writer/reader contract and `$restore()` are described in [tsolver_options]
#' and [TSolver_Base].
#'
#' @section ML methods (`ML*`, `ml_*`):
#' Scalers [MLZScoreScaler] and [MLLpNormScaler]; regressions [MLRidge],
#' [MLRidgeCV], [MLElasticNet] and [MLElasticNetCV]; decompositions [MLSVD]
#' and [MLPCA]; hierarchical clustering [ml_agglomerative_cluster] and
#' [ml_cut_tree] with the constants [ml_linkage_method] and
#' [ml_distance_metric].
#'
#' @section Memory-mapped matrices (`mem_*`):
#' [mem_matrix] opens or creates a disk-backed matrix and [mem_matrix_from]
#' writes an in-memory matrix to one; every selector, solver and ML method
#' accepts the result in place of a matrix.
#'
#' @keywords internal
"_PACKAGE"
