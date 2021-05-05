#' @useDynLib G3proj

#' Bayesian Logistic Regression Model (BLRM) Tuning for Proposal Standard Deviation
#'
#' Performs hyper parameter tuning of proposal standard deviation
#' that can be used in Bayesian Logistic Regression Model training
#' with user specified parameters and data
#'
#' Runs nMc/b number of batches of size b. In each batch the algorithm calculates
#' the Metropolis acceptance rate and adjusts the proposal standard deviation
#' accordingly. As such, nMC must be a multiple of b.
#'
#' @param Y0 vector of responses
#' @param X0 covariate matrix
#' @param PriorVar variance of prior distribution of beta
#' @param nMC number of MCMC samples, should be the multiple of b
#' @param b batch size
#' @param seed set seed for random number generation
#'
#' @return a nested list of beta samples, beta acceptance rates and proposal SD for each batch
#'
#' @examples
#'
#' ## simulate data;
#'
#' set.seed(1);
#' N  = 8000;
#' p  = 10;
#'
#' X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
#' beta_true = c(rep(1,p/2),rep(0,p/2))
#' eta = X %*% beta_true
#' pi = exp(eta) / (1 + exp(eta))
#' Y  = rbinom(N,1,pi)
#'
#' ## Compute optimal proposal SD;
#' test1 <- BLRM_Tuning(Y0 = Y, X0 = X, PriorVar = 1000,
#'                      nMC = 10000, b = 50)
#' @export
BLRM.Tuning <- function(Y0, X0, PriorVar, nMC = 10000L, b = 50L, seed = 1L) {
  BLRM_Tuning(Y0, X0, PriorVar, nMC = 10000L, b = 50L, seed = 1L)
}

#' @export
BLRM.fit.mwg <- function(Y0, X0, PriorVar, propSD0, nMC = 1000L, nBI = 250L, thin = 5L, seed = 1L) {
  BLRM_fit_mwg(Y0, X0, PriorVar, propSD0, nMC = 1000L, nBI = 250L, thin = 5L, seed = 1L)
}

#' @export
SSVS.Tuning <- function(Y0, X0, c0, tau0, nMC = 1000L, b = 50L, seed = 1L) {
  SSVS_Tuning(Y0, X0, c0, tau0, nMC = 1000L, b = 50L, seed = 1L)
}

#' @export
SSVS.Logistic <- function(Y0, X0, propSD0, c0, tau0, nMC = 1000L, nBI = 250L, thin = 5L, seed = 1L){
  SSVS_Logistic(Y0, X0, propSD0, c0, tau0, nMC = 1000L, nBI = 250L, thin = 5L, seed = 1L)
}
