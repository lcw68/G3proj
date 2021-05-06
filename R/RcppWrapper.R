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
#' N  = 100;
#' p  = 10;
#'
#' X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
#' beta_true = c(rep(1,p/2),rep(0,p/2))
#' eta = X %*% beta_true
#' pi = exp(eta) / (1 + exp(eta))
#' Y  = rbinom(N,1,pi)
#'
#' ## Compute optimal proposal SD;
#' test1 <- BLRM.Tuning(Y0 = Y, X0 = X, PriorVar = 1000,
#'                      nMC = 100, b = 10)
#' @useDynLib G3proj
#' @export
BLRM.Tuning <- function(Y0, X0, PriorVar, nMC = 10000L, b = 50L, seed = 1L) {
  if(length(Y0) != dim(X0)[1]){
    stop("Y and X have incompatible dimensions")
  }

  BLRM_Tuning(Y0, X0, PriorVar, nMC, b, seed)
}


#' Bayesian Logistic Regression Model (BLRM) training
#'
#' Performs Bayesian Logistic Regression Model training by sampling beta
#' from posterior distribution with user specified parameters and data
#'
#' @param Y0 vector of responses
#' @param X0 covariate matrix
#' @param PriorVar variance of prior distribution of beta
#' @param propSD0 vector of standard deviations for normal proposal density
#' @param nMC number of MCMC samples
#' @param nBI number of burn-in samples
#' @param thin number of samples to skip over in thinning
#' @param seed set seed for random number generation
#'
#' @return a nested list of beta samples, and beta acceptance rates
#'
#' @examples
#'
#' ## simulate data;
#'
#' set.seed(1);
#' N  = 100;
#' p  = 10;
#'
#' X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
#' beta_true = c(rep(1,p/2),rep(0,p/2))
#' eta = X %*% beta_true
#' pi = exp(eta) / (1 + exp(eta))
#' Y  = rbinom(N,1,pi)
#' propSD = rep(1,p)
#'
#' ## fit model;
#' test1 <- G3proj::BLRM.fit.mwg(Y0 = Y, X0 = X, PriorVar = 1000, propSD0 = propSD,
#'                       nMC = 500, nBI = 100, thin = 5)
#' @export
BLRM.fit.mwg <- function(Y0, X0, PriorVar, propSD0,
                         nMC = 1000L, nBI = 250L, thin = 5L, seed = 1L) {
  if(length(Y0) != dim(X0)[1]){
    stop("Y and X have incompatible dimensions")
  }
  if(dim(X0)[2] != length(propSD0)){
    stop("Must specify proposal standard deviation for each column of X")
  }

  BLRM_fit_mwg(Y0, X0, PriorVar, propSD0, nMC, nBI, thin, seed)
}

#' MH Random Walk Proposal Tuning for SSVS under Logistic Regression Model
#'
#' Tunes the normal proposal distribution standard deviation for Metropolis Hastings MCMC for the SSVS variable selection method in a Logistic Regression model.
#'
#' Runs nMc/b number of batches of size b. In each batch the algorithm calculates the Metropolis acceptance rate and adjusts the proposal standard deviation accordingly. As such, nMC must be a multiple of b.
#'
#' @param Y0 vector of responses
#' @param X0 covariate matrix without intercept
#' @param c0 parameter for spike and slab prior of beta
#' @param tau0 parameter for spike and slab prior of beta
#' @param nMC number of MCMC samples
#' @param b batch size
#' @param seed set seed for random number generation
#'
#' @return A nested list of gamma samples, beta samples, beta acceptance rates for each batch, and the proposal standard deviations at the final batch.
#'
#' @examples
#'
#' ## simulate data;
#'
#' set.seed(1);
#' N  = 100;
#' p  = 10;
#'
#' X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
#' beta_true = c(rep(1,p/2),rep(0,p/2))
#' eta = X %*% beta_true
#' pi = exp(eta) / (1 + exp(eta))
#' Y  = rbinom(N,1,pi)
#'
#' ## fit model;
#' test1 <- G3proj::SSVS.Tuning(Y0 = Y, X0 = X, c0 = 10,
#'                              tau0 = 0.4, nMC = 1000, b = 50)
#' @export
SSVS.Tuning <- function(Y0, X0, c0, tau0, nMC = 1000L, b = 50L, seed = 1L) {
  if(length(Y0) != dim(X0)[1]){
    stop("Y and X have incompatible dimensions")
  }

  SSVS_Tuning(Y0, X0, c0, tau0, nMC, b, seed)
}

#' Stochastic Search Variable Selection (SSVS) for Logistic Regression
#'
#' Performs SSVS for a logistic regression model with user specified parameters and data
#'
#' @param Y0 vector of responses
#' @param X0 covariate matrix without intercept
#' @param propSD0 vector of standard deviations for normal proposal density
#' @param c0 parameter for spike and slab prior of beta
#' @param tau0 parameter for spike and slab prior of beta
#' @param nMC number of MCMC samples
#' @param nBI number of burn-in samples
#' @param thin number of samples to skip over in thinning
#' @param seed set seed for random number generation
#'
#' @return a nested list of gamma samples, beta samples, and beta acceptance rates
#'
#' @examples
#'
#' ## simulate data
#'
#' set.seed(1)
#' N  = 100
#' p  = 10
#'
#' X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
#' beta_true = c(rep(1,p/2),rep(0,p/2))
#' eta = X %*% beta_true
#' pi = exp(eta) / (1 + exp(eta))
#' Y  = rbinom(N,1,pi)
#' propSD0 = rep(.5,p)
#'
#' ## fit model;
#' test1 <- G3proj::SSVS.Logistic(Y0 = Y, X0 = X, propSD0, c0 = 10,
#'                                tau0 = 0.4, nMC = 1000, nBI = 100, thin=1, seed=1)
#' @export
SSVS.Logistic <- function(Y0, X0, propSD0, c0, tau0, nMC = 1000L, nBI = 250L, thin = 5L, seed = 1L){
  if(length(Y0) != dim(X0)[1]){
    stop("Y and X have incompatible dimensions")
  }
  if(dim(X0)[2] != length(propSD0)){
    stop("Must specify proposal standard deviation for each column of X")
  }

  SSVS_Logistic(Y0, X0, propSD0, c0, tau0, nMC, nBI, thin, seed)
}
