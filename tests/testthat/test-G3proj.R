test_that("SSVS Logistic gives correct result",{

  set.seed(1)
  N  = 800
  p  = 4
  X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
  beta_true = c(rep(1,p/2),rep(0,p/2))
  eta = X %*% beta_true
  pi = exp(eta) / (1 + exp(eta))
  Y  = rbinom(N,1,pi)

  ## fit model;
  test1 <- G3proj::SSVS_Logistic(Y0 = Y, X0 = X, propSD0 = rep(.5,p), c0 = 10,
                              tau0 = 0.4, nMC = 500, nBI = 50, seed=1)
  expect_equal(colMeans(test1$gamma.samples),c(0.630, 0.532, 0.082, 0.100))
})

test_that("SSVS Tuning gives correct result",{
  set.seed(1)
  N  = 800
  p  = 4
  X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
  beta_true = c(rep(1,p/2),rep(0,p/2))
  eta = X %*% beta_true
  pi = exp(eta) / (1 + exp(eta))
  Y  = rbinom(N,1,pi)

  ## fit model;
  test1 <- G3proj::SSVS_Tuning(Y0 = Y, X0 = X, c0 = 10,
                                 tau0 = 0.4, nMC = 500, b = 50, seed=1)

  expect_equal(colMeans(test1$gamma.samples),c(0.622, 0.542, 0.102, 0.070))
})

test_that("BLRM fitting gives correct result",{
  set.seed(1)
  N  = 800
  p  = 4
  X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
  beta_true = c(rep(1,p/2),rep(0,p/2))
  eta = X %*% beta_true
  pi = exp(eta) / (1 + exp(eta))
  Y  = rbinom(N,1,pi)
  propSD = rep(1,p)

  ## fit model;
  test1 <- G3proj::BLRM_fit_mwg(Y0 = Y, X0 = X, PriorVar = 1000, propSD0 = propSD,
                                nMC = 500, nBI = 50, seed = 1)
  test1$beta.act.rate

  expect_equal(test1$beta.act.rate, c(0.1276, 0.1188, 0.1044, 0.1068))
})

test_that("BLRM Tuning gives correct result",{
  set.seed(1)
  N  = 800
  p  = 4
  X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
  beta_true = c(rep(1,p/2),rep(0,p/2))
  eta = X %*% beta_true
  pi = exp(eta) / (1 + exp(eta))
  Y  = rbinom(N,1,pi)

  ## fit model;
  test1 <- G3proj::BLRM_Tuning(Y0 = Y, X0 = X, PriorVar = 1000,
                                nMC = 500, b = 50, seed=1)

  expect_equal(colMeans(test1$prop.sd.batch),c(0.945, 0.945, 0.945, 0.945))
})

