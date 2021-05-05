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

test_that("BLRM Prediction gives correct result",{

  ## simulate data
  set.seed(1)
  N.train = 800
  N.test  = 800
  p = 4
  beta_true = c(rep(1,p/2),rep(0,p/2))

  X.train   = matrix(data = rnorm(N.train*p), nrow=N.train, ncol=p)
  eta.train = X.train %*% beta_true
  pi.train  = exp(eta.train) / (1 + exp(eta.train))
  Y.train   = rbinom(N.train,1,pi.train)

  X.test   = matrix(data = rnorm(N.test*p), nrow=N.test, ncol=p)
  eta.test = X.test %*% beta_true
  pi.test  = exp(eta.test) / (1 + exp(eta.test))
  Y.test   = rbinom(N.test,1,pi.test)

  ## Fit BLRM and get MCMC Beta samples
  fit <- G3proj::BLRM_fit_mwg(Y0 = Y.train, X0 = X.train,
                              PriorVar = 1000, propSD0 = rep(.5,p),
                              nMC = 500, nBI = 50, seed = 1)

  ## predict based on MCMC beta samples
  prediction = G3proj::predict_BLRM(Y.test = Y.test, X.test = X.test, fit$beta.samples)

  ## Classification Matrix with cutoff = 0.5
  table = caret::confusionMatrix(data = factor(prediction$data),
                                  reference = factor(ifelse(prediction$pred.prob > 0.5, 1, 0)),
                                  positive = "1")$table
  table_true =as.table(matrix(c(330, 102, 94, 274), nrow = 2,
                              dimnames = list(Prediction = c(0,1), Reference = c(0,1))))
  all.equal(table, table_true)
  expect_equal(table, table_true)
})
