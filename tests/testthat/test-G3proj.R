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
  test1 <- G3proj::SSVS.Logistic(Y0 = Y, X0 = X, propSD0 = rep(.5,p), c0 = 10,
                              tau0 = 0.4, nMC = 500, nBI = 50, seed=1)
  expect_equal(colMeans(test1$gamma.samples),c(0.630, 0.532, 0.082, 0.100))
})

test_that("SSVS Logistic gives error for incorrect dimensions",{
  set.seed(1)
  N  = 800
  p  = 4
  X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
  beta_true = c(rep(1,p/2),rep(0,p/2))
  eta = X %*% beta_true
  pi = exp(eta) / (1 + exp(eta))
  Y  = rbinom(N,1,pi)

  ## fit model with extra dimension on Y
  expect_error(G3proj::SSVS.Logistic(Y0 = c(Y,1), X0 = X, propSD0 = rep(.5,p), c0 = 10,
                                 tau0 = 0.4, nMC = 500, nBI = 50, seed=1))
  ## fit model with wrong number of proposal standard deviations
  expect_error(G3proj::SSVS.Logistic(Y0 = Y, X0 = X, propSD0 = 1, c0 = 10,
                                     tau0 = 0.4, nMC = 500, nBI = 50, seed=1))
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
  test1 <- G3proj::SSVS.Tuning(Y0 = Y, X0 = X, c0 = 10,
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
  test1 <- G3proj::BLRM.fit.mwg(Y0 = Y, X0 = X, PriorVar = 1000, propSD0 = propSD,
                                nMC = 500, nBI = 50, seed = 1)

  expect_equal(as.vector(test1$beta.act.rate), c(0.1276, 0.1188, 0.1044, 0.1068))
})

test_that("BLRM fitting gives error for incorrect dimensions",{
  set.seed(1)
  N  = 800
  p  = 4
  X  = matrix(data = rnorm(N*p), nrow=N, ncol=p)
  beta_true = c(rep(1,p/2),rep(0,p/2))
  eta = X %*% beta_true
  pi = exp(eta) / (1 + exp(eta))
  Y  = rbinom(N,1,pi)

  ## fit model with extra dimension on Y
  expect_error(G3proj::BLRM.fit.mwg(Y0 = c(Y,1), X0 = X, PriorVar = 1000, propSD0 = rep(.5,p),
                                    nMC = 500, nBI = 50, seed=1))
  ## fit model with wrong number of proposal standard deviations
  expect_error(G3proj::BLRM.fit.mwg(Y0 = Y, X0 = X, PriorVar = 1000, propSD0 = c(1),
                                    nMC = 500, nBI = 50, seed=1))
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
  test1 <- G3proj::BLRM.Tuning(Y0 = Y, X0 = X, PriorVar = 1000,
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
  fit <- G3proj::BLRM.fit.mwg(Y0 = Y.train, X0 = X.train,
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



test_that("glmlasso gives error for bad input",{
  set.seed(1232)
  Nz = 500
  pz = 10
  Xz = scale(matrix(rnorm(Nz*pz), ncol=pz))
  bz = c(.5, -.5, .25, -.25, .125, -.125, rep(0, pz-6))
  yz = rbinom(Nz+1,1,exp(Xz %*% bz)/(1+exp(Xz %*% bz)))
  yz1 = factor(rbinom(Nz,2,exp(Xz %*% bz)/(1+exp(Xz %*% bz))))
  lambda = .1
  expect_error(glmlasso(Xz,yz,lambda,tol=1e-12))
  expect_error(glmlasso(Xz,yz1,lambda,tol=1e-12))

})

test_that("GLM lasso result consistent with glmnet",{
  set.seed(1232)
  Nz = 500
  pz = 10
  Xz = scale(matrix(rnorm(Nz*pz), ncol=pz))
  bz = c(.5, -.5, .25, -.25, .125, -.125, rep(0, pz-6))
  yz = rbinom(Nz,1,exp(Xz %*% bz)/(1+exp(Xz %*% bz)))
  lambda = .1
  lambda = .1
  fit <- glmnet::glmnet(Xz,yz,family="binomial",lambda = 0.1,intercept = FALSE)
  fit0 = coef(fit,s = lambda)[-1,1]
  names(fit0) = NULL
  fit1 <- glmlasso(Xz,yz,lambda,tol=1e-12)
  expect_equal(fit1[,1],fit0,tolerance = 0.005)

})

test_that("optim.lambda gives error for bad input",{
  set.seed(1232)
  Nz = 500
  pz = 10
  Xz = scale(matrix(rnorm(Nz*pz), ncol=pz))
  bz = c(.5, -.5, .25, -.25, .125, -.125, rep(0, pz-6))
  yz = rbinom(Nz,1,exp(Xz %*% bz)/(1+exp(Xz %*% bz)))
  yz1 = factor(rbinom(Nz,2,exp(Xz %*% bz)/(1+exp(Xz %*% bz))))

  expect_error(optim.lambda(Xz,yz,lambda.min = 0.1,lambda.max = 0.05,len = 200))
  expect_error(optim.lambda(Xz,yz1,lambda.min = 0.01,lambda.max = 0.05,len = 200))

})

test_that("the seleted lambda close to glmnetBIC", {
  library(glmnet)
  set.seed(8675309)
  Nz = 500
  pz = 10
  Xz = scale(matrix(rnorm(Nz*pz), ncol=pz))
  bz = c(.5, -.5, .25, -.25, .125, -.125, rep(0, pz-6))
  yz = rbinom(Nz,1,exp(Xz %*% bz)/(1+exp(Xz %*% bz)))

  lambda1 <- optim.lambda(Xz,yz,lambda.min = 0,lambda.max = 0.1,len = 200)[[1]]
  fit <- glmnet(Xz, yz,lambda = seq(0.1,0,length=200), family = "binomial")
  tLL <- fit$nulldev - deviance(fit)
  k <- fit$df
  n <- fit$nobs
  BIC<-log(n)*k - tLL
  min_BIC=which(BIC==min(BIC))
  lambda_expected <- seq(0.1,0,length=200)[min_BIC]
  expect_equal(lambda1,lambda_expected, tolerance = 0.05)
})

