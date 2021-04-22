#' Soft thresholding calculation
#'
#'
#' @param x a numeric vector
#' @param b threshold values
#'
#' @return a numeric vector with the same length as x after soft thresholding
#'
#' @examples
#' set.seed(1)
#' x <- rnorm(5)
#' b <- 0.2
#' soft_thresh(x,b)
#'
#'
#'
#' @export
#'
soft_thresh <- function(x, b) {
  out = rep(0, length(x))
  out[x >  b] = x[x > b] - b
  out[x < -b] = x[x < -b] + b
  out
}



#' z-value calculation
#'
#'
#' @param beta numeric vector
#' @param x design matrix
#' @param y binary outcome
#' @param W weighted matrix
#'
#' @return response value
#' @noRd

getzed <- function(beta,x,y,W)
{
  p0 = exp(x%*%beta)/(1+exp(x%*%beta))
  z = x%*%beta + (y-p0)/W
  z
}


#' leave-one-out prediction calculation
#'
#'
#' @param beta numeric vector
#' @param dim index for which coordinate should be removed
#' @param x design matrix
#' @param y binary outcome
#'
#' @return leave-one-out residual
#' @noRd
get_res_logi <- function(beta,dim,x,y)
{
  beta[dim] = 0
  y_pred = 1/(1+exp(-x%*%beta))
  return(y-y_pred)
}




#' Soft-thresholding to get updated coefficient
#'
#'
#' @param lambda threshold
#' @param XX each column from design matrix
#' @param y binary outcome
#' @param W weight matrix
#'
#' @return soft thresholding result
#' @noRd
#'
one_dim_logis <- function(lambda,XX,y,W)
{
  N = length(y)
  W = diag(c(W),N,N)
  wnum = t(XX)%*%W%*%y/N
  wden = t(XX)%*%W%*%XX/N
  betaj = wnum/wden
  wlass <- soft_thresh(betaj,lambda/wden)
  return(wlass)
}


#' Lasso regression for logistics model
#'
#'
#' @param X Design Matrix
#' @param y binary outcome
#' @param lambda Lasso penalty parameter
#' @param tol convergence threshold
#' @param iter iteration times
#'
#' @return Lasso estimate of coefficients
#'
#' @examples
#' set.seed(1232)
#' Nz = 500
#' pz = 10
#' Xz = scale(matrix(rnorm(Nz*pz), ncol=pz))
#' bz = c(.5, -.5, .25, -.25, .125, -.125, rep(0, pz-6))
#' yz = rbinom(Nz,1,exp(Xz %*% bz)/(1+exp(Xz %*% bz)))
#' lambda = .1
#' require(glmnet)
#' fit <- glmnet(Xz,yz,family="binomial",lambda = 0.1,intercept = FALSE)
#' coef(fit,s = 0.1)

#' fit1 <- glmlasso(Xz,yz,tol=1e-12)
#'
#'
#'
#' @export
#'
glmlasso <- function(
  X,                   # model matrix
  y,                   # target
  lambda  = .1,        # penalty parameter
  tol     = 1e-6,      # tolerance
  iter    = 100      # number of max iterations
)
{
  #w = coef(glm.fit(X,y,family=binomial()))
  #w[is.na(w)]=1
  w = rep(1,ncol(X))
  i = 1
  tol_curr = 1
  w = t(w)
  w = t(w)
  while(tol < tol_curr && i < iter)
  {
    w_old = w
    for(j in 1:length(w))
    {
      W0 = exp(X%*%w)/(1+exp(X%*%w))^2   ###W = \pi(xi)*(1-\pi(xi))
      z = getzed(w,X,y,W0)
      res <- get_res_logi(w,j,X,z)
      w[j,] <- one_dim_logis(lambda,X[,j],y=res,W0)
    }
    i = i+1
    tol_curr <- crossprod(w - w_old)
  }
  return(w)
}


