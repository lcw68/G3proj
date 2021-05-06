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

#' fit1 <- glmlasso(Xz,yz,lambda,tol=1e-12)
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
  if(is.factor(y)){

    if(length(levels(y)) >2)
    {
      stop("Response have more than 2 level")
    }
    y <- as.numeric(y)-1
  }
  if(nrow(X)!= length(y))
  {
    stop("Length of y is not equal to number of X's row")
  }

  w = rep(1,ncol(X))
  i = 1
  tol_curr = 1
  w = t(w)
  w = t(w)
  while(tol < tol_curr && i < iter)
  {
    w_old = w
    W0 = exp(X%*%w)/(1+exp(X%*%w))^2   ###W = \pi(xi)*(1-\pi(xi))
    z = getzed(w,X,y,W0)
    w_update <- function(j)
    {
      res <- get_res_logi(w,j,X,z)
      wn =one_dim_logis(lambda,X[,j],y=res,W0)
      return(wn)
    }

    w = t(t(sapply(1:length(w),w_update)))
    # for(j in 1:length(w))
    # {
    #   res <- get_res_logi(w,j,X,z)
    #   w[j,] <- one_dim_logis(lambda,X[,j],y=res,W0)
    # }
    i = i+1
    tol_curr <- crossprod(w - w_old)
  }
  return(w)
}


#' Prediction of Lasso logistics regression
#'
#'
#' @param fit fitted object from glmlasso
#' @param newdata a data frame in which to look for variables with which to predict.
#' @param type the type of prediction required. The default is on the scale of the response variable.
#' \code{type="class"} gives the predicted classification result
#' @param threshold Threshold values decided for classification
#'
#' @return if \code{type="class"} a vector of predicted class will be returned. Otherwise return a vector of raw probability.
#'
#' @examples
#' set.seed(1232)
#' Nz = 500
#' pz = 10
#' Xz = scale(matrix(rnorm(Nz*pz), ncol=pz))
#' bz = c(.5, -.5, .25, -.25, .125, -.125, rep(0, pz-6))
#' yz = rbinom(Nz,1,exp(Xz %*% bz)/(1+exp(Xz %*% bz)))
#' lambda = .1
#' fit1 <- glmlasso(Xz,yz,lambda,tol=1e-12)
#' pred <- predict_glmlasso(fit1,newdata = Xz)
#'
#'@export
predict_glmlasso <- function(fit, newdata, type="response", threshold = 0.5)
{
  pred = exp(newdata %*% fit)/(1+exp(newdata %*% fit))
  if(type == "class")
  {
    pred  = ifelse(pred > threshold,1,0)
  }
  return(pred)
}


#'select optimal tuning parameter lambda for lasso based on BIC criterion.
#'
#'This function will fit the glmlasso with a sequence value of tuning parameter lambdas and
#'return a plot of BIC vs lambda and the optimal lambda with the smallest BIC.
#'
#'This function will need glmlasso function.
#'
#'@param Xz the design matrix for glmlasso
#'@param yz a vector of binary outcomes
#'@param lambda.min the smallest lambda
#'@param lambda.max the largest lambda
#'@param len the number of lambdas for fitting
#'@param plot logic variable, indicating if the lambda versus BIC plot should be returned
#'
#'@return a plot of BIC vs lambda and the optimal lambda with the smallest BIC
#'
#'@examples
#'
#'set.seed(8675309)
#'Xz = scale(matrix(rnorm(5000), ncol=10))
#'bz = c(.5, -.5, .25, -.25, .125, -.125, rep(0, 4))
#'yz = rbinom(Nz,1,exp(Xz %*% bz)/(1+exp(Xz %*% bz)))
#'optim.lambda(Xz,yz,lambda.min = 0,lambda.max = 0.1,len = 200)
#'
#'
#'@export
optim.lambda <- function(Xz,yz,lambda.min,lambda.max,len,plot=FALSE){
  if(is.factor(yz)){

    if(length(levels(yz)) >2)
    {
      stop("Response have more than 2 level")
    }
    yz <- as.numeric(yz)-1
  }
  if(nrow(Xz)!= length(yz))
  {
    stop("Length of y is not equal to number of X's row")
  }

  if(lambda.min > lambda.max)
  {
    stop("min could not be greater than max!")
  }

  lambda = seq(lambda.min,lambda.max,length = len+1)[-1]
  BIC.cal <- function(i){
    ww <- glmlasso(Xz,yz,lambda =lambda[i])
    xb = Xz%*%ww
    pp = exp(xb)/(1+exp(xb))
    Devv <- -2*(t(yz)%*%(xb)+sum(log(1-pp))) #deviance for logistic regression
    k = sum(ww!=0) #df
    BIC = Devv+ k*log(length(yz))
    return(BIC)
  }
  BIC = sapply(1:len, BIC.cal)

  # Choose the minimum value
   min_BIC = which(BIC == min(BIC))
   # opt.lambda
 l1 <- lambda[min_BIC]
  # plot of BIC v.s. lambda
  if (plot==TRUE){
    l2 <- plot(lambda, BIC,
         xlim = c(lambda.min, lambda.max),
         xlab = expression(lambda),
         type = 'l', lwd = 2, lty = 1, col = 1)
    abline(v=lambda[min_BIC], col = 2)
    return(list(l1,l2))
  }
 else{
   return(list(l1))
 }
}




