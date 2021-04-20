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
