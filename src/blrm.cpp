// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

class BLRM
{
  // short for Bayesian Logistic Regression Model
  private:
    // private member functions
    double    loglike  (arma::vec&);
    double    logPrior (arma::vec&, double);

  public:
    // public data objects
    //data
    int p;
    int N;

    arma::vec Y;
    arma::mat X;

    // proposal distribution information
    arma::vec propSD;

    // public member functions;
    void      setValues    (arma::vec&, arma::mat&, arma::vec&);
    double    proposeBeta  (int);
    double    RejectRatio  (arma::vec&, arma::vec&, double);
};


void      BLRM::setValues   (arma::vec & Y0, arma::mat & X0,
                             arma::vec & propSD0)
{
  // Set class values to data provided by user
  Y      = Y0;
  X      = X0;

  p      = X0.n_cols;
  N      = X0.n_rows;

  propSD = propSD0;
}


double    BLRM::loglike     (arma::vec & betaValue)
{
  // Evaluate log likelihood
  arma::vec eta = X*betaValue;
  arma::vec mu  = exp(eta) / (1 + exp(eta));

  double logLike = 0;
  for (int i=0;i<N;i++)
  {
    logLike += R::dbinom(Y[i],1,mu[i],1);
  }

  return logLike;
}

double    BLRM::proposeBeta   (int j)
{
  // Generate random walk from proposal distribution for MwG sampling
  return R::rnorm(0,propSD(j));
}

double    BLRM::logPrior      (arma::vec & betaValue, double PriorVar)
{
  // Evaluate log prior for beta
  if (PriorVar == 0)
    // non-informative improper uniform prior
  {
    return 0;
  }
  else
    // N(0, PriorVar * I) prior on beta
  {
    arma::vec kernel = -0.5*(1/PriorVar)*betaValue.t()*betaValue;
    return kernel[0];
  }
}

double    BLRM::RejectRatio   (arma::vec & betaProp, arma::vec & betaCurr, double PriorVar)
{
  return   loglike(betaProp) + logPrior(betaProp, PriorVar)
         - loglike(betaCurr)  - logPrior(betaCurr, PriorVar);
}


// [[Rcpp::export]]
Rcpp::List BLRM_fit_mwg(arma::vec & Y0, arma::mat & X0,
                       double PriorVar, arma::vec & propSD0,
                       int nMC = 1000, int nBI = 250, int thin = 5, int seed=1)
{
  // Set seed
  srand(seed);

  // Declare brlm class
  BLRM brlm;

  // Set class parameters to user specified values
  brlm.setValues(Y0, X0, propSD0);

  // Number of covariates
  int p0 = X0.n_cols;

  // Declare vectors to hold current and proposed beta values
  // initialize beta from standard normal distribution
  arma::vec betaCurr = arma::randn(p0);
  arma::vec betaProp(p0);

  // Declare and initialize vector to hold acceptance rate for each beta
  arma::vec accept(p0);
  accept.zeros();

  // container for log MH ratio and uniform random variable
  double logr,logu;

  // container for MwG sampled beta
  arma::mat betaSamples(nMC, p0);

  // main code for sampling
  for (int i = -nBI; i < nMC; i++)
  {
    for (int k = 0; k < thin; k++)
    {
      // loop for thinning purpose
      for (int j = 0; j < p0; j++)
      {

        // sample beta by Metropolis within Gibbs

        // set proposal equal to previous
        betaProp = betaCurr;

        // only update the jth component
        betaProp(j) = betaProp(j) + brlm.proposeBeta(j); // N(0, propSD0(j)) random walk

        // calculate log MH ratio
        logr  = brlm.RejectRatio(betaProp, betaCurr, PriorVar);
        logu  = log(R::runif(0,1));

        if (logu <= logr)
        {
          // if accept new proposal, then set current betaCurr as the new proposal
          // (otherwise current betaCurr remains)
          betaCurr = betaProp;
          // increase acceptance number
          if (i>=0) {accept(j)++;}
        }
      }
    }

    // save sampled beta after all coordinate updates finished
    if (i>=0)
    {
      betaSamples.row(i) = betaCurr.t();
    }
  }

  // compute acceptance rate for each beta
  accept = accept / (nMC*thin);

  // output list
  Rcpp::List lst = Rcpp::List::create(
    Rcpp::Named("beta.samples")  = betaSamples,
    Rcpp::Named("beta.act.rate") = accept
  );
  return lst;

}
