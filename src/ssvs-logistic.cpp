// [[Rcpp::depends(RcppArmadillo)]]
#include<RcppArmadillo.h>

class SSVS {

private:
  // private member function
  double logLikeBeta(arma::vec&);
  double logPrior(arma::vec&, arma::vec&);

public:
  // public data objects
  // data
  int p;
  int N;

  arma::vec Y;
  arma::mat X;

  // prior hyperparameters
  // beta - normal mixture prior
  double c;
  double tau;

  // proposal distribution information
  arma::vec propSD;

  // public member functions
  void setValues (arma::vec&, arma::mat&, arma::vec&, double&, double&);
  double proposeBeta(int);
  double rejectionRatioBeta(arma::vec&, arma::vec&, arma::vec&);
  double sampGamma(arma::vec&, int&);
};


void SSVS::setValues(arma::vec & Y0, arma::mat & X0,
                    arma::vec & propSD0, double & c0, double & tau0)
{
  // Set class values to data provided by user
  Y      = Y0;
  X      = X0;

  p      = X0.n_cols;
  N      = X0.n_rows;

  c = c0;
  tau = tau0;

  propSD = propSD0;
}

double SSVS::sampGamma(arma::vec& beta, int& j)
{
  // sample gamma from its full conditional

  double a = exp(-0.5*beta(j)*beta(j) / (c*c*tau*tau)) / c;
  double b = exp(-0.5*beta(j)*beta(j) / (tau*tau));

  double pp = a/(a+b);
  int gamma_j = R::rbinom(1,pp);
  return(gamma_j);
}

double SSVS::logLikeBeta(arma::vec& beta){
  // Evaluate log likelihood for beta
  arma::vec eta = X*beta;
  arma::vec mu  = exp(eta) / (1 + exp(eta));

  double logLike = 0;
  for (int i=0;i<N;i++)
  {
    logLike += R::dbinom(Y[i],1,mu[i],1);
  }

  return logLike;
}

double SSVS::proposeBeta(int j)
{
  // Generate value from proposal distribution
  return(R::rnorm(0,propSD(j)));
}

double SSVS::logPrior(arma::vec& beta, arma::vec& gamma)
{
  // Evaluate log prior for beta
  arma::vec kernelvec;
  kernelvec = beta / (c*tau*gamma + tau*(1-gamma));
  return(-0.5*sum(kernelvec % kernelvec));
}

double SSVS::rejectionRatioBeta(arma::vec& betaProp, arma::vec& betaCurr, arma::vec& gamma)
{
  // Compute MH rejection ratio for beta
  return   logLikeBeta(betaProp) + logPrior(betaProp, gamma)
         - logLikeBeta(betaCurr) - logPrior(betaCurr, gamma);
}

// [[Rcpp::export]]
Rcpp::List SSVS_Logistic(arma::vec & Y0, arma::mat & X0,
                arma::vec & propSD0, double & c0, double & tau0,
                int nMC = 1000, int nBI = 250, int thin=5, int seed=1)
{
  //set seed
  srand(seed);
  // Declare ssvs class
  SSVS ssvs;

  // Set class parameters to user specified values
  ssvs.setValues(Y0, X0, propSD0, c0, tau0);

  // Number of covariates
  int p0 = X0.n_cols;

  // Declare and initialize gamma vector
  arma::vec gamma(p0);
  gamma = Rcpp::rbinom(p0,1,.5);

  // Declare vectors to hold current and proposed beta values
  arma::vec betaCurr = arma::randn(p0); // initialize beta from standard normal
  arma::vec betaProp(p0);

  // Declare and initialize vector to hold acceptance rate for each beta
  arma::vec acceptb(p0);
  acceptb.zeros();

  // container for log MH ratio and uniform random variable
  double logr, logu;

  // container for MC sampled beta and gamma
  arma::mat betaSamples(nMC, p0);
  arma::mat gammaSamples(nMC, p0);

  for(int i = -nBI; i < nMC*thin; i++){

    for(int j = 0; j < p0; j++){

      // sample beta by Metropolis within Gibbs

      // set proposal equal to previous
      betaProp = betaCurr;

      // only update the jth component
      betaProp(j) = betaProp(j) + ssvs.proposeBeta(j); // N(0,propSD0) random walk

      // calculate log MH ratio
      logr  = ssvs.rejectionRatioBeta(betaProp, betaCurr, gamma);
      logu  = log(R::runif(0,1));

      if (logu <= logr)
      {
        // if accept new proposal, then set current betaVal as the new proposal
        // (otherwise current betaVal remains)
        betaCurr = betaProp;
        // increase acceptance number
        if (i>=0 && i % thin == 0) {acceptb(j) = acceptb(j) + 1;}
      }


      // sample gamma from its full conditional directly (bernoulli distn)
      gamma(j) = ssvs.sampGamma(betaCurr,j);

    }


    // save sampled beta and gamma
    if(i >= 0 && i % thin == 0){
      betaSamples.row(i/thin) = betaCurr.t();
      gammaSamples.row(i/thin) = gamma.t();
    }
  }

  // compute acceptance rate for each beta
  acceptb = acceptb/nMC;

  // output list
  Rcpp::List lst = Rcpp::List::create(
    Rcpp::Named("beta.samples")  = betaSamples,
    Rcpp::Named("gamma.samples")  = gammaSamples,
    Rcpp::Named("beta.act.rate") = acceptb
  );

  return lst;

}
