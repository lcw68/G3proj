// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

class BLRM_Tune
{
  // short for Bayesian Logistic Regression Model Tuning
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
  void      setValues    (arma::vec&, arma::mat&);
  double    proposeBeta  (double);
  double    RejectRatio  (arma::vec&, arma::vec&, double);
};


void      BLRM_Tune::setValues   (arma::vec & Y0, arma::mat & X0)
{
  // Set class values to data provided by user
  Y      = Y0;
  X      = X0;

  p      = X0.n_cols;
  N      = X0.n_rows;

}


double    BLRM_Tune::loglike     (arma::vec & betaValue)
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

double    BLRM_Tune::proposeBeta   (double propSigma)
{
  // Generate random walk from proposal distribution for MwG sampling
  return R::rnorm(0,propSigma);
}

double    BLRM_Tune::logPrior      (arma::vec & betaValue, double PriorVar)
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

double    BLRM_Tune::RejectRatio   (arma::vec & betaProp, arma::vec & betaCurr, double PriorVar)
{
  return   loglike(betaProp) + logPrior(betaProp, PriorVar)
  - loglike(betaCurr)  - logPrior(betaCurr, PriorVar);
}


// [[Rcpp::export]]
Rcpp::List BLRM_Tuning(arma::vec & Y0, arma::mat & X0,
                        double PriorVar,
                        int nMC = 10000, int b = 50, int seed=1)
{
  // Set seed
  srand(seed);

  // Declare BLRM_Tune class
  BLRM_Tune brlm;

  // Set class parameters to user specified values
  brlm.setValues(Y0, X0);

  // Number of covariates
  int p0 = X0.n_cols;

  // Declare vectors to hold current and proposed beta values
  // initialize beta from standard normal distribution
  arma::vec betaCurr = arma::randn(p0);
  arma::vec betaProp(p0);

  // Declare and initialize vector to hold acceptance rate for each beta
  arma::vec acceptb(p0);
  acceptb.zeros();

  // Declare batch index
  int batch = 0;
  int maxbatch = nMC/b;

  // container for acceptance rate/proposal sd for each beta in each batch
  arma::mat ARbatch(maxbatch, p0);
  arma::mat SDbatch(maxbatch, p0);

  // container for log MH ratio and uniform random variable
  double logr,logu;

  // container for MwG sampled beta
  arma::mat betaSamples(nMC, p0);

  // initialize proposal variance
  arma::vec propsd(p0);
  propsd.ones();

  // container for increment for proposal variance
  double deltab;


  // main code for sampling
  for (int i = 0; i < nMC; i++)
  {
    for (int j = 0; j < p0; j++)
    {

      // sample beta by Metropolis within Gibbs

      // set proposal equal to previous
      betaProp = betaCurr;

      // only update the jth component
      betaProp(j) = betaProp(j) + brlm.proposeBeta(propsd(j)); // N(0, propSD0(j)) random walk

      // calculate log MH ratio
      logr  = brlm.RejectRatio(betaProp, betaCurr, PriorVar);
      logu  = log(R::runif(0,1));

      if (logu <= logr)
      {
        // if accept new proposal, then set current betaCurr as the new proposal
        // (otherwise current betaCurr remains)
        betaCurr = betaProp;
        // increase acceptance number
        acceptb(j)++;
      }
    }

    // at the end of batch
    if(floor((i+1)/(double)b) == ceil((i+1)/(double)b))
    {

      //Rcpp::Rcout << (batch+1) << "-th batch \n";

      // compute acceptance rate
      acceptb = acceptb/b;

      // calculate increment for proposal variacne
      deltab = std::min(0.01, 1/sqrt(i));

      // loop over proposal density variance vector
      for(int j = 0; j < p0; j++)
      {

        if(acceptb(j) > 0.45)
        {
          // if greater, add to sd
          propsd(j) = exp(log(propsd(j) + deltab));
        }

        else if(acceptb(j) < 0.43)
        {
          // otherwise, subtract
          propsd(j) = exp(log(propsd(j) - deltab));
        }
      }

      // Rcpp::Rcout << "AR for beta_1 is " << acceptb(1) << "\n";
      // Rcpp::Rcout << "propsd for beta_1 is " << propsd(1) << "\n\n";
      // Rcpp::Rcout << "AR for beta_2 is " << acceptb(2) << "\n";
      // Rcpp::Rcout << "propsd for beta_2 is " << propsd(2) << "\n\n";

      // save AR and proposal sd for this batch
      ARbatch.row(batch) = acceptb.t();
      SDbatch.row(batch) = propsd.t();

      // reset batch counter
      acceptb.zeros();

      // increase batch index
      batch++;
    }


    // save sampled beta after all coordinate updates finished
    betaSamples.row(i) = betaCurr.t();

  }

  // output list
  Rcpp::List lst = Rcpp::List::create(
    Rcpp::Named("beta.samples")   = betaSamples,
    Rcpp::Named("act.rate.batch") = ARbatch,
    Rcpp::Named("prop.sd.batch")  = SDbatch
  );

  return lst;

}
