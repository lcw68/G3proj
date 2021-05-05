// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

class SSVS_Tune {

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

  // public member functions
  void setValues (arma::vec&, arma::mat&, double&, double&);
  double proposeBeta(double);
  double rejectionRatioBeta(arma::vec&, arma::vec&, arma::vec&);
  double sampGamma(arma::vec&, int&);
};


void SSVS_Tune::setValues(arma::vec & Y0, arma::mat & X0,
                     double & c0, double & tau0)
{
  // Set class values to data provided by user
  Y      = Y0;
  X      = X0;

  p      = X0.n_cols;
  N      = X0.n_rows;

  c = c0;
  tau = tau0;

}

double SSVS_Tune::sampGamma(arma::vec& beta, int& j)
{
  // sample gamma from its full conditional

  double a = exp(-0.5*beta(j)*beta(j) / (c*c*tau*tau)) / c;
  double b = exp(-0.5*beta(j)*beta(j) / (tau*tau));

  double pp = a/(a+b);
  int gamma_j = R::rbinom(1,pp);
  return(gamma_j);
}

double SSVS_Tune::logLikeBeta(arma::vec& beta){
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

double SSVS_Tune::proposeBeta(double propSigma)
{
  // Generate value from proposal distribution
  return(R::rnorm(0,propSigma));
}

double SSVS_Tune::logPrior(arma::vec& beta, arma::vec& gamma)
{
  // Evaluate log prior for beta
  arma::vec kernelvec;
  kernelvec = beta / (c*tau*gamma + tau*(1-gamma));
  return(-0.5*sum(kernelvec % kernelvec));
}

double SSVS_Tune::rejectionRatioBeta(arma::vec& betaProp, arma::vec& betaCurr, arma::vec& gamma)
{
  // Compute MH rejection ratio for beta
  return   logLikeBeta(betaProp) + logPrior(betaProp, gamma)
  - logLikeBeta(betaCurr) - logPrior(betaCurr, gamma);
}


// [[Rcpp::export]]
Rcpp::List SSVS_Tuning(arma::vec & Y0, arma::mat & X0,
                         double & c0, double & tau0,
                         int nMC = 1000, int b=50, int seed=1)
{
  //set seed
  srand(seed);
  // Declare ssvs class
  SSVS_Tune ssvs;

  // Set class parameters to user specified values
  ssvs.setValues(Y0, X0, c0, tau0);

  // Number of covariates
  int p0 = X0.n_cols;

  // Declare and initialize gamma vector
  arma::vec gamma(p0);
  gamma = Rcpp::rbinom(p0,1,.5);

  // Declare vectors to hold current and proposed beta values
  arma::vec betaCurr = arma::randn(p0); // initialize beta from standard normal
  arma::vec betaProp(p0);

  // Declare and initialize vector to hold acceptance rate for each beta in a given batch
  arma::vec acceptb(p0);
  acceptb.zeros();

  // Declare batch index
  int batch = 0;
  int maxbatch = nMC/b;

  // container for acceptance rate/proposal sd for each beta in each batch
  arma::mat ARbatch(maxbatch, p0);
  arma::mat SDbatch(maxbatch, p0);

  // container for log MH ratio and uniform random variable
  double logr, logu;

  // container for MC sampled beta and gamma
  arma::mat betaSamples(nMC, p0);
  arma::mat gammaSamples(nMC, p0);

  // initialize proposal variance
  arma::vec propsd(p0);
  propsd.ones();

  // container for increment for proposal variance
  double deltab;

  for(int i = 0; i < nMC; i++){

    for(int j = 0; j < p0; j++){

      // sample beta by Metropolis within Gibbs

      // set proposal equal to previous
      betaProp = betaCurr;

      // only update the jth component
      betaProp(j) = betaProp(j) + ssvs.proposeBeta(propsd(j)); // N(0,propSigma20) random walk

      // calculate log MH ratio
      logr  = ssvs.rejectionRatioBeta(betaProp, betaCurr, gamma);
      logu  = log(R::runif(0,1));

      if (logu <= logr)
      {
        // if accept new proposal, then set current betaVal as the new proposal
        // (otherwise current betaVal remains)
        betaCurr = betaProp;
        // increase acceptance number
        acceptb(j) = acceptb(j) + 1;
      }


      // sample gamma from its full conditional directly (bernoulli distn)
      gamma(j) = ssvs.sampGamma(betaCurr,j);

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

    // save sampled beta and gamma
    betaSamples.row(i) = betaCurr.t();
    gammaSamples.row(i) = gamma.t();

  }


  // output list
  Rcpp::List lst = Rcpp::List::create(
    Rcpp::Named("beta.samples")  = betaSamples,
    Rcpp::Named("gamma.samples")  = gammaSamples,
    Rcpp::Named("act.rate.batch") = ARbatch,
    Rcpp::Named("prop.sd.batch") = SDbatch
  );

  return lst;

}
