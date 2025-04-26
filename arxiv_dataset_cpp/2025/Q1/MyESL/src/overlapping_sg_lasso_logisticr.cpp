
#include "overlapping_sg_lasso_logisticr.hpp"
#include "overlapping.hpp"
#include <sstream>
#include <iomanip>


OLSGLassoLogisticR::OLSGLassoLogisticR(const arma::mat& features,
                                   const arma::rowvec& responses,
                                   const arma::mat& weights,
                                   const arma::rowvec& field,
                                   double* lambda,
                                   std::map<std::string, std::string> slep_opts,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  Train(features, responses, weights, slep_opts, field, intercept);
}


OLSGLassoLogisticR::OLSGLassoLogisticR(const arma::mat& features,
                                   const arma::rowvec& responses,
                                   const arma::mat& weights,
                                   const arma::rowvec& field,
                                   double* lambda,
                                   std::map<std::string, std::string> slep_opts,
                                   const arma::rowvec& xval_idxs,
                                   int xval_id,
                                   const bool intercept) :
    lambda(lambda),
    intercept(intercept)
{
  //subset features and responses according to xval_id and xval_idxs
  arma::uvec indices = arma::find(xval_idxs != xval_id);
  Train(features.cols(indices), responses.elem(indices).t(), weights, slep_opts, field, intercept);
}


void OLSGLassoLogisticR::writeModelToXMLStream(std::ofstream& XMLFile)
{
  int i_level = 0;
  //std::string XMLString = "";
  XMLFile << std::string(i_level * 8, ' ') + "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<model>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<parameters>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<n_rows>" + std::to_string(this->parameters.n_cols) + "</n_rows>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_cols>" + std::to_string(this->parameters.n_rows) + "</n_cols>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_elem>" + std::to_string(this->parameters.n_elem) + "</n_elem>" + "\n";
  //for(int i=0; i<this->parameters.n_cols; i++)
  for(int i=0; i<this->parameters.n_elem; i++)
  {
    std::ostringstream streamObj;
    streamObj << std::setprecision(17) << std::scientific << this->parameters(i);
    XMLFile << std::string(i_level * 8, ' ') + "<item>" + streamObj.str() + "</item>" + "\n";
  }
  i_level--;
  XMLFile << std::string(i_level * 8, ' ') + "</parameters>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<lambda1>" + std::to_string(this->lambda[0]) + "</lambda1>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<lambda2>" + std::to_string(this->lambda[1]) + "</lambda2>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<intercept_value>" + std::to_string(this->intercept_value) + "</intercept_value>" + "\n";
  i_level--;
  XMLFile << std::string(i_level * 8, ' ') + "</model>" + "\n";

}

void OLSGLassoLogisticR::writeSparseMappedWeightsToStream(std::ofstream& MappedWeightsFile, std::ifstream& FeatureMap)
{
  /*
  int i_level = 0;
  //std::string XMLString = "";
  XMLFile << std::string(i_level * 8, ' ') + "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\" ?>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<model>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<parameters>" + "\n";
  i_level++;
  XMLFile << std::string(i_level * 8, ' ') + "<n_rows>" + std::to_string(this->parameters.n_cols) + "</n_rows>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_cols>" + std::to_string(this->parameters.n_rows) + "</n_cols>" + "\n";
  XMLFile << std::string(i_level * 8, ' ') + "<n_elem>" + std::to_string(this->parameters.n_elem) + "</n_elem>" + "\n";
  */
  std::string line;
  std::getline(FeatureMap, line);
  //for(int i=0; i<this->parameters.n_cols; i++)
  for(int i=0; i<this->parameters.n_elem; i++)
  {
    std::getline(FeatureMap, line);
    if (this->parameters(i) == 0.0)
    {
	  continue;
    }
    std::istringstream iss(line);
    std::string feature_label;
    std::getline(iss, feature_label, '\t');
    std::getline(iss, feature_label, '\t');
    std::ostringstream streamObj;
    streamObj << std::setprecision(17) << std::scientific << this->parameters(i);
    MappedWeightsFile << feature_label + "	" + streamObj.str() + "\n";
  }
  MappedWeightsFile << "Intercept	" + std::to_string(this->intercept_value) + "\n";
  FeatureMap.clear();
  FeatureMap.seekg(0);
}


arma::rowvec& OLSGLassoLogisticR::Train(const arma::mat& features,
                               const arma::rowvec& responses,
                               const arma::mat& weights,
                               std::map<std::string, std::string> slep_opts,
                               const arma::rowvec& field,
                               const bool intercept)
{
  this->intercept = intercept;
  auto trim = [](std::string& s)
  {
     size_t p = s.find_first_not_of(" \t\r\n");
     s.erase(0, p);

     p = s.find_last_not_of(" \t\r\n");
     if (std::string::npos != p)
        s.erase(p+1);
  };

  //Set all optional parameters to defaults
  int opts_maxIter = 100;
  int opts_init = 2; //Previously set to 2
  int opts_tFlag = 5;
  int opts_nFlag = 0;
  int opts_rFlag = 1;
  int opts_mFlag = 0;
  double opts_tol = 0.0001;
  int opts_disableEC = 0;
  arma::mat opts_ind = weights;
  opts_ind.cols(0,1) = opts_ind.cols(0,1) - 1;
  arma::rowvec opts_field = field - 1;

  //Set overlapping specific parameters to defaults

  int opts_maxIter2 = 100;
  double opts_tol2 = 0.0001;
  int opts_flag2 = 2;
  //Overwrite default options with those found in slep_opts file.
  if ( slep_opts.find("maxIter") != slep_opts.end() ) {
	opts_maxIter = std::stoi(slep_opts["maxIter"]);
  }
  int opts_rStartNum = opts_maxIter;
  if ( slep_opts.find("init") != slep_opts.end() ) {
	opts_init = std::stoi(slep_opts["init"]);
  }
  if ( slep_opts.find("tFlag") != slep_opts.end() ) {
	opts_tFlag = std::stoi(slep_opts["tFlag"]);
  }
  if ( slep_opts.find("nFlag") != slep_opts.end() ) {
	opts_nFlag = std::stoi(slep_opts["nFlag"]);
  }
  if ( slep_opts.find("rFlag") != slep_opts.end() ) {
	opts_rFlag = std::stoi(slep_opts["rFlag"]);
  }
  if ( slep_opts.find("mFlag") != slep_opts.end() ) {
	opts_mFlag = std::stoi(slep_opts["mFlag"]);
  }
  if ( slep_opts.find("tol") != slep_opts.end() ) {
	opts_tol = std::stod(slep_opts["tol"]);
  }
  if ( slep_opts.find("tol2") != slep_opts.end() ) {
	opts_tol2 = std::stod(slep_opts["tol2"]);
  }
  if ( slep_opts.find("maxIter2") != slep_opts.end() ) {
	opts_maxIter2 = std::stoi(slep_opts["maxIter2"]);
  }
  if ( slep_opts.find("flag2") != slep_opts.end() ) {
	opts_flag2 = std::stoi(slep_opts["flag2"]);
  }
  if ( slep_opts.find("disableEC") != slep_opts.end() ) {
	opts_disableEC = std::stoi(slep_opts["disableEC"]);
  }
  std::string line;
  if ( slep_opts.find("nu") != slep_opts.end() ) {
        std::vector<float> opts_nu;
        std::ifstream nuFile (slep_opts["nu"]);
        if (nuFile.is_open())
        {
          while (getline(nuFile, line))
          {
            trim(line);
            opts_nu.push_back(std::stod(line));
          }
        }
  }
  if ( slep_opts.find("mu") != slep_opts.end() ) {
        std::vector<float> opts_mu;
        std::ifstream muFile (slep_opts["mu"]);
        if (muFile.is_open())
        {
          while (getline(muFile, line))
          {
            trim(line);
            opts_mu.push_back(std::stod(line));
          }
        }
  }
  std::vector<float> opts_sWeight;
  if ( slep_opts.find("sWeight") != slep_opts.end() ) {
        std::ifstream sWeightFile (slep_opts["sWeight"]);
        if (sWeightFile.is_open())
        {
          while (getline(sWeightFile, line))
          {
            trim(line);
            opts_sWeight.push_back(std::stod(line));
          }
        }
  }


  // Reminder: Armadillo stores data in column major order
  const size_t nCols = features.n_cols;

  arma::mat A = features.t();
  arma::mat& ind = opts_ind;
  arma::colvec y = responses.t();
  double* z;
  z = this->Lambda();
  double lambda2_max;
  const size_t m = A.n_rows;
  const size_t n = A.n_cols;
  arma::colvec m_ones(m, arma::fill::ones);
  arma::colvec m_zeros(m, arma::fill::zeros);
  arma::colvec n_zeros(n, arma::fill::zeros);

  double lambda1 = z[0];
  double lambda2 = z[1];

  double *gap;
  gap = (double*) malloc(sizeof(double));
  double penalty2 [5];

  if (lambda1<0 || lambda2<0)
  {
	throw std::invalid_argument("\n z should be nonnegative!\n");
  }

  if(ind.n_cols != 3)
  {
	throw std::invalid_argument("\n Check opts_ind, expected 3 cols\n");
  }

  int groupNum = ind.n_rows;
  arma::colvec Y(opts_field.n_cols, arma::fill::zeros);

  arma::colvec sample_weights(m);
  arma::uvec p_flag = arma::find(y == 1);
  arma::uvec not_p_flag = arma::find(y != 1);
  double m1, m2;
  if (opts_sWeight.size() == 2)
  {
    std::cout << "Using sample weights of " << opts_sWeight[0] << "(positive) and " << opts_sWeight[1] << "(negative)" << std::endl;
    m1 = p_flag.n_elem * opts_sWeight[0];
    m2 = not_p_flag.n_elem * opts_sWeight[1];
    sample_weights(p_flag).fill(opts_sWeight[0] / (m1 + m2));
    sample_weights(not_p_flag).fill(opts_sWeight[1] / (m1 + m2));

  } else if (opts_sWeight.size() != 0) {
    std::cout << "Invalid sample weights specified, defaulting to unweighted samples." << std::endl;
    sample_weights.fill(1.0/m);
  } else {
  sample_weights.fill(1.0/m);
  }

  int rsL2 = 0; //Todo: add overlapping_LogisticR.m:257-261

  arma::colvec b(m);
//std::cout << "p_flag.n_elem:" << p_flag.n_elem << std::endl;
  //double m1 = static_cast<double>(p_flag.n_elem) / (double)m;
  m1 = arma::sum(sample_weights(p_flag)) / arma::sum(sample_weights);
  m2 = 1 - m1;

  double* lambda;
  bool estimate_l2 = false;

  if (opts_rFlag == 0)
  {
	  lambda = z;
  } else {
	 if (lambda1<0 || lambda1>1 || lambda2<0 || lambda2>1)
	 {
		throw std::invalid_argument("\n opts.rFlag=1, so z should be in [0,1]\n");
	 }
	 b(p_flag) = arma::colvec(p_flag.n_elem, arma::fill::ones) * m2;   b(not_p_flag) = arma::colvec(not_p_flag.n_elem, arma::fill::ones) * (-m1);
	 b = b % sample_weights;

	 arma::mat ATb = A.t() * b;

	 arma::mat temp = arma::abs(ATb);

	 double lambda1_max = arma::as_scalar(arma::max(temp));

	 lambda1 = lambda1 * lambda1_max;
	 std::cout << "lambda1_max: " << lambda1_max << " lambda1: " << lambda1 << std::endl;

	 temp = arma::max(temp - lambda1, n_zeros);


	 if (temp.n_rows==field.n_cols)
	 {
		lambda2_max = computeLambda2Max(temp.t(), n, weights, weights.n_rows);
	 }
	 else
	 {
		 lambda2_max = 1;
		 std::cout << "Could not compute Lambda2 max, attempting to estimate instead..." << std::endl;
		 estimate_l2 = true;
	 }
  }

  arma::colvec x(n, arma::fill::zeros);   //x.fill(0);
  double c = std::log(m1/m2);

//std::cout << "A_cols:" << A.n_cols << " A_rows:" << A.n_rows << " x_cols:" << x.n_cols << " x_rows:" << x.n_rows << std::endl;

  arma::mat Ax = A * x;

//std::cout << "Ax_cols:" << Ax.n_cols << " Ax_rows:" << Ax.n_rows << std::endl;

  int bFlag = 0;
  double L = (1.0/m) + rsL2;

  arma::colvec weighty = sample_weights % y;

  arma::colvec xp = x;   arma::colvec Axp = Ax;   arma::colvec xxp(n, arma::fill::zeros);
  double cp = c;   double ccp = 0;

  double alphap = 0;   double alpha = 1;

  double beta, sc, gc, fun_s, fun_x, l_sum, r_sum, tree_norm;
//  arma::mat As;
  arma::colvec aa, bb, prob, s, v;
  arma::rowvec ValueL(opts_maxIter);
  arma::rowvec funVal(opts_maxIter);
  //arma::mat ind_work(ind.n_rows + 1, ind.n_cols);

std::cout << "m:" << m << " n:" << n << std::endl;

opts_ind = opts_ind.t(); //This might be wrong

  for (int iterStep = 0; iterStep < opts_maxIter; iterStep = iterStep + 1)
  {
    beta = (alphap - 1)/alpha;   s = x + (xxp * beta);   sc = c + (beta * ccp);
//std::cout << "sc:" << sc << " c:" << c << " beta:" << beta << " ccp:" << ccp << " m1:" << m1 << " m2:" << m2 << std::endl;
//std::cout << "As_elems:" << As.n_elem << "Ax_elems:" << Ax.n_elem << " Axp_elems:" << Axp.n_elem << " beta:" << beta << std::endl;
    arma::mat As = Ax + ((Ax - Axp) * beta);
    aa = -y % (As + sc);
    bb = arma::max(aa,m_zeros);
    fun_s = arma::as_scalar(sample_weights.t() * (arma::log(arma::exp(-bb) + arma::exp(aa - bb)) + bb)) + arma::as_scalar(rsL2/2 * s.t() * s);
    prob = m_ones / (m_ones + arma::exp(aa));
    b = -weighty % (m_ones - prob);

    gc = arma::sum(b);
    arma::mat g = A.t() * b;
    g = g + (rsL2 * s);
    xp = x;   Axp = Ax;   cp = c;

	if (iterStep == 0)
	{
	  if (estimate_l2)
	  {
		  v = s - g/L;
		  double l2_target, l2_low = 0, l2_high = 1;
		  //Check l2_high
		  for(int i = 1; i <= 100; i = i + 1)
		  {
			//Run overlapping
			overlapping(x.memptr(), gap, penalty2, v.memptr(),  n, groupNum, lambda1/L, l2_high/L, opts_ind.memptr(), opts_field.memptr(), Y.memptr(), opts_maxIter2, opts_flag2, opts_tol2);
			//Check if x is zeroed out
			if (arma::max(arma::abs(x)) == 0)
			{
				x.fill(0);
				std::cout << "Lambda2 High set to " << l2_high << std::endl;
				break;
			} else {
				x.fill(0);
				l2_high = l2_high * 2;
			}
		  }
		  //Estimate l2_max
		  for(int i = 1; i <= 100; i = i + 1)
		  {
			l2_target = (l2_high + l2_low) / 2.0;
			//Run overlapping
			overlapping(x.memptr(), gap, penalty2, v.memptr(),  n, groupNum, lambda1/L, l2_target/L, opts_ind.memptr(), opts_field.memptr(), Y.memptr(), opts_maxIter2, opts_flag2, opts_tol2);
			//Check if x is zeroed out
			if (arma::max(arma::abs(x)) == 0)
			{
				l2_high = l2_target;
			} else {
				l2_low = l2_target;
			}
			//Set x back to initial conditions
			x.fill(0);
			if (l2_high - l2_low < 0.0001)
			{
				lambda2_max = l2_high;
				std::cout << "Lambda2 Max set to " << l2_high << std::endl;
				break;
			}
		  }
	  }

	  if (opts_rFlag != 0)
	  {
		  lambda2 = lambda2 * lambda2_max;
		  std::cout << "Lambda2 set to " << lambda2 << std::endl;
	  }
	}
    int firstFlag = 1;
    while (true)
    {
      v = s - g/L; c = sc - gc/L;
      overlapping(x.memptr(), gap, penalty2, v.memptr(),  n, groupNum, lambda1/L, lambda2/L, opts_ind.memptr(), opts_field.memptr(), Y.memptr(), opts_maxIter2, opts_flag2, opts_tol2);

//std::cout << "temp_x_rows:" << temp_x.n_rows << " temp_x_cols:" << temp_x.n_cols << " x_rows:" << x.n_rows << " x_cols:" << x.n_cols << std::endl;

      v = x - s;
      int nonzero_x_count = 0;
      for (int i = 0; i < x.n_rows; i = i + 1)
      {
		 //std::cout << "aa["<< i << "]:" << aa(i) << " bb[" << i << "]:" << bb(i) << " Ax[" << i << "]:" << Ax(i) << std::endl;
		 if (x(i) != 0) { nonzero_x_count = nonzero_x_count + 1;}
	  }
//std::cout << "nonzero_x_count:" << nonzero_x_count << std::endl;

      Ax = A * x;
      aa = -y % (Ax + c);
      bb = arma::max(aa, m_zeros);
//std::cout << "aa.n_rows:" << aa.n_rows << " bb.n_rows:" << bb.n_rows << " c:" << c << " sc:" << sc << " gc:" << gc << " L:" << L << std::endl;
//      for (int i = 0; i <= bb.n_rows; i = i + 1)
//      {
//		 std::cout << "aa["<< i << "]:" << aa(i) << " bb[" << i << "]:" << bb(i) << " Ax[" << i << "]:" << Ax(i) << " y[" << i << "]:" << y(i) << std::endl;
//	  }
      fun_x = arma::as_scalar(sample_weights.t() * (arma::log(arma::exp(-bb) + arma::exp(aa - bb)) + bb)) + arma::as_scalar((rsL2/2)*x.t()*x);
      r_sum = (arma::as_scalar(v.t() * v) + std::pow(c - sc, 2)) / 2;
      l_sum = fun_x - fun_s - arma::as_scalar(v.t() * g) - (c - sc) * gc;

//std::cout << "r_sum:" << r_sum << " l_sum:" << l_sum << " L:" << L << " fun_x:" << fun_x << std::endl;

      if (r_sum <= std::pow(0.1, 20))
      {
	     bFlag = 1;
	     break;
	  }

//	  if (l_sum <= r_sum * L) // Changed to epsilon comparison on 4/17/2024 to match windows output
	  if ((opts_disableEC==0 && (l_sum < r_sum * L || abs(l_sum - (r_sum * L)) < std::pow(0.1, 12)) || opts_disableEC==1 && l_sum <= r_sum * L))
	  {
	     break;
	  } else {
	     L = std::max(2*L, l_sum/r_sum);
	  }
    }

    alphap = alpha;   alpha = (1 + std::pow(4 * alpha * alpha + 1.0, 0.5))/2.0;
    ValueL(iterStep) = L;
    xxp = x - xp;   ccp = c - cp;
    funVal(iterStep) = fun_x + lambda1 * arma::sum(arma::abs(x));

    if (bFlag) {break;}

    double norm_xp, norm_xxp;

    switch (opts_tFlag)
    {
	  case 0:
        if (iterStep >=1)
        {
	      if (std::abs(funVal(iterStep) - funVal(iterStep - 1)) <= opts_tol * funVal(iterStep - 1))
	      {
	        bFlag = 1;
	      }
	    }
	    break;
	  case 1:
	    if (iterStep >=1)
	    {
		  if (abs(funVal(iterStep) - funVal(iterStep - 1)) <= opts_tol * funVal(iterStep - 1))
	      {
			bFlag = 1;
	      }
		}
		break;
	  case 2:
	    if (funVal(iterStep) <= opts_tol)
	    {
		  bFlag = 1;
	    }
	    break;
	  case 3:
	    norm_xxp = sqrt(arma::as_scalar(xxp.t() * xxp));
	    if (norm_xxp <= opts_tol)
	    {
		  bFlag = 1;
	    }
	    break;
	  case 4:
	    norm_xp = sqrt(arma::as_scalar(xp.t() * xp)); norm_xxp = sqrt(arma::as_scalar(xxp.t() * xxp));
	    if (norm_xxp <= opts_tol * std::max(norm_xp, 1.0))
	    {
		  bFlag = 1;
	    }
	    break;
	  case 5:
        if (iterStep >= opts_maxIter)
        {
          bFlag = 1;
        }
        break;
    }
    if (bFlag) {break;}

	if ((iterStep + 1) % opts_rStartNum == 0)
	{
	  alphap = 0;   alpha = 1;   xp = x;   Axp = Ax;   xxp = n_zeros;   L = L/2;
	}

  }

  arma::rowvec x_row = x.as_row();
  parameters = x_row.t();
  std::cout << "Intercept: " << c << std::endl;
  this->intercept_value = c;

  this->nz_gene_count = countNonZeroGenes(parameters, weights, opts_field);

  return x_row;

}



const arma::colvec OLSGLassoLogisticR::altra(const arma::colvec& v_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
	double *x;
	x = (double*) malloc(n*sizeof(double));
	const double* v = v_in.memptr();
    int i, j, m;
    double lambda,twoNorm, ratio;
    double ind[ind_mat.n_cols * ind_mat.n_rows];

    for(int k=0;k<ind_mat.n_cols;k++)
    {
	    for(int l=0;l<ind_mat.n_rows;l++)
	    {
		    ind[(l*3)+k] = ind_mat(l,k);
		}
	}

//	for(int k=0;k<ind_mat.n_cols*ind_mat.n_rows;k++)
//	{
//	    std::cout << "ind[" << k << "]:" << ind[k] << std::endl;
//	}


//std::cout << "Altra 2..." << std::endl;
    /*
     * test whether the first node is special
     */
    if ((int) ind[0]==-1){

        /*
         *Recheck whether ind[1] equals to zero
         */
        if ((int) ind[1]!=-1){
            printf("\n Error! \n Check ind");
            exit(1);
        }

        lambda=ind[2];

        for(j=0;j<n;j++){
            if (v[j]>lambda)
                x[j]=v[j]-lambda;
            else
                if (v[j]<-lambda)
                    x[j]=v[j]+lambda;
                else
                    x[j]=0;
        }

        i=1;
    }
    else{
        memcpy(x, v, sizeof(double) * n);
        i=0;
    }

    /*
     * sequentially process each node
     *
     */
	for(;i < nodes; i++){
        /*
         * compute the L2 norm of this group
         */
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);

        lambda=ind[3*i+2];
        if (twoNorm>lambda){
            ratio=(twoNorm-lambda)/twoNorm;

            /*
             * shrinkage this group by ratio
             */
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[j]*=ratio;
        }
        else{
            /*
             * threshold this group to zero
             */
            for(j=(int) ind[3*i]-1;j<(int) ind[3*i+1];j++)
                x[j]=0;
        }
	}

	arma::colvec x_col(&x[0], n);
	free(x);

	return x_col;
}


const double OLSGLassoLogisticR::treeNorm(const arma::rowvec& x_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
	double tree_norm;
	const double* x = x_in.memptr();
    int i, j, m;
    double twoNorm, lambda;
    double ind[ind_mat.n_cols * ind_mat.n_rows];

    for(int k=0;k<ind_mat.n_cols;k++)
    {
	    for(int l=0;l<ind_mat.n_rows;l++)
	    {
		    ind[(l*3)+k] = ind_mat(l,k);
		}
	}

//	for(int k=0;k<ind_mat.n_cols*ind_mat.n_rows;k++)
//	{
//	    std::cout << "ind[" << k << "]:" << ind[k] << std::endl;
//	}

    tree_norm=0;

    /*
     * test whether the first node is special
     */
    if ((int) ind[0]==-1){

        /*
         *Recheck whether ind[1] equals to zero
         */
        if ((int) ind[1]!=-1){
            printf("\n Error! \n Check ind");
            exit(1);
        }

        lambda=ind[2];

        for(j=0;j<n;j++){
            tree_norm+=fabs(x[j]);
        }

        tree_norm=tree_norm * lambda;

        i=1;
    }
    else{
        i=0;
    }

    /*
     * sequentially process each node
     *
     */
	for(;i < nodes; i++){
        /*
         * compute the L2 norm of this group
         */
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);

        lambda=ind[3*i+2];

        tree_norm=tree_norm + lambda*twoNorm;
	}

	return tree_norm;
}


const double OLSGLassoLogisticR::computeLambda2Max(const arma::rowvec& x_in,
                            const int n,
                            const arma::mat& ind_mat,
                            const int nodes) const
{
    int i, j, m;
    double lambda,twoNorm;
    const double* x = x_in.memptr();
    double ind[ind_mat.n_cols * ind_mat.n_rows];

    double lambda2_max = 0;

    for(int k=0;k<ind_mat.n_cols;k++)
    {
	    for(int l=0;l<ind_mat.n_rows;l++)
	    {
		    ind[(l*3)+k] = ind_mat(l,k);
		}
	}

    for(i=0;i < nodes; i++){
        /*
         * compute the L2 norm of this group
         */
		twoNorm=0;
		for(j=(int) ind[3*i]-1;j< (int) ind[3*i+1];j++)
			twoNorm += x[j] * x[j];
        twoNorm=sqrt(twoNorm);

        twoNorm=twoNorm/ind[3*i+2];

        if (twoNorm >lambda2_max )
            lambda2_max=twoNorm;
	}

	return lambda2_max;
}


int countNonZeroGenes(const arma::vec& arr, const arma::mat& ranges, const arma::rowvec& field) {
    auto detectNonZeroInRange = [&arr](int start, int end, const arma::rowvec& field) -> int {
        for (int i = start; i <= end; ++i) {
            if (arr(field[i]) != 0) {
                return 1;
            }
        }
        return 0;
    };
    int count = 0;

    for (arma::uword i = 0; i < ranges.n_rows; ++i) {
        int start = static_cast<int>(ranges(i, 0))-1;
        int end = static_cast<int>(ranges(i, 1))-1;
        count = count + detectNonZeroInRange(start, end, field);
    }

    //std::cout << "Number of non-zero genes: " << count << std::endl;
    return count;
}