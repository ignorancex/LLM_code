/**
 * sweep_lr.cpp
 *
 * Sweep L1-regularized logistic regression across a range of lambda values,
 * storing the train/test accuracies in a CSV file.
 */
#include "logistic_regression.hpp"
#include "libsvm.hpp"
#include "data_utils.hpp"
#include <iostream>

using namespace std;

void help(char** argv)
{
  cout << "Usage: " << argv[0] << " input_data.svm output_file.csv seed "
       << "min_reg max_reg count [start [verbose]]" << endl
       << endl
       << " - note: lambda values are between 10^{min_reg} and 10^{max_reg}"
       << endl
       << " - if start is given, the grid will start from that index"
       << endl
       << " - verbose output is given if *any* argument is given"
       << endl;
}

int main(int argc, char** argv)
{
  // Make sure we got the right number of arguments.
  if (argc != 7 && argc != 8 && argc != 9)
  {
    help(argv);
    exit(1);
  }

  const std::tuple<std::string, std::string> inputFiles =
      SplitDatasetArgument(argv[1]);
  const std::string inputFile(std::get<0>(inputFiles));
  const std::string testFile(std::get<1>(inputFiles));
  const std::string outputFile(argv[2]);
  const size_t seed = atoi(argv[3]);
  const double minReg = atof(argv[4]);
  const double maxReg = atof(argv[5]);
  const size_t count = atoi(argv[6]);
  const size_t start = (argc == 8) ? atoi(argv[7]) : 0;
  const bool verbose = (argc == 9);

  const std::tuple<arma::sp_mat, arma::rowvec> t = load_libsvm<arma::sp_mat>(
      inputFile);

  const arma::sp_mat& data = std::get<0>(t);
  const arma::rowvec& labels = std::get<1>(t);

  srand(seed);
  arma::arma_rng::set_seed(seed);

  // Split into training and test sets.
  arma::sp_mat trainData, testData;
  arma::rowvec trainLabels, testLabels;
  if (testFile.empty())
  {
    TrainTestSplit(data, labels, 0.8, trainData, trainLabels, testData,
        testLabels);
  }
  else
  {
    // Load the test file separately.
    const std::tuple<arma::sp_mat, arma::rowvec> t2 = load_libsvm<arma::sp_mat>(
        testFile);

    trainData = std::move(data);
    trainLabels = std::move(labels);
    testData = std::move(std::get<0>(t2));
    testLabels = std::move(std::get<1>(t2));

    // Ensure matrices have the same dimension.
    const size_t maxDimension = std::max(trainData.n_rows, testData.n_rows);
    trainData.resize(maxDimension, trainData.n_cols);
    testData.resize(maxDimension, testData.n_cols);
  }

  fstream f(outputFile, fstream::out | fstream::ate);
  if (!f.is_open())
  {
    std::cerr << "Failed to open output file '" << outputFile << "'!"
        << std::endl;
    exit(1);
  }

  if (start == 0)
  {
    f << "method,index,lambda,lambda_pow,nnz,train_acc,test_acc,time" << endl;
  }

  // We don't want to use more than four threads for each individual training
  // process (that's about where it starts not helping out).
  const size_t totalThreads = omp_get_max_threads();
  std::cout << "Total threads: " << totalThreads << "." << std::endl;

  arma::wall_clock c;
  c.tic();
  LogisticRegression lr;
  lr.verbose = verbose;
  lr.seed = seed;
  const double step = (maxReg - minReg) / (count - 1);
  const double pow = maxReg - (step * start);
  double lambda = std::pow(10.0, pow);
  lr.lambda = lambda;
  // The first run needs to be done separately to populate the LIBLINEAR
  // features members.
  lr.Train(trainData, trainLabels, false);
  const double trainTime = c.toc();

  const std::tuple<double, double> accs = TrainTestAccuracy(lr, trainData,
      trainLabels, testData, testLabels);

  f << "full," << start << "," << lambda << "," << pow << ","
      << lr.modelNonzeros << "," << std::get<0>(accs) << ","
      << std::get<1>(accs) << "," << trainTime << endl;
  cout << "L1-regularized logistic regression, lambda 10^" << pow << ": "
      << trainTime << "s training time; " << lr.modelNonzeros << " nonzeros; "
      << std::get<0>(accs) << " training accuracy; " << std::get<1>(accs)
      << " testing accuracy." << endl;

  // Now split into parallel runs for all the other points on the grid.
  #pragma omp parallel for schedule(dynamic) \
                           default(shared) \
                           num_threads(totalThreads)
  for (size_t i = start + 1; i < count; ++i)
  {
    const double powThread = maxReg - (step * i);
    const double lambdaThread = std::pow(10.0, powThread);

    LogisticRegression lrThread(lr);

    arma::wall_clock cThread;
    cThread.tic();
    lrThread.lambda = lambdaThread;
    lrThread.Retrain(false); // same dataset
    const double trainTimeThread = cThread.toc();

    const std::tuple<double, double> accsThread = TrainTestAccuracy(lrThread,
        trainData, trainLabels, testData, testLabels);

    #pragma omp critical
    {
      f << "full," << i << "," << lambdaThread << "," << powThread << ","
          << lrThread.modelNonzeros << "," << std::get<0>(accsThread) << ","
          << std::get<1>(accsThread) << "," << trainTimeThread << endl;
      cout << "L1-regularized logistic regression, lambda 10^" << powThread
          << ": " << trainTimeThread << "s training time; "
          << lrThread.modelNonzeros << " nonzeros; " << std::get<0>(accsThread)
          << " training accuracy; " << std::get<1>(accsThread)
          << " testing accuracy." << endl;
    }
  }
}
