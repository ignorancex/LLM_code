/**
 * sweep_prox_csl.cpp
 *
 * Sweep distributed lasso + full prox CSL updates
 * across a range of lambda values,
 * storing the train/test accuracies in a CSV file.
 */
#include "global_step.hpp"
#include "libsvm.hpp"
#include "data_utils.hpp"
#include <iostream>

using namespace std;

void help(char** argv)
{
  cout << "Usage: " << argv[0] << " input_data.svm output_file.csv seed "
       << "min_reg max_reg count partitions start_mode alpha adaptive" << endl
       << endl
       << " - note: lambda values are between 10^{min_reg} and 10^{max_reg}"
       << endl
       << " - start_mode determines initial solution: 0=zeros, 1=naive, 2=owa"
       << endl;
}

int main(int argc, char** argv)
{
  // Make sure we got the right number of arguments.
  if (argc != 11)
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
  const size_t partitions = atoi(argv[7]);
  const size_t start_mode = atoi(argv[8]);
  const double alpha = atof(argv[9]);
  const double adaptive = atoi(argv[10]);
  const size_t start = 0;
  const bool verbose = false;
  const bool adapt = (adaptive == 1);

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

  // Check if we will be able to split into partitions easily.  If not, just
  // drop some extra columns...
  const size_t partitionPoints = (trainData.n_cols + partitions - 1) /
      partitions;
  if (partitionPoints * (partitions - 1) >= trainData.n_cols)
  {
    // Just drop the extra points until it divides evenly...
    const size_t evenPartitionPoints = trainData.n_cols / partitions;
    std::cout << "Things don't divide evenly; dropping points "
        << evenPartitionPoints * partitions << " to "
        << trainData.n_cols - 1 << "; this gives ";
    trainData.shed_cols(evenPartitionPoints * partitions,
                        trainData.n_cols - 1);
    std::cout << trainData.n_cols << " points overall." << std::endl;
    trainLabels.shed_cols(evenPartitionPoints * partitions,
                          trainLabels.n_elem - 1);
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

  const size_t totalThreads = omp_get_max_threads();
  std::cout << "Total threads: " << totalThreads << "." << std::endl;

  arma::wall_clock c;
  c.tic();
  const double step = (maxReg - minReg) / (count - 1);
  const double pow = maxReg - (step * start);
  double lambda = std::pow(10.0, pow);
  
  GlobalStep<arma::sp_mat> Gs(lambda, partitions);
  Gs.numThreads = totalThreads;
  Gs.verbose = verbose;
  Gs.seed = seed;

  // The first run needs to be done separately to populate the LIBLINEAR
  // features members.

  // Set the initialization for w_0
  if (start_mode == 1)
    Gs.NaiveTrain(trainData, trainLabels);
  else if (start_mode == 2)
    Gs.Train(trainData, trainLabels);
  else if (start_mode == 0)
  {
    Gs.NaiveTrain(trainData, trainLabels);
    for (int j = 0; j < trainData.n_rows; ++j)
      Gs.model[j] = 0;
    Gs.modelNonzeros = arma::accu(Gs.model != 0);
  }
  const double initTrainTime = c.toc();

  const std::tuple<double, double> accs = TrainTestAccuracy(Gs, trainData,
      trainLabels, testData, testLabels);

  f << "pcsl-" << partitions << "," << start << "," << lambda << ","
      << pow << "," << Gs.modelNonzeros << "," << std::get<0>(accs) << ","
      << std::get<1>(accs) << "," << initTrainTime << endl;
  cout << "proxCSL, " << partitions << " partitions, lambda 10^" << pow
      << ": " << initTrainTime << "s training time; " << Gs.modelNonzeros
      << " nonzeros; " << std::get<0>(accs) << " training accuracy; "
      << std::get<1>(accs) << " testing accuracy." << endl;


  // global update 1
  c.tic();
  Gs.ProxCSLUpdate(-1, alpha, adapt);
  const double updateTime1 = c.toc();
  const std::tuple<double, double> accs1 = TrainTestAccuracy(Gs, trainData, trainLabels, testData, testLabels);
  f << "pcsl1-" << partitions << "," << start << "," << lambda << ","
      << pow << "," << Gs.modelNonzeros << "," << std::get<0>(accs1) << ","
      << std::get<1>(accs1) << "," << updateTime1 << endl;

  // global update 2
  c.tic();
  Gs.ProxCSLUpdate(-1, alpha, adapt);
  const double updateTime2 = c.toc();
  const std::tuple<double, double> accs2 = TrainTestAccuracy(Gs, trainData, trainLabels, testData, testLabels);
  f << "pcsl2-" << partitions << "," << start << "," << lambda << ","
      << pow << "," << Gs.modelNonzeros << "," << std::get<0>(accs2) << ","
      << std::get<1>(accs2) << "," << updateTime2 << endl;

  // global update 3
  c.tic();
  Gs.ProxCSLUpdate(-1, alpha, adapt);
  const double updateTime3 = c.toc();
  const std::tuple<double, double> accs3 = TrainTestAccuracy(Gs, trainData, trainLabels, testData, testLabels);
  f << "pcsl3-" << partitions << "," << start << "," << lambda << ","
      << pow << "," << Gs.modelNonzeros << "," << std::get<0>(accs3) << ","
      << std::get<1>(accs3) << "," << updateTime3 << endl;

  // global update 4
  c.tic();
  Gs.ProxCSLUpdate(-1, alpha, adapt);
  const double updateTime4 = c.toc();
  const std::tuple<double, double> accs4 = TrainTestAccuracy(Gs, trainData, trainLabels, testData, testLabels);
  f << "pcsl4-" << partitions << "," << start << "," << lambda << ","
      << pow << "," << Gs.modelNonzeros << "," << std::get<0>(accs4) << ","
      << std::get<1>(accs4) << "," << updateTime4 << endl;


  // Instead of having a 3-level nested parallel run (over lambdas, partitions,
  // and then LIBLINEAR runs), we'll treat lambdas and partitions as OpenMP
  // tasks, allowing us a little more flexibility to assign them.
  for (size_t i = start + 1; i < count; ++i)
  {
    const double powThread = maxReg - (step * i);
    const double lambdaThread = std::pow(10.0, powThread);

    GlobalStep<arma::sp_mat> GsThread(Gs);

    arma::wall_clock cThread;
    cThread.tic();
    GsThread.lambda = lambdaThread;
    GsThread.numThreads = totalThreads;
    GsThread.Retrain(); // same dataset

    // Set the initialization for w_0
    if (start_mode == 1)
      GsThread.NaiveRetrain();
    else if (start_mode == 2)
      GsThread.Retrain();
    else if (start_mode == 0)
    {
      GsThread.NaiveRetrain();
      for (int j = 0; j < trainData.n_rows; ++j)
        GsThread.model[j] = 0;
        GsThread.modelNonzeros = arma::accu(GsThread.model != 0);
    }
        
    const double initTrainTimeThread = cThread.toc();

    const std::tuple<double, double> accsThread = TrainTestAccuracy(
        GsThread, trainData, trainLabels, testData, testLabels);

    #pragma omp critical
    {
      f << "pcsl-" << partitions << "," << i << "," << lambdaThread << ","
          << powThread << "," << GsThread.modelNonzeros << ","
          << std::get<0>(accsThread) << "," << std::get<1>(accsThread)
          << "," << initTrainTimeThread << endl;
      cout << "proxCSL, " << partitions << " partitions, lambda 10^"
          << powThread << ": " << initTrainTimeThread << "s training time; "
          << GsThread.modelNonzeros << " nonzeros; "
          << std::get<0>(accsThread) << " training accuracy; "
          << std::get<1>(accsThread) << " testing accuracy." << endl;
    }

    // global update 1
    c.tic();
    GsThread.ProxCSLUpdate(-1, alpha, adapt);
    const double updateTimeThread1 = c.toc();
    const std::tuple<double, double> accsThread1 = TrainTestAccuracy(
        GsThread, trainData, trainLabels, testData, testLabels);
    f << "pcsl1-" << partitions << "," << i << "," << lambdaThread << ","
        << powThread << "," << GsThread.modelNonzeros << ","
        << std::get<0>(accsThread1) << "," << std::get<1>(accsThread1)
        << "," << updateTimeThread1 << endl;

    // global update 2
    c.tic();
    GsThread.ProxCSLUpdate(-1, alpha, adapt);
    const double updateTimeThread2 = c.toc();
    const std::tuple<double, double> accsThread2 = TrainTestAccuracy(
        GsThread, trainData, trainLabels, testData, testLabels);
    f << "pcsl2-" << partitions << "," << i << "," << lambdaThread << ","
        << powThread << "," << GsThread.modelNonzeros << ","
        << std::get<0>(accsThread2) << "," << std::get<1>(accsThread2)
        << "," << updateTimeThread2 << endl;

    // global update 3
    c.tic();
    GsThread.ProxCSLUpdate(-1, alpha, adapt);
    const double updateTimeThread3 = c.toc();
    const std::tuple<double, double> accsThread3 = TrainTestAccuracy(
        GsThread, trainData, trainLabels, testData, testLabels);
    f << "pcsl3-" << partitions << "," << i << "," << lambdaThread << ","
        << powThread << "," << GsThread.modelNonzeros << ","
        << std::get<0>(accsThread3) << "," << std::get<1>(accsThread3)
        << "," << updateTimeThread3 << endl;

    // global update 4
    c.tic();
    GsThread.ProxCSLUpdate(-1, alpha, adapt);
    const double updateTimeThread4 = c.toc();
    const std::tuple<double, double> accsThread4 = TrainTestAccuracy(
        GsThread, trainData, trainLabels, testData, testLabels);
    f << "pcsl4-" << partitions << "," << i << "," << lambdaThread << ","
        << powThread << "," << GsThread.modelNonzeros << ","
        << std::get<0>(accsThread4) << "," << std::get<1>(accsThread4)
        << "," << updateTimeThread4 << endl;
  }
}
