/**
 * sweep_mpi_dane.cpp
 *
 * Sweep MPI OWA logistic regression with DANE updates across a range of
 * lambda values, storing the train/test accuracies in a CSV file.
 */
#include <mpi.h>
#include "mpi_dane.hpp"
#include "../libsvm.hpp"
#include "../data_utils.hpp"
#include <iostream>

using namespace std;

void help(char** argv)
{
  cout << "Usage: " << argv[0] << " <input_data_basename> <test_data_basename> "
       << "<extension> data_dim output_file.csv seed min_reg max_reg count [start "
       << " [verbose]]" << endl
       << endl
       << " - note: first two arguments should be basename of data;"
       << endl
       << "     data should be stored as input_data.partition.ext"
       << endl
       << "     for some extension ext (svm/csv/etc.)"
       << endl
       << " - note: data dimensionality must be provided"
       << endl
       << " - note: lambda values are between 10^{min_reg} and 10^{max_reg}"
       << endl
       << " - if start is given, the grid will start from that index"
       << endl
       << " - verbose output is given if *any* argument is given"
       << endl
       << " - number of partitions is set by number of MPI workers"
       << endl;
}

int main(int argc, char** argv)
{
  // Make sure we got the right number of arguments.
  if (argc != 10 && argc != 11 && argc != 12)
  {
    help(argv);
    exit(1);
  }

  const std::string inputFilePrefix(argv[1]);
  const std::string testFilePrefix(argv[2]);
  const std::string extension(argv[3]);
  const size_t dataDim = atoi(argv[4]);
  const std::string outputFile(argv[5]);
  const size_t seed = atoi(argv[6]);
  const double minReg = atof(argv[7]);
  const double maxReg = atof(argv[8]);
  const size_t count = atoi(argv[9]);
  const size_t start = (argc == 11) ? atoi(argv[10]) : 0;
  const bool verbose = (argc == 12);

  // We don't load the data here but instead let each MPI worker do that.
  srand(seed);
  arma::arma_rng::set_seed(seed);

  MPI_Init(NULL, NULL);
  int worker, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worker);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  fstream f;
  if (worker == 0)
  {
    f = fstream(outputFile, fstream::out | fstream::ate);
    if (!f.is_open())
    {
      std::cerr << "Failed to open output file '" << outputFile << "'!"
          << std::endl;
      exit(1);
    }

    if (start == 0)
    {
      f << "method,index,lambda,lambda_pow,nnz,train_acc,test_acc,train_time" << endl;
    }
  }

  arma::wall_clock c;
  c.tic();
  const double step = (maxReg - minReg) / (count - 1);
  double pow = maxReg - (step * start);
  double lambda = std::pow(10.0, pow);

  // This creates the LIBLINEAR features.
  MPIDANE owa(inputFilePrefix, testFilePrefix, extension, dataDim, lambda,
      verbose);
  for (size_t i = start + 1; i < count; ++i)
  {
    // Make sure no workers are hung up on something, so that our timing is
    // accurate.
    MPI_Barrier(MPI_COMM_WORLD);

    pow = maxReg - (step * i);
    lambda = std::pow(10.0, pow);

    // Step 0: Train OWA
    arma::wall_clock c;
    c.tic();
    owa.lambda = lambda;
    owa.TrainNaive();
    const double trainTime = c.toc();

    MPI_Barrier(MPI_COMM_WORLD);

    // Distribute models to all workers for accurate train/test accuracy
    // calculations.
    owa.DistributeModel();

    const double trainAcc = owa.TrainAccuracy();
    const double testAcc = owa.TestAccuracy();
    const size_t nonzeros = arma::accu(owa.model != 0.0);

    if (worker == 0)
    {
      f << "dane-" << worldSize << "," << i << "," << lambda << ","
          << pow << "," << nonzeros << ","
          << trainAcc << "," << testAcc << "," << trainTime << endl;
      cout << "DANE, " << worldSize << " partitions, lambda 10^"
          << pow << ": " << trainTime << "s training time; "
          << nonzeros << " nonzeros; "
          << trainAcc << " training accuracy; "
          << testAcc << " testing accuracy." << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 1: Dane Update 1
    c.tic();
    owa.Update();
    const double updateTime1 = c.toc() + trainTime;

    MPI_Barrier(MPI_COMM_WORLD);

    // Distribute models to all workers for accurate train/test accuracy
    // calculations.
    owa.DistributeModel();

    const double trainAcc1 = owa.TrainAccuracy();
    const double testAcc1 = owa.TestAccuracy();
    const size_t nonzeros1 = arma::accu(owa.model != 0.0);

    if (worker == 0)
    {
      f << "dane1-" << worldSize << "," << i << "," << lambda << ","
          << pow << "," << nonzeros1 << ","
          << trainAcc1 << "," << testAcc1 << "," << updateTime1 << endl;
      cout << "DANE1, " << worldSize << " partitions, lambda 10^"
          << pow << ": " << updateTime1 << "s training time; "
          << nonzeros1 << " nonzeros; "
          << trainAcc1 << " training accuracy; "
          << testAcc1 << " testing accuracy." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 2: Dane Update 2
    c.tic();
    owa.Update();
    const double updateTime2 = c.toc() + updateTime1;

    MPI_Barrier(MPI_COMM_WORLD);

    // Distribute models to all workers for accurate train/test accuracy
    // calculations.
    owa.DistributeModel();

    const double trainAcc2 = owa.TrainAccuracy();
    const double testAcc2 = owa.TestAccuracy();
    const size_t nonzeros2 = arma::accu(owa.model != 0.0);

    if (worker == 0)
    {
      f << "dane2-" << worldSize << "," << i << "," << lambda << ","
          << pow << "," << nonzeros2 << ","
          << trainAcc2 << "," << testAcc2 << "," << updateTime2 << endl;
      cout << "DANE2, " << worldSize << " partitions, lambda 10^"
          << pow << ": " << updateTime2 << "s training time; "
          << nonzeros2 << " nonzeros; "
          << trainAcc2 << " training accuracy; "
          << testAcc2 << " testing accuracy." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: Dane Update 3
    c.tic();
    owa.Update();
    const double updateTime3 = c.toc();

    MPI_Barrier(MPI_COMM_WORLD);

    // Distribute models to all workers for accurate train/test accuracy
    // calculations.
    owa.DistributeModel();

    const double trainAcc3 = owa.TrainAccuracy();
    const double testAcc3 = owa.TestAccuracy();
    const size_t nonzeros3 = arma::accu(owa.model != 0.0);

    if (worker == 0)
    {
      f << "dane3-" << worldSize << "," << i << "," << lambda << ","
          << pow << "," << nonzeros3 << ","
          << trainAcc3 << "," << testAcc3 << "," << updateTime3 << endl;
      cout << "DANE3, " << worldSize << " partitions, lambda 10^"
          << pow << ": " << updateTime3 << "s training time; "
          << nonzeros3 << " nonzeros; "
          << trainAcc3 << " training accuracy; "
          << testAcc3 << " testing accuracy." << endl;
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

  }

  MPI_Finalize();
}
