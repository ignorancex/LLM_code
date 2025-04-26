#include <string>
#include <sstream>
#include "mpi.h"
#include "dynamic_scaling.hpp"

/*
 * This is the test program for Dynamic Scaling (Scale In / Scale Out)
 */
int main(int argc, char *argv[]) {
  int NUM_ADD = 0;
  int NUM_REMOVE = 0;
  std::string ADD_HOST;

  MPI_Init(&argc, &argv);
  int initRank, initSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &initRank);
  MPI_Comm_size(MPI_COMM_WORLD, &initSize);

  /* Check the process is child or not */
  MPI_Comm parentCom = MPI_COMM_NULL;
  MPI_Comm_get_parent(&parentCom);
  bool isInitProcesses = parentCom == MPI_COMM_NULL;

  if (isInitProcesses && argc < 2) {
    if (initRank == 0) {
      std::cout << "How to use: $ mpirun -np <# Init MPI Processes> " << argv[0] << " <# Additional> <# Removal> [<additional hosts(comma separated)>]" << std::endl;
      std::cout << "  Ex.)    : $ mpirun -np 10 " << argv[0] << " 20 10 host1,host2,host3" << std::endl;

    }
    MPI_Finalize();
    exit(0);
  }

  if (isInitProcesses) {
    NUM_ADD = std::atoi(argv[1]);
    NUM_REMOVE = std::atoi(argv[2]);
    if (argc == 4) {
      ADD_HOST = std::string(argv[3]);
    }
  }

  DynamicScaler dynamicScaler;
  dynamicScaler.LOG_ = true;

  ///////////////////////////////////////////////////
  /////////        Scale Out start          /////////
  ///////////////////////////////////////////////////
  double startScaleOut = MPI_Wtime();
  MPI_Comm newCommAfterScaling;
  if (isInitProcesses) {
    /* Run in Initial Processes */
    std::string commandForNewProcess = std::string(argv[0]); // the same program runs in new process
    MPI_Comm initCommWorld = MPI_COMM_NULL;
    MPI_Comm_dup(MPI_COMM_WORLD, &initCommWorld);

    std::vector<std::string> hosts;
    if (ADD_HOST.size() != 0) {
      std::istringstream iss(ADD_HOST);
      std::string host;
      while (std::getline(iss, host, ',')) {
        hosts.push_back(host + " slots=32");
      }
    }

    dynamicScaler.scaleOut(initCommWorld, NUM_ADD, commandForNewProcess, newCommAfterScaling, hosts);
    MPI_Comm_free(&initCommWorld);
  } else {
    /* Run in Additional Processes */
    dynamicScaler.initNewProcess(newCommAfterScaling);
  }
  MPI_Barrier(newCommAfterScaling);
  double endScaleOut = MPI_Wtime();
  ///////////////////////////////////////////////////
  /////////        Scale Out end           //////////
  ///////////////////////////////////////////////////


  int afterScaleOutRank, afterScaleOutSize;
  MPI_Comm_rank(newCommAfterScaling, &afterScaleOutRank);
  MPI_Comm_size(newCommAfterScaling, &afterScaleOutSize);

  /* Check Scale Out */
  if (isInitProcesses) {
    /* Run in Initial Processes */
    if (initRank == 0) {
      std::cout << "=========================================" << std::endl;
      std::cout << "  # of Init Processes : " << initSize << std::endl;
      std::cout << "  # of Add Processes  : " << NUM_ADD;
      if (NUM_ADD == (afterScaleOutSize - initSize)) {std::cout << " OK " << std::endl;}
      else {std::cout << " NG!!! Check " << __LINE__ << " at " << __FILE__ << std::endl;}
      std::cout << "  # of Total Processes: " << afterScaleOutSize << std::endl;
      std::cout << "  Scale Out Time (sec): " << (endScaleOut - startScaleOut) << std::endl;
      std::cout << "=========================================" << std::endl;
    }
  }

  MPI_Barrier(newCommAfterScaling);

  ///////////////////////////////////////////////////
  /////////        Scale In start          //////////
  ///////////////////////////////////////////////////
  MPI_Comm newCommAfterScalingIn = MPI_COMM_NULL;
  int tmp = 0;
  MPI_Allreduce(&NUM_REMOVE, &tmp, 1, MPI_INT, MPI_MAX, newCommAfterScaling);
  NUM_REMOVE = tmp;
  double startScaleIn = MPI_Wtime();
  bool isRemoving = afterScaleOutRank >= afterScaleOutSize - NUM_REMOVE; // Remove from the tail rank
  bool hostCanBeRemoved = false;

  dynamicScaler.scaleIn(newCommAfterScaling, isRemoving, newCommAfterScalingIn, hostCanBeRemoved);
  if (isRemoving) {
    if (hostCanBeRemoved) {
      char myhostname[256]; gethostname(myhostname, sizeof(myhostname));
      std::cout << "Host: " << myhostname << " don't have any MPI processes. This host can be terminated." << std::endl;
    }
    // Kill the removing host
    exit(0);
  }

  MPI_Comm_free(&newCommAfterScaling);
  MPI_Barrier(newCommAfterScalingIn);
  double endScaleIn = MPI_Wtime();
  ///////////////////////////////////////////////////
  /////////        Scale In end            //////////
  ///////////////////////////////////////////////////


  int afterScaleInRank, afterScaleInSize;
  MPI_Comm_rank(newCommAfterScalingIn, &afterScaleInRank);
  MPI_Comm_size(newCommAfterScalingIn, &afterScaleInSize);

  /* Check Scale In */
  if (isInitProcesses) {
    int initRank, initSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &initRank);
    MPI_Comm_size(MPI_COMM_WORLD, &initSize);
    /* Run in Initial Processes */
    if (initRank == 0) {
      std::cout << "=========================================" << std::endl;
      std::cout << "  # of Init Processes  : " << afterScaleOutSize << std::endl;
      std::cout << "  # of Remove Processes: " << NUM_REMOVE;
      if (NUM_REMOVE == (afterScaleOutSize - afterScaleInSize)) {std::cout << " OK " << std::endl;}
      else {std::cout << " NG!!! Check " << __LINE__ << " at " << __FILE__ << std::endl;}
      std::cout << "  # of Total Processes : " << afterScaleInSize << std::endl;
      std::cout << "  Scale In Time (sec)  : " << (endScaleIn - startScaleIn) << std::endl;
      std::cout << "=========================================" << std::endl;
    }
  }

  MPI_Barrier(newCommAfterScalingIn);
  MPI_Comm_free(&newCommAfterScalingIn);
  MPI_Finalize();
}

