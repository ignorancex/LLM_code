/*
 * GPUMotifCalculator.cpp
 *
 *  Created on: Dec 2, 2018
 *
 */

#include "../includes/GPUMotifCalculator.h"
#include "../includes/MotifVariationConstants.h"
#include "device_launch_parameters.h"
#include "../includes/CacheGraph.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort =
true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   Global managed variables   ////////////////////////////////////////////////
__managed__ unsigned int globalNumOfNodes;
__managed__ unsigned int globalNumOfMotifs;
__managed__ unsigned int globalNumOfEdges;

__managed__ bool globalDirected;

// DEVICE VARIABLES

//Pointers to the device vectors declared above - no need to delete as the device_vectors live on the stack
__managed__ unsigned int* globalDevicePointerMotifVariations;
__managed__ unsigned int* globalDevicePointerRemovalIndex;
__managed__ unsigned int* globalDevicePointerSortedNodesByDegree;

// For the original graph
__managed__ int64* globalDeviceOriginalGraphOffsets;
__managed__ unsigned int* globalDeviceOriginalGraphNeighbors;

// For the full graph
__managed__ int64* globalDeviceFullGraphOffsets;
__managed__ unsigned int* globalDeviceFullGraphNeighbors;

// Feature array
__managed__ unsigned int* globalDeviceFeatures;

/////////////////////////////   END Global managed variables   /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////

void GPUMotifCalculator::init() {

    CacheGraph inverse(true);
    mGraph->InverseGraph(inverse);
    mGraph->CureateUndirectedGraph(inverse, fullGraph);
    this->numOfNodes = this->mGraph->GetNumberOfNodes();
    this->numOfEdges = this->mGraph->GetNumberOfEdges();
    this->LoadMotifVariations(level, directed);
    this->SetAllMotifs();
    this->SetSortedNodes();
    this->SetRemovalIndex();
    this->InitFeatureCounters();
    this->CopyAllToDevice();
}

GPUMotifCalculator::GPUMotifCalculator(int level, bool directed, int cudaDevice) :
        directed(directed), nodeVariations(NULL), allMotifs(NULL), removalIndex(
        NULL), sortedNodesByDegree(NULL), fullGraph(false), numOfMotifs(0), deviceFeatures(
        NULL), cudaDevice(cudaDevice) {
    //check level
    if (level != 3 && level != 4)
        throw invalid_argument("Level must be 3 or 4");
    this->level = level;
    this->features = new std::vector<vector<unsigned int> *>;
}

void GPUMotifCalculator::InitFeatureCounters() {
    for (int node = 0; node < numOfNodes; node++) {
        vector<unsigned int> *motifCounter = new vector<unsigned int>;
        std::set<int> s(this->allMotifs->begin(), this->allMotifs->end());
        this->numOfMotifs = s.size() - 1;
        for (auto motif : s)
            if (motif != -1)
                motifCounter->push_back(0);

        features->push_back(motifCounter);
    }
    delete this->allMotifs;
}

void GPUMotifCalculator::LoadMotifVariations(int level, bool directed) {
    const char* motifVariations[4] = { undirected3, directed3, undirected4,
                                       directed4 };
    const int numOfMotifsOptions[4] = { 8, 64, 64, 4096 };

    int variationIndex = 2 * (level - 3) + (directed ? 1 : 0);
    this->nodeVariations = new std::vector<unsigned int>(
            numOfMotifsOptions[variationIndex]);
    std::istringstream f(motifVariations[variationIndex]);
    std::string line;
    std::string a, b;
    while (getline(f, line)) {
        int x, y;
        int n = line.find(" ");
        a = line.substr(0, n);
        b = line.substr(n);
        try {
            x = stoi(a);
            y = stoi(b);
        } catch (exception &e) {
            y = -1;
        }

        (*nodeVariations)[x] = y;
    }
}

void GPUMotifCalculator::SetAllMotifs() {
    this->allMotifs = new std::vector<int>();

    for (const auto &x : *(this->nodeVariations))
        this->allMotifs->push_back(x);
}

void GPUMotifCalculator::SetSortedNodes() {
    this->sortedNodesByDegree = mGraph->SortedNodesByDegree();
//    if (globalDirected){//must sort before
//        cout<<"directed";
//        std::vector<unsigned int>* sortedNodes = new std::vector<unsigned int>();
//        sortedNodes->reserve(this->numOfNodes);
//
////        std::vector<unsigned int> nodeDegrees = CacheGraph::ComputeNodeDegrees();
//        std::vector<NodeWithDegree> nodesWithDegrees;
//        nodesWithDegrees.reserve(this->numOfNodes);
//        auto offsetList=this->fullGraph.GetOffsetList();
//        for (unsigned int node = 0; node < this->numOfNodes; node++){
//            unsigned int root= node;//globalDevicePointerSortedNodesByDegree[node];
//            nodesWithDegrees.push_back( { node, (unsigned int)(offsetList[root+1]-offsetList[root]) });
//        }
//        std::sort(nodesWithDegrees.begin(), nodesWithDegrees.end(),
//                cmpNodesByDegree);
//
//        for (NodeWithDegree nd : nodesWithDegrees)
//            sortedNodes->push_back(nd.node);
//        this->sortedNodesByDegree=sortedNodes;
////        std::memcpy(this->devicePointerSortedNodesByDegree,
////                this->sortedNodesByDegree->data(),
////                sortedNodesByDegree->size() * sizeof(unsigned int));
////        this->SetRemovalIndex();
////        std::memcpy(this->devicePointerRemovalIndex, this->removalIndex->data(),
////                removalIndex->size() * sizeof(unsigned int));
////        globalDevicePointerRemovalIndex = this->devicePointerRemovalIndex;
////        globalDevicePointerSortedNodesByDegree =
////            this->devicePointerSortedNodesByDegree;
//    }
}
/**
 * We iterate over the list of sorted nodes.
 * If node p is in index i in the list, it means that it will be removed after the i-th iteration,
 * or conversely that it is considered "not in the graph" from iteration i+1 onwards.
 * This means that any node with a removal index of j is not considered from iteration j+1.
 * (The check is (removalIndex[node] > currentIteration).
 */
void GPUMotifCalculator::SetRemovalIndex() {
    this->removalIndex = new std::vector<unsigned int>();
    for (int i = 0; i < numOfNodes; i++) {
        removalIndex->push_back(0);
    }
    for (unsigned int index = 0; index < numOfNodes; index++) {
        auto node = sortedNodesByDegree->at(index);
        removalIndex->at(node) = numOfNodes-1-index;
    }
}

void GPUMotifCalculator::CopyAllToDevice() {

    /*
     * 1) Allocate unified memory
     * 2) Copy vectors to the memory
     * 3) delete the memory in d'tor
     */
    gpuErrchk(
            cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                               size_t(10) * size_t(numOfNodes) * size_t(numOfNodes)
                               * sizeof(int64)));
    size_t currentLimit;
    gpuErrchk(cudaDeviceGetLimit(&currentLimit, cudaLimitMallocHeapSize));

    gpuErrchk(
            cudaMallocManaged(&(this->devicePointerMotifVariations),
                              nodeVariations->size() * sizeof(unsigned int)));

    std::memcpy(this->devicePointerMotifVariations,
                &((*(this->nodeVariations))[0]),
                nodeVariations->size() * sizeof(unsigned int));
    // Removal index
    gpuErrchk(
            cudaMallocManaged(&(this->devicePointerRemovalIndex),
                              removalIndex->size() * sizeof(unsigned int)));
    std::memcpy(this->devicePointerRemovalIndex, this->removalIndex->data(),
                removalIndex->size() * sizeof(unsigned int));
    //Sorted nodes
    gpuErrchk(
            cudaMallocManaged(&(this->devicePointerSortedNodesByDegree),
                              sortedNodesByDegree->size() * sizeof(unsigned int)));
    std::memcpy(this->devicePointerSortedNodesByDegree,
                this->sortedNodesByDegree->data(),
                sortedNodesByDegree->size() * sizeof(unsigned int));

    // Feature matrix
    unsigned int size = this->numOfNodes * this->numOfMotifs // was this->nodeVariations->size()
                        * sizeof(unsigned int);
    gpuErrchk(cudaMallocManaged(&(this->deviceFeatures), size));

    // Original graph
    gpuErrchk(
            cudaMallocManaged(&deviceOriginalGraphOffsets,
                              (this->numOfNodes + 1) * sizeof(int64)));
    gpuErrchk(
            cudaMallocManaged(&deviceOriginalGraphNeighbors,
                              (this->numOfEdges) * sizeof(unsigned int)));
    std::memcpy(deviceOriginalGraphOffsets, this->mGraph->GetOffsetList(),
                (this->numOfNodes + 1) * sizeof(int64));
    std::memcpy(deviceOriginalGraphNeighbors, this->mGraph->GetNeighborList(),
                (this->numOfEdges) * sizeof(unsigned int));

    // Full graph
    gpuErrchk(
            cudaMallocManaged(&deviceFullGraphOffsets,
                              (this->fullGraph.GetNumberOfNodes() + 1) * sizeof(int64)));
    gpuErrchk(
            cudaMallocManaged(&deviceFullGraphNeighbors,
                              (this->fullGraph.GetNumberOfEdges())
                              * sizeof(unsigned int)));
    std::memcpy(deviceFullGraphOffsets, this->fullGraph.GetOffsetList(),
                (this->fullGraph.GetNumberOfNodes() + 1) * sizeof(int64));
    std::memcpy(deviceFullGraphNeighbors, this->fullGraph.GetNeighborList(),
                (this->fullGraph.GetNumberOfEdges()) * sizeof(unsigned int));


    //Assign to global variables

    globalNumOfNodes = this->numOfNodes;
    globalNumOfMotifs = this->numOfMotifs;
    globalNumOfEdges = this->numOfEdges;
    globalDirected = this->directed;

    globalDevicePointerMotifVariations = this->devicePointerMotifVariations;
    globalDevicePointerRemovalIndex = this->devicePointerRemovalIndex;
    globalDevicePointerSortedNodesByDegree =
            this->devicePointerSortedNodesByDegree;

    globalDeviceOriginalGraphOffsets = this->deviceOriginalGraphOffsets;
    globalDeviceOriginalGraphNeighbors = this->deviceOriginalGraphNeighbors;
    globalDeviceFullGraphOffsets = this->deviceFullGraphOffsets;
    globalDeviceFullGraphNeighbors = this->deviceFullGraphNeighbors;
    globalDeviceFeatures = this->deviceFeatures;

}

// Kernel and friend functions
__global__
void Motif3Kernel() {
    //GPU allocations
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    auto n = globalNumOfNodes;

    int y_index=blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;
    //offsets of the neighbors graph
    const int64 *offsets = globalDeviceFullGraphOffsets;
    // all neighbors - ancestors and descendants
    const unsigned int *neighbors = globalDeviceFullGraphNeighbors;
    for (int i = index; i < n; i += stride){
        unsigned int root=globalDevicePointerSortedNodesByDegree[i];
        int idx_root = globalDevicePointerRemovalIndex[root];
        int64 n1_idx_start = offsets[root]+y_index;
        for (int64 n1_idx = n1_idx_start; n1_idx < offsets[root+1]; n1_idx += stride_y){
            unsigned int n1 = neighbors[n1_idx];

            //n1 already handled
            if (globalDevicePointerRemovalIndex[n1] <= idx_root)
                continue;

             // CASE 1: MOTIF 3 OF DEPTH 2

            //loop on second neighbors
            for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) {
                unsigned int n2 = neighbors[n2_idx];
                //n2 already handled
                if (globalDevicePointerRemovalIndex[n2] <= idx_root)
                    continue;
                bool edgeExists = AreNeighbors(root,n2) || AreNeighbors(n2,root);
                //if n2 is actually a first neighbor of root
                if (edgeExists) {
                    //n2 is after n1 (stops counting the motif twice)
                    if (edgeExists && n1 < n2) {
                        unsigned int arr[] = { root, n1, n2 };
                        //update motif counter [r,n1,n2]
                        GroupUpdater(arr, 3);
                    }
                } else {
                    unsigned int arr[] = { root, n1, n2 };
                    // update motif counter [r,n1,n2]
                    GroupUpdater(arr, 3);
                }										   // end ELSE
            }

            // CASE 2: MOTIF 3 OF DEPTH 1
            // All 2-combinations of root's neighbors
            for (int64 j = n1_idx + 1; j < offsets[root + 1]; j++) {
                //combination
                unsigned int n2 = neighbors[j];
                // motif already handled
                if (globalDevicePointerRemovalIndex[n1] <= idx_root
                    || globalDevicePointerRemovalIndex[n2] <= idx_root)
                    continue;
                // check n1, n2 not neighbors, and check n1 < n2 to avoid double counting
                if ((n1 < n2) && !(AreNeighbors(n1, n2) || AreNeighbors(n2, n1))) {
                    unsigned int arr[] = { root, n1, n2 };
                    // update motif counter [r,n1,n2]
                    GroupUpdater(arr, 3);
                }
            }
        }
    }
}
__global__
void Motif4Kernel() {
    //GPU allocations
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    auto n = globalNumOfNodes;
    if (index>n){
    return;}

    int y_index=blockIdx.y * blockDim.y + threadIdx.y;
    int stride_y = blockDim.y * gridDim.y;
    if (y_index>n){
    return;}

    //offsets of the neighbors graph
    const int64 *offsets = globalDeviceFullGraphOffsets;

    // all neighbors - ancestors and descendants
    const unsigned int *neighbors = globalDeviceFullGraphNeighbors;

    for (int i = index; i < n; i += stride){
        long long int start=0;
        long long int stop=0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));

        unsigned int root=globalDevicePointerSortedNodesByDegree[i];
        int idx_root = globalDevicePointerRemovalIndex[root];

        int64 n1_idx_start = offsets[root]+y_index;

        if(y_index>=offsets[root + 1]-offsets[root]){
        return;}
        for (int64 n1_idx = n1_idx_start; n1_idx < offsets[root + 1]; n1_idx+=stride_y) { // loop first neighbors
            unsigned int n1 = neighbors[n1_idx];
            if (globalDevicePointerRemovalIndex[n1] <= idx_root) // n1 already handled
                continue;

             /* --Motif depth one Builder-- */
            // Case 1: Depth 1
            // Loop on All 3 combinations of root's neighbors
            for (int64 j = n1_idx + 1; j < offsets[root + 1]; j++) {
                if (j == offsets[root + 1] - 1) //if j is the last element, we can't add an element and therefore it's not a 3-combination
                    continue;
                for (int64 k = j + 1; k < offsets[root + 1]; k++) {
                    // 3 combinations of root's neighbors
                    unsigned int n11 = neighbors[n1_idx];
                    unsigned int n12 = neighbors[j];
                    unsigned int n13 = neighbors[k];
                    // In this case, the motif already handled
                    if (globalDevicePointerRemovalIndex[n11] <= idx_root
                        || globalDevicePointerRemovalIndex[n12] <= idx_root
                        || globalDevicePointerRemovalIndex[n13] <= idx_root)
                        continue;
                    unsigned int arr[] = { root, n11, n12, n13 };
                    // update motif counter [r,n11,n12,n13]
                    GroupUpdater(arr, 4);
                }
            }

            // All other cases

            // Case 2: 2*Depth 1 and 1*Depth 2
            /* --Motif depth 2 Builder *//*
            *//* Motifs of the form root, two first neighbors and a second neighbor (root-n1-n2-n11) */
            for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (again)
                unsigned int n2 = neighbors[n2_idx];
                if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
                    continue;
                for (int64 n11_idx = offsets[root]; n11_idx < offsets[root + 1];
                     n11_idx++) { // loop first neighbors
                    unsigned int n11 = neighbors[n11_idx];
                    if (globalDevicePointerRemovalIndex[n11] <= idx_root) // n2 already handled
                        continue;
                    /* If there is an edge between n1 and n11, we would have already counted the motif as a motif of depth 2.
                     If no such edge exists, we only want to count the motif once.  */
                    if (!((AreNeighbors(root,n2)) || (AreNeighbors(n2,root))) && n11 != n1) {
                        bool edgeExists = AreNeighbors(n2, n11)
                                          || AreNeighbors(n11, n2);
                        if (!edgeExists || (edgeExists && n1 < n11)) {
                            unsigned int arr[] = { root, n1, n11, n2 };
                            // update motif counter [r,n1,n11,n2]
                            GroupUpdater(arr, 4);
                        }												// end if
                    }
                }								// end loop INNER FIRST NEIGHBORS
            }									// end loop SECOND NEIGHBORS AGAIN


            // Case 3: 1* Depth 1 and 2* Depth 2
            /* Motifs of the form root, a first neighbors and two second neighbors - The case of root-n1-n21-n22
             2-combinations on second neighbors */
            for (int64 l = offsets[n1]; l < offsets[n1 + 1]; l++) {
                for (int64 j = l + 1; j < offsets[n1 + 1]; j++) {

                    unsigned int n21 = neighbors[l];
                    unsigned int n22 = neighbors[j];
                    // In this case - motif already handled
                    if (globalDevicePointerRemovalIndex[n21] <= idx_root
                        || globalDevicePointerRemovalIndex[n22] <= idx_root)
                        continue;
                     // If both nodes are not neighbors of root (were seen at level 2) then the motif wasn't counted as a depth 1 motif
                    if (!AreNeighbors(root,n21) && !AreNeighbors(n21,root) && !AreNeighbors(root,n22) &&
                        !AreNeighbors(n22,root)) {
                        unsigned int arr[] = { root, n1, n21, n22 };
                        GroupUpdater(arr, 4); // update motif counter [r,n1,n21,n22]
                    }
                }
            } // end loop SECOND NEIGHBOR COMBINATIONS

            // Case 4: Depth 3
            /* Motif 4 depth 3 builder - The case of n1-n2-n3 */
            for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (third time's the charm)
                unsigned int n2 = neighbors[n2_idx];
                if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
                    continue;

                /* According to the rules we proved, check : */
                // n2 is not a first neighbor
                if(AreNeighbors(root,n2) || AreNeighbors(n2,root)) {
                    continue;
                }

                // loop third neighbors
                for (int64 n3_idx = offsets[n2]; n3_idx < offsets[n2 + 1];
                     n3_idx++) {
                    unsigned int n3 = neighbors[n3_idx];
                    // n3 already handled
                    if (globalDevicePointerRemovalIndex[n3] <= idx_root)
                        continue;

                    /* According to the proved rules, check that n3 is not a first neighbor, not a neighbor of the first neighbor n1 and
                    not the neighbor n1 itself. If it is any of them, continue */
                    if ((AreNeighbors(root,n3) || (AreNeighbors(n3,root))) || (AreNeighbors(n1,n3) || (AreNeighbors(n3,n1) )) || (n1==n3)) {
                        continue;
                    }
                    unsigned int arr[] = { root, n1, n2, n3 };
                    GroupUpdater(arr, 4);
                }									// end loop THIRD NEIGHBORS
            }				// end loop SECOND NEIGHBORS THIRD TIME'S THE CHARM

        } // end loop FIRST NEIGHBORS

        int numNeighbors = offsets[root+1]-offsets[root];

        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(stop));
        float elapsedTime=(stop-start)/1000000000.0;
//        printf("Elapsed time of Node number %d in Ythread %d with degree %d: %f s\n" ,root,threadIdx.y,numNeighbors,elapsedTime);
    }
}


vector<vector<unsigned int> *> *GPUMotifCalculator::Calculate() {
    if (globalDirected){//must sort before
        cout<<"directed";
        std::vector<unsigned int>* sortedNodes = new std::vector<unsigned int>();
        sortedNodes->reserve(this->numOfNodes);

        std::vector<NodeWithDegree> nodesWithDegrees;
        nodesWithDegrees.reserve(this->numOfNodes);
        for (unsigned int node = 0; node < this->numOfNodes; node++){
            unsigned int root= node;
            nodesWithDegrees.push_back( { node, (unsigned int)(globalDeviceFullGraphOffsets[root+1]-globalDeviceFullGraphOffsets[root]) });
        }
        std::sort(nodesWithDegrees.begin(), nodesWithDegrees.end(),
                cmpNodesByDegree);

        for (NodeWithDegree nd : nodesWithDegrees)
            sortedNodes->push_back(nd.node);
        this->sortedNodesByDegree=sortedNodes;
        std::memcpy(this->devicePointerSortedNodesByDegree,
                this->sortedNodesByDegree->data(),
                sortedNodesByDegree->size() * sizeof(unsigned int));
        this->SetRemovalIndex();
        std::memcpy(this->devicePointerRemovalIndex, this->removalIndex->data(),
                removalIndex->size() * sizeof(unsigned int));
        globalDevicePointerRemovalIndex = this->devicePointerRemovalIndex;
        globalDevicePointerSortedNodesByDegree =
            this->devicePointerSortedNodesByDegree;
    }
//    for (int i=0;i<=this->numOfNodes;i++){
//        unsigned int root= globalDevicePointerSortedNodesByDegree[i];
//        std::cout<<"Node number "<<root<<" Deg = "<<globalDeviceFullGraphOffsets[root+1]-globalDeviceFullGraphOffsets[root]<<" removal index "<<globalDevicePointerRemovalIndex[root]<<"\n";
//    }
    const int maxBlocksPerDim=32768;

    //Prefetch all relevant memory
    cudaSetDevice(this->cudaDevice);
    int device = -1;
    cudaGetDevice(&device);


    std::cout<<cudaPeekAtLastError();
    std::cout<<cudaGetErrorString(cudaPeekAtLastError());


    int offsetSize = this->numOfNodes + 1;
    int neighborSize = this->numOfEdges;

    cudaMemPrefetchAsync(globalDevicePointerMotifVariations,
                         nodeVariations->size() * sizeof(unsigned int), device, NULL);
    cudaMemPrefetchAsync(globalDevicePointerRemovalIndex,//
                         this->numOfNodes * sizeof(unsigned int), device, NULL);//
    cudaMemPrefetchAsync(globalDevicePointerSortedNodesByDegree,//
                         this->numOfNodes * sizeof(unsigned int), device, NULL);//
    cudaMemPrefetchAsync(globalDeviceOriginalGraphOffsets,//
                         offsetSize * sizeof(int64), device, NULL);//
    cudaMemPrefetchAsync(globalDeviceOriginalGraphNeighbors,
                         neighborSize * sizeof(unsigned int), device, NULL);
    cudaMemPrefetchAsync(globalDeviceFullGraphOffsets,
                         offsetSize * sizeof(int64), device, NULL);
    cudaMemPrefetchAsync(globalDeviceFullGraphNeighbors,//
                         neighborSize * sizeof(unsigned int), device, NULL);//
    cudaMemPrefetchAsync(globalDeviceFeatures,
                         (this->numOfNodes * this->numOfMotifs)
                         * sizeof(unsigned int), device, NULL);


    std::cout<<cudaPeekAtLastError();
    std::cout<<cudaGetErrorString(cudaPeekAtLastError());

    dim3 blockSize(16,16);
    int numBlocksX=(this->numOfNodes + blockSize.x - 1) / (blockSize.x);
    int numBlocksY=(this->numOfNodes + blockSize.y - 1) / (blockSize.y);

    numBlocksX=numBlocksX<maxBlocksPerDim?numBlocksX:maxBlocksPerDim;
    numBlocksY=numBlocksY<maxBlocksPerDim?numBlocksY:maxBlocksPerDim;
    std::cout<<"numBLocksX = "<<numBlocksX<<"\n";
    std::cout<<"numBLocksY = "<<numBlocksY<<"\n";
    dim3 numBlocks(numBlocksX,numBlocksY);

    if (this->level == 3) {
        Motif3Kernel<<<numBlocks, blockSize>>>();
    } else {
        Motif4Kernel<<<numBlocks, blockSize>>>();
    }
    std::cout<<cudaPeekAtLastError();
    std::cout<<cudaGetErrorString(cudaPeekAtLastError());
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::cout<<"num of motifs"+std::to_string(this->numOfMotifs)+"\n";
    std::cout<<"node variation"+std::to_string(this->nodeVariations->size())+"\n";
    for (int node = 0; node < this->numOfNodes; node++) {
        for (int motif = 0; motif < this->numOfMotifs; motif++) {
            this->features->at(node)->at(motif) = globalDeviceFeatures[motif
                                                                       + this->numOfMotifs * node];
        }
    }
    std::cout<<"Yes, its gpu\n";
    return this->features;
}

/*  3-Motif building algorithm
 3-motifs can only be of depth 1 or 2.
 we count the motifs in the following order:
1. Motifs of depth 2; then
2. Motifs of depth 1
*/
__device__
void Motif3Subtree(unsigned int root) {

/* Since cacheGraph cannot be modified at runtime, we hold Removal Index,
    which will be the first iteration where the node no longer exists in the graph.
    we will verify that the removal index of a node is lower than the removal index of the root of the current subtree.
    root_idx is also our current iteration */

    int idx_root = globalDevicePointerRemovalIndex[root];
    // all neighbors - ancestors and descendants
    const unsigned int *neighbors = globalDeviceFullGraphNeighbors;
    //offsets of the neighbors graph
    const int64 *offsets = globalDeviceFullGraphOffsets;

    // CASE 1: MOTIF 3 OF DEPTH 2
    //Loop on the first neighbors
    for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) {
        unsigned int n1 = neighbors[n1_idx];
        //n1 already handled
        if (globalDevicePointerRemovalIndex[n1] <= idx_root)
            continue;
        //loop on second neighbors
        for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) {
            unsigned int n2 = neighbors[n2_idx];
            //n2 already handled
            if (globalDevicePointerRemovalIndex[n2] <= idx_root)
                continue;
            bool edgeExists = AreNeighbors(root,n2) || AreNeighbors(n2,root);
            //if n2 is actually a first neighbor of root
            if (edgeExists) {
                //n2 is after n1 (stops counting the motif twice)
                if (edgeExists && n1 < n2) {
                    unsigned int arr[] = { root, n1, n2 };
                    //update motif counter [r,n1,n2]
                    GroupUpdater(arr, 3);
                }
            } else {
                unsigned int arr[] = { root, n1, n2 };
                // update motif counter [r,n1,n2]
                GroupUpdater(arr, 3);
            }										   // end ELSE
        }											// end LOOP_SECOND_NEIGHBORS

    } // end LOOP_FIRST_NEIGHBORS



    // CASE 2: MOTIF 3 OF DEPTH 1
    // All 2-combinations of root's neighbors
    for (int64 i = offsets[root]; i < offsets[root + 1]; i++) {
        for (int64 j = i + 1; j < offsets[root + 1]; j++) {
            //combination
            unsigned int n1 = neighbors[i];
            unsigned int n2 = neighbors[j];
            // motif already handled
            if (globalDevicePointerRemovalIndex[n1] <= idx_root
                || globalDevicePointerRemovalIndex[n2] <= idx_root)
                continue;
            // check n1, n2 not neighbors, and check n1 < n2 to avoid double counting
            if ((n1 < n2) && !(AreNeighbors(n1, n2) || AreNeighbors(n2, n1))) {
                unsigned int arr[] = { root, n1, n2 };
                // update motif counter [r,n1,n2]
                GroupUpdater(arr, 3);
            }
        }
    } // end loop COMBINATIONS_NEIGHBORS_N1
}


/*  4-Motif building algorithm
 4-motifs can be of depth 1, 2 or 3.
1. Motifs of depth 1
2. Motifs of depth 2 can be one of the following:
-  Motifs of the form root, two first neighbors and a second neighbor
-  Motifs of the form root, a first neighbors and two second neighbors
3. Motifs of depth 3.  can only be a chain of the root, and a first, second and third neighbors.
*/
__device__
void Motif4Subtree(unsigned int root) {

    // Same as motif3, idx_root is also our current iteration
    int idx_root = globalDevicePointerRemovalIndex[root];
    // all neighbors - ancestors and descendants
    const unsigned int *neighbors = globalDeviceFullGraphNeighbors;
    //offsets of the neighbors graph
    const int64 *offsets = globalDeviceFullGraphOffsets;

    /* --Motif depth one Builder-- */
    int64 end = offsets[root + 1];
    // Loop on All 3 combinations of root's neighbors
    for (int64 i = offsets[root]; i < end; i++) {
        for (int64 j = i + 1; j < end; j++) {
            if (j == end - 1) //if j is the last element, we can't add an element and therefore it's not a 3-combination
                continue;
            for (int64 k = j + 1; k < end; k++) {
                // 3 combinations of root's neighbors
                unsigned int n11 = neighbors[i];
                unsigned int n12 = neighbors[j];
                unsigned int n13 = neighbors[k];
                // In this case, the motif already handled
                if (globalDevicePointerRemovalIndex[n11] <= idx_root
                    || globalDevicePointerRemovalIndex[n12] <= idx_root
                    || globalDevicePointerRemovalIndex[n13] <= idx_root)
                    continue;
                unsigned int arr[] = { root, n11, n12, n13 };
                // update motif counter [r,n11,n12,n13]
                GroupUpdater(arr, 4);
            }
        }
    }

    // All other cases
    /* --Motif depth 2 Builder */
    for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
        unsigned int n1 = neighbors[n1_idx];
        if (globalDevicePointerRemovalIndex[n1] <= idx_root) // n1 already handled
            continue;


        /* Motifs of the form root, two first neighbors and a second neighbor (root-n1-n2-n11) */
        for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (again)
            unsigned int n2 = neighbors[n2_idx];
            if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
                continue;
            for (int64 n11_idx = offsets[root]; n11_idx < offsets[root + 1];
                 n11_idx++) { // loop first neighbors
                unsigned int n11 = neighbors[n11_idx];
                if (globalDevicePointerRemovalIndex[n11] <= idx_root) // n2 already handled
                    continue;
                /* If there is an edge between n1 and n11, we would have already counted the motif as a motif of depth 2.
                 If no such edge exists, we only want to count the motif once.  */
                if (!((AreNeighbors(root,n2)) || (AreNeighbors(n2,root))) && n11 != n1) {
                    bool edgeExists = AreNeighbors(n2, n11)
                                      || AreNeighbors(n11, n2);
                    if (!edgeExists || (edgeExists && n1 < n11)) {
                        unsigned int arr[] = { root, n1, n11, n2 };
                        // update motif counter [r,n1,n11,n2]
                        GroupUpdater(arr, 4);
                    }												// end if
                }
            }								// end loop INNER FIRST NEIGHBORS
        }									// end loop SECOND NEIGHBORS AGAIN

        /* Motifs of the form root, a first neighbors and two second neighbors - The case of root-n1-n21-n22
         2-combinations on second neighbors */
        end = offsets[n1 + 1];
        for (int64 i = offsets[n1]; i < end; i++) {
            for (int64 j = i + 1; j < end; j++) {

                unsigned int n21 = neighbors[i];
                unsigned int n22 = neighbors[j];
                // In this case - motif already handled
                if (globalDevicePointerRemovalIndex[n21] <= idx_root
                    || globalDevicePointerRemovalIndex[n22] <= idx_root)
                    continue;
                 // If both nodes are not neighbors of root (were seen at level 2) then the motif wasn't counted as a depth 1 motif
                if (!AreNeighbors(root,n21) && !AreNeighbors(n21,root) && !AreNeighbors(root,n22) &&
                    !AreNeighbors(n22,root)) {
                    unsigned int arr[] = { root, n1, n21, n22 };
                    GroupUpdater(arr, 4); // update motif counter [r,n1,n21,n22]
                }
            }
        } // end loop SECOND NEIGHBOR COMBINATIONS
    }

    /* Motif 4 depth 3 builder - The case of n1-n2-n3 */
    for (int64 n1_idx = offsets[root]; n1_idx < offsets[root + 1]; n1_idx++) { // loop first neighbors
        unsigned int n1 = neighbors[n1_idx];
        // n1 already handled
        if (globalDevicePointerRemovalIndex[n1] <= idx_root)
            continue;
        for (int64 n2_idx = offsets[n1]; n2_idx < offsets[n1 + 1]; n2_idx++) { // loop second neighbors (third time's the charm)
            unsigned int n2 = neighbors[n2_idx];
            if (globalDevicePointerRemovalIndex[n2] <= idx_root) // n2 already handled
                continue;

            /* According to the rules we proved, check : */
            // n2 is not a first neighbor
            if(AreNeighbors(root,n2) || AreNeighbors(n2,root)) {
                continue;
            }

            // loop third neighbors
            for (int64 n3_idx = offsets[n2]; n3_idx < offsets[n2 + 1];
                 n3_idx++) {
                unsigned int n3 = neighbors[n3_idx];
                // n2 already handled
                if (globalDevicePointerRemovalIndex[n3] <= idx_root)
                    continue;

                /* According to the proved rules, check that n3 is not a first neighbor, not a neighbor of the first neighbor n1 and
                not the neighbor n1 itself. If it is any of them, continue */
                if ((AreNeighbors(root,n3) || (AreNeighbors(n3,root))) || (AreNeighbors(n1,n3) || (AreNeighbors(n3,n1) )) || (n1==n3)) {
                    continue;
                }
                unsigned int arr[] = { root, n1, n2, n3 };
                GroupUpdater(arr, 4);
            }									// end loop THIRD NEIGHBORS
        }				// end loop SECOND NEIGHBORS THIRD TIME'S THE CHARM

    } // end loop FIRST NEIGHBORS

}

__device__
bool AreNeighbors(unsigned int p, unsigned int q) {
    int first = globalDeviceOriginalGraphOffsets[p],//first array element
    last = globalDeviceOriginalGraphOffsets[p + 1] - 1,	//last array element
    middle;		//mid point of search

    while (first <= last) {
        middle = (int)(first + last) / 2; //this finds the mid point
        if (globalDeviceOriginalGraphNeighbors[middle] == q) {
            return true;
        } else if (globalDeviceOriginalGraphNeighbors[middle] < q)
        {
            first = middle + 1;      //if it's in the upper half

        } else {
            last = middle - 1; // if it's in the lower half
        }
    }
    return false;  // not found

}

__device__
void GroupUpdater(unsigned int group[], int size) {
    int groupNumber = GetGroupNumber(group, size);
    int motifNumber = (globalDevicePointerMotifVariations)[groupNumber];
    if (motifNumber != -1) {
        for (int i = 0; i < size; i++)
            atomicAdd(
                    globalDeviceFeatures
                    + (motifNumber + globalNumOfMotifs * group[i]), 1);	//atomic add + access as 1D array : features[motif + M*node] // @suppress("Function cannot be resolved")
        // where M is the number of motifs
    }
}

__device__
int GetGroupNumber(unsigned int group[], int size) {
    int sum = 0;
    int power = 1;
    bool hasEdge;
    if (globalDirected) {
        // Use permutations
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (i != j) {
                    hasEdge = AreNeighbors(group[i], group[j]);
                    if (hasEdge)
                        sum += power;
                    power *= 2;
                }
            }
        }
    } else {
        // Use combinations
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {

                hasEdge = AreNeighbors(group[i], group[j]);
                if (hasEdge)
                    sum += power;
                power *= 2;
            }
        }

    }
    return sum;
}

GPUMotifCalculator::~GPUMotifCalculator() {
    //map the group num to the iso motif
    delete nodeVariations;
    //the index in which we remove the node from the graph. Basically, from this index on the node doesen't exist.
    delete removalIndex;
    //the nodes, sorted in descending order by the degree.
    delete sortedNodesByDegree;

    // Memory resources

    cudaFree(devicePointerMotifVariations);
    cudaFree(devicePointerRemovalIndex);
    cudaFree(devicePointerSortedNodesByDegree);
    // For the original graph
    cudaFree(deviceOriginalGraphOffsets); // @suppress("Function cannot be resolved")
    cudaFree(deviceOriginalGraphNeighbors); // @suppress("Function cannot be resolved")

    // For the full graph
    cudaFree(deviceFullGraphOffsets); // @suppress("Function cannot be resolved")
    cudaFree(deviceFullGraphNeighbors); // @suppress("Function cannot be resolved")

    // Feature array
    cudaFree(deviceFeatures); // @suppress("Function cannot be resolved")
}
