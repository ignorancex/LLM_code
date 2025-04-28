/*
 * CacheGraph.cpp
 * 
 * A container for the cache aware representation of a graph.
 *
 *  Created on: Oct 28, 2018
 *
 */

#include "../includes/CacheGraph.h"

void CacheGraph::Clear() {
	m_NumberOfNodes = 0;
	m_NumberOfEdges = 0;
	delete m_Graph;
	m_Graph = NULL;
	delete m_Offsets;
	m_Offsets = NULL;
}

/*
 Initialize the graph from two lists (the offset list and the adjacency list).
 */
void CacheGraph::Assign(const std::vector<int64>& NodeOffsets,
		const std::vector<unsigned int>& Neighbours) {
	//Clear the graph so we can assign new data
	Clear();

	//copy the offset list into a new array
	m_NumberOfNodes = static_cast<unsigned int>(NodeOffsets.size() - 1);
	m_Offsets = new int64[m_NumberOfNodes + 1];
	std::memcpy(m_Offsets, &NodeOffsets[0], NodeOffsets.size() * sizeof(int64));

	m_NumberOfEdges = m_Offsets[m_NumberOfNodes];
	//copy the adjacency list into a new array
	m_Graph = new unsigned int[m_NumberOfEdges];
	std::memcpy(m_Graph, &Neighbours[0],
			Neighbours.size() * sizeof(unsigned int));

}

void CacheGraph::Assign(const std::vector<int64>& NodeOffsets,
		const std::vector<unsigned int>& Neighbours,
		const std::vector<double>& weights) {
	this->Assign(NodeOffsets, Neighbours);
	m_Weights = new double[m_NumberOfEdges];
	std::memcpy(m_Weights, &weights[0], weights.size() * sizeof(double));
	weighted = true;
}

/*
 Save the graph to a binary file.
 */
bool CacheGraph::SaveToFile(const std::string& FileName) const {

	//open or create a file of a binary format
	FILE* hFile = std::fopen(FileName.c_str(), "w+b");

	//write the class variables to the file
	std::fwrite(&m_NumberOfNodes, sizeof(unsigned int), 1, hFile);
	std::fwrite(&m_NumberOfEdges, sizeof(int64), 1, hFile);
	std::fwrite(m_Offsets, sizeof(int64), m_NumberOfNodes + 1, hFile);
	std::fwrite(m_Graph, sizeof(unsigned int), m_NumberOfEdges, hFile);
	std::fwrite(&weighted, sizeof(bool), 1, hFile);
	if (weighted)
		std::fwrite(m_Weights, sizeof(double), m_NumberOfEdges, hFile);
	std::fwrite(&directed, sizeof(bool), 1, hFile);

	//close the file
	std::fclose(hFile);
	hFile = NULL;

	return true;

}

/*
 Read the data from the binary file
 */
bool CacheGraph::LoadFromFile(const std::string& FileName) {
	Clear();
	FILE* hFile;
	//open network file
	try {
		hFile = std::fopen(FileName.c_str(), "rb");
	} catch (std::exception& e) {
		return false;
	}

	//read the number of nodes
	std::fread(&m_NumberOfNodes, sizeof(unsigned int), 1, hFile);

	//read the number of edges
	int64 NumberOfEdges = 0;
	std::fread(&NumberOfEdges, sizeof(int64), 1, hFile);
	m_NumberOfEdges = NumberOfEdges;
	//create an array to store the indices (offsets) of the nodes in the graph array and read into it
	m_Offsets = new int64[m_NumberOfNodes + 1];
	std::fread(m_Offsets, sizeof(int64), m_NumberOfNodes + 1, hFile);

	//create the main array containing the lists of neighbors.
	/*
	 NOTE:
	 Here there is an assumption of a directed graph.
	 An undirected graph is created by having two edges saved in the file for each edge
	 in the network.
	 */
	m_Graph = new unsigned int[NumberOfEdges];
	std::fread(m_Graph, sizeof(unsigned int), NumberOfEdges, hFile);
	std::fread(&weighted, sizeof(bool), 1, hFile);
	if (weighted)
		std::fread(m_Weights, sizeof(double), m_NumberOfEdges, hFile);
	std::fread(&directed, sizeof(bool), 1, hFile);

	std::fclose(hFile);
	hFile = NULL;
	return true;

}

/*
 Utility function: create the full path from the directory and the file name and then call the
 overloaded function.
 */
bool CacheGraph::LoadFromFile(const std::string& DirectroyName,
		const std::string& BaseFileName) {
	std::string FileName = GetFileNameFromFolder(DirectroyName, BaseFileName);
	return LoadFromFile(FileName);
}

/*
 Utility function: append the directory and file names into one string
 */
std::string CacheGraph::GetFileNameFromFolder(const std::string& DirectroyName,
		const std::string& BaseFileName) {
	std::stringstream FileName;
	FileName << DirectroyName << BaseFileName << "_" << std::setfill('0')
			<< std::setw(2) << ".bin";
	return FileName.str();
}

/*
 Assuming our graph is directed, generate the inverted graph.
 i.e. change every edge e=(a,b) to (b,a) in the inverted graph.
 */
void CacheGraph::InverseGraph(CacheGraph& InvertedGraph) const {
	//get the number of edges in the graph
	const int64 NumberOfEdges = m_Offsets[m_NumberOfNodes];
	//clear the inverted graph
	InvertedGraph.Clear();
	//assign the member variables to the inverted graph
	InvertedGraph.m_NumberOfNodes = m_NumberOfNodes;
	//allocate the needed memory
	InvertedGraph.m_Offsets = new int64[m_NumberOfNodes + 1];
	InvertedGraph.m_Graph = new unsigned int[NumberOfEdges];
//	InvertedGraph.m_Weights = new double[NumberOfEdges]; 

	//invert the adjancency list
	unsigned int* InDegrees = new unsigned int[m_NumberOfNodes];
	std::memset(InDegrees, 0, m_NumberOfNodes * sizeof(unsigned int));
	for (const unsigned int* p = m_Graph; p < m_Graph + NumberOfEdges; ++p) {
		++InDegrees[*p];
	}
	InvertedGraph.m_Offsets[0] = 0;
	std::partial_sum(InDegrees, InDegrees + m_NumberOfNodes,
			InvertedGraph.m_Offsets + 1);
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
		for (int peerIndex = m_Offsets[NodeID];
				peerIndex < m_Offsets[NodeID + 1]; peerIndex++) {
			unsigned int peer = m_Graph[peerIndex];
			InvertedGraph.m_Graph[InvertedGraph.m_Offsets[peer]] = NodeID;
//			InvertedGraph.m_Weights[InvertedGraph.m_Offsets[peer]] = m_Weights[peerIndex];  
			++InvertedGraph.m_Offsets[peer];
		}
	}
	InvertedGraph.m_Offsets[0] = 0;
	std::partial_sum(InDegrees, InDegrees + m_NumberOfNodes,
			InvertedGraph.m_Offsets + 1);
	//clear memory
	delete[] InDegrees;
	InDegrees = NULL;

}

/*
 * Create the undirected version of the graph represented in this instance.
 * Essentially, we're combining the directed graph and it's inverse.
 */
void CacheGraph::CureateUndirectedGraph(const CacheGraph& InvertedGraph,
		CacheGraph& UndirectedGraph) const {
	//calculate the number of edges in the graph
	int64 NumberOfEdges = m_Offsets[m_NumberOfNodes];
	//clear the new graph
	UndirectedGraph.Clear();
	//assign member variables
	UndirectedGraph.m_NumberOfNodes = m_NumberOfNodes;
	//allocate memory
	UndirectedGraph.m_Offsets = new int64[m_NumberOfNodes + 1];
	std::memset(UndirectedGraph.m_Offsets, 0,
			(m_NumberOfNodes) * sizeof(int64));
	//note that in an undirected graph, there are twice as many edges in the
	//adjacency list as in a directed graph.
	unsigned int* temp_Graph = new unsigned int[2 * NumberOfEdges];
//	double* temp_weights = new double[2 * NumberOfEdges];  
	std::memset(temp_Graph, 0, 2 * NumberOfEdges * sizeof(unsigned int));
	unsigned int *temp_graph_pointer = temp_Graph;
//	double* temp_weight_pointer = temp_weights;     

	//combine the inverted and normal graphs into an undirected one.
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
		auto p1 = m_Graph + m_Offsets[NodeID]; //current neighbor
		auto p2 = InvertedGraph.m_Graph + InvertedGraph.m_Offsets[NodeID]; //inverted current neighbor

//		auto w1 = m_Weights + m_Offsets[NodeID]; //current weight  
//		auto w2 = InvertedGraph.m_Weights + InvertedGraph.m_Offsets[NodeID]; //inverted current weight 

		while (p1 < m_Graph + m_Offsets[NodeID + 1]
				&& p2
						< InvertedGraph.m_Graph
								+ InvertedGraph.m_Offsets[NodeID + 1]) { //while we are in both neighbor lists
			if (*p1 == *p2) { //bi-directional edge
				*temp_graph_pointer = *p1;
//				*temp_weight_pointer = *w1;  
				++p1;
				++p2;
//				++w1;                        
//				++w2;                        
			} else if (*p1 < *p2) {
				*temp_graph_pointer = *p1;
//				*temp_weight_pointer = *w1;  
				++p1;
//				++w1;                        
			} else // if (*p1>*p2)
			{
				*temp_graph_pointer = *p2;
//				*temp_weight_pointer = *w2;  
				++p2;
//				++w2;                        
			}
			++temp_graph_pointer;
		} //END WHILE
		if (p1 < m_Graph + m_Offsets[NodeID + 1]) { //if we haven't read all of the neighbors of the node
			int64 RemainingElements = m_Graph + m_Offsets[NodeID + 1] - p1;
			std::memcpy(temp_graph_pointer, p1,
					sizeof(unsigned int) * RemainingElements);
//			std::memcpy(temp_weight_pointer, p1,
//					sizeof(double) * RemainingElements);     
			temp_graph_pointer += RemainingElements;
//			temp_weight_pointer += RemainingElements;        
		} else if (p2
				< InvertedGraph.m_Graph + InvertedGraph.m_Offsets[NodeID + 1]) { //or if we haven't read all the neighbors in the inverted graph
			int64 RemainingElements = InvertedGraph.m_Graph
					+ InvertedGraph.m_Offsets[NodeID + 1] - p2;
			std::memcpy(temp_graph_pointer, p2,
					sizeof(unsigned int) * RemainingElements);
//			std::memcpy(temp_weight_pointer, p2,
//					sizeof(double) * RemainingElements);      
			temp_graph_pointer += RemainingElements;
//			temp_weight_pointer += RemainingElements;   

		}
		//Otherwise - we've read all neighbors

		UndirectedGraph.m_Offsets[NodeID + 1] = temp_graph_pointer - temp_Graph;
	}

	UndirectedGraph.m_NumberOfEdges =
			UndirectedGraph.m_Offsets[m_NumberOfNodes];

	// Copy neighbor list from temp
	UndirectedGraph.m_Graph = new unsigned int[UndirectedGraph.m_NumberOfEdges];
	std::memcpy(UndirectedGraph.m_Graph, temp_Graph,
			UndirectedGraph.m_NumberOfEdges * sizeof(unsigned int));

	//Copy weights from temp
	UndirectedGraph.m_Weights = new double[UndirectedGraph.m_NumberOfEdges];
//	std::memcpy(UndirectedGraph.m_Weights,temp_weights,UndirectedGraph.m_NumberOfEdges*sizeof(double)); 
	//clear memory
	delete[] temp_Graph;
}

std::vector<unsigned int> CacheGraph::ComputeNodeDegrees() const {
	std::vector<unsigned int> Degrees(m_NumberOfNodes, 0);
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
		Degrees[NodeID] = static_cast<unsigned int>(m_Offsets[NodeID + 1]
				- m_Offsets[NodeID]);
	}
	return Degrees;
}

std::vector<float> CacheGraph::ComputeNodePageRank(float dumping,
    unsigned int NumberOfIterations) const {
        // This code base on https://www.geeksforgeeks.org/page-rank-algorithm-implementation/ code in python.
		// It can be optimized to a better speed, and it should be considered.
		float alpha = dumping;

		std::vector<float> x(m_NumberOfNodes, 0);
		std::vector<float> xlast(m_NumberOfNodes);

		for (int iteration = 0; iteration < NumberOfIterations; ++iteration) {
			// xlast = x.copy()
			// x = 0
			double sum = 0;	// estimate the change in the value for early stopping
			for (int i = 0; i < m_NumberOfNodes; i++) {
				sum += std::abs(xlast[i] - x[i]);
				xlast[i] = x[i];
				x[i] = 0;
			}

			// Early stopping - l1 norm
			if (iteration && (sum / m_NumberOfNodes <= 0.000001)) {
				return xlast;
			}

			// danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
			float danglesum = 0;
			for (int i = 0; i < m_NumberOfNodes; ++i) {
				if (m_Offsets[i] == m_Offsets[i+1])
					danglesum += alpha * xlast[i];
			}

			for (int i = 0; i < m_NumberOfNodes; ++i) {
				// runnig over all the nodes in the graph
				for (auto nbr = m_Graph + m_Offsets[i]; nbr < m_Graph + m_Offsets[i+1]; ++nbr) {
					// running over all the nodes which the vertex has edge to. *p is the vertex numebr.
					x[*nbr] += alpha * xlast[i] / (m_Offsets[i+1] - m_Offsets[i]);	// x[nbr] += alpha * xlast[n] * (W[n][nbr][weight])
				}
				x[i] += danglesum / m_NumberOfNodes + (1 - alpha) / m_NumberOfNodes; // x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]
			}
		}
		return x;
}

std::vector<unsigned short> CacheGraph::ComputeKCore() const {
	// Note: This code assumes that the lol graph is undirected.
	// Any other input may cause a wrong output.

	unsigned int nodes_finished = 0;	// number of nodes we already set their k_core
	const unsigned short UNSET_K_CORE = static_cast<unsigned short>(-1);	// const that sign that this nodes hasn't k_core yet

	std::vector<unsigned short> KShell(m_NumberOfNodes, UNSET_K_CORE);
	std::vector<unsigned int> Degrees(m_NumberOfNodes, 0);

	// Initializes the degrees vector:
	// Sets the degrees for undirected graph or the out-degrees for directed graph.
	for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID)
		Degrees[NodeID] = static_cast<unsigned int>(m_Offsets[NodeID + 1] - m_Offsets[NodeID]);

	unsigned short CurrentShell = 0;
	bool any_degree_changed = false;

	while (nodes_finished < m_NumberOfNodes) {
		do {
			any_degree_changed = false;
			for (unsigned int NodeID = 0; NodeID < m_NumberOfNodes; ++NodeID) {
				// running all over the nodes
				if (KShell[NodeID] == UNSET_K_CORE
						&& Degrees[NodeID] <= CurrentShell) {
					// if we didn't set k_shell for this node, and the degree of the node is less or equal to the current degree:

					KShell[NodeID] = CurrentShell;	// sets the k_shell
					nodes_finished++;

					// remove the node from the graph - reduce the degree of the neighbors.
					for (auto p = m_Graph + m_Offsets[NodeID];
							p < m_Graph + m_Offsets[NodeID + 1]; ++p) {
						if (KShell[*p] == UNSET_K_CORE) {
							--Degrees[*p];
							any_degree_changed = true;
						}
					}
				}
			}
		} while (any_degree_changed);
		++CurrentShell;
	}

	return KShell;
}

/*
 Check wether q is a neighbor of p (i.e. if there exists an edge p -> q)
 For an undirected graph, the order does not matter.
 Input: the two nodes to check
 Output: whether there is an edge p->q
 Note: we are working under the assumption that the list of p's neighbors is ordered,
 and so we use binary search.
 Usage of the binary search keeps the proccess to O(log(V))
 */
bool CacheGraph::areNeighbors(const unsigned int p,
		const unsigned int q) const {
	int first = m_Offsets[p],  //first array element
			last = m_Offsets[p + 1] - 1,     //last array element
			middle;                       //mid point of search
	while (first <= last) {
		middle = (int) (first + last) / 2; //this finds the mid point
		if (m_Graph[middle] == q) {
			return true;
		} else if (m_Graph[middle] < q)
				{
			first = middle + 1;     //if it's in the upper half

		} else { //m_Graph[middle] < q
			last = middle - 1; // if it's in the lower half
		}
	}
	return false;  // not found

}

std::vector<unsigned int>* CacheGraph::SortedNodesByDegree() const {
	std::vector<unsigned int>* sortedNodes = new std::vector<unsigned int>();
	sortedNodes->reserve(m_NumberOfNodes);

	std::vector<unsigned int> nodeDegrees = ComputeNodeDegrees();
	std::vector<NodeWithDegree> nodesWithDegrees;
	nodesWithDegrees.reserve(m_NumberOfNodes);
	for (unsigned int node = 0; node < m_NumberOfNodes; node++)
		nodesWithDegrees.push_back( { node, nodeDegrees[node] });

	std::sort(nodesWithDegrees.begin(), nodesWithDegrees.end(),
			cmpNodesByDegree);

	for (NodeWithDegree nd : nodesWithDegrees)
		sortedNodes->push_back(nd.node);
	return sortedNodes;
}
