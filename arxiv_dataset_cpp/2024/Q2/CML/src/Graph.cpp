#include "Graph.h"

// Constructor for graph with adj. mat. given by user
Graph::Graph(size_t nodes,StringVector node_names,
             NumericMatrix adj,bool verbose) : 
  amat(adj),verbose(verbose),m_p(nodes),m_names(node_names) {
  validateAdjMatrix(adj);
  if (nodes < adj.ncol()){
    warning("Number of nodes < rows and columns of the adjacency matrix.");
    warning("Changing the value to match the adj. matrix.");
    m_p = adj.ncol();
  }
};

// Initializes a complete graph without names
Graph::Graph(size_t nodes,bool verbose) : 
  verbose(verbose),m_p(nodes) {
  m_names = StringVector(nodes);
  amat = NumericMatrix(nodes);
  std::fill(amat.begin(), amat.end(), 1);
  amat.fill_diag(0);
}

void Graph::emptyGraph(){
  std::fill(amat.begin(),amat.end(),0);  
}

// Returns a vector of nodes adjacent to node i in the graph
NumericVector Graph::getAdjacent(const size_t &i){
  validateIndex(i);
  NumericVector adj_nodes;
  for (size_t j = 0; j < m_p; ++j){
    if (amat(i,j) != 0 || amat(j,i) != 0){
      adj_nodes.push_back(j);
    }
  }
  return adj_nodes;
}

// Returns a vector of nodes that are not adjacent to i in the graph
NumericVector Graph::getNonAdjacent(const size_t &i){
  validateIndex(i);
  NumericVector non_adj;
  
  for (size_t j=0;j<m_p;++j){
    if (amat(i,j)==0 && amat(j,i)==0 && j!=i){
      non_adj.push_back(j);
    }
  }
  
  return non_adj;
}

void Graph::printAmat(){
  printMatrix(amat);
}

/*
 * Minimum Discriminating Path
 * Purpose: Find a minimum discriminating path that ends with nodes a,b,c
 */

// Obtain values "d" that haven't been visited and d*->a
NumericVector Graph::get_d_vals(size_t a,const std::vector<bool> &visited){
  NumericVector d_vals;
  for (size_t i=0;i<m_p;++i){
    // We need d *-> a
    if (amat(a,i)!=0 && amat(i,a)==2 && !visited[i]){
      d_vals.push_back(i);
    }
  }
  if (verbose) printVecElementsNoNames(d_vals,"\nPotential values: ","\n");
  return d_vals;
}

NumericVector Graph:: get_r_vals(size_t d,const std::vector<bool> &visited){
  NumericVector r_vals;
  for (size_t i=0;i<m_p;++i){
    // We need r *-> d
    if (amat(d,i)!=0 && amat(i,d)==2 && !visited[i]){
      r_vals.push_back(i);  
    }
  }
  if (verbose) printVecElementsNoNames(r_vals,"Potential values for the path: ","\n");
  return r_vals;
}

List createPathList(size_t a,NumericVector set,bool verbose=false){
  if (verbose) Rcout << "Creating path list\n";
  size_t num_paths = set.length();
  NumericVector path = {0};
  path[0] = a;
  
  List paths_to_try(num_paths);
  
  NumericVector new_path;
  
  for (size_t i = 0;i<num_paths;++i){
    new_path = clone(path);
    new_path.push_back(set(i));
    paths_to_try[i] = new_path;
    if (verbose) printVecElementsNoNames(paths_to_try[i],"New Path: ","\n");
  }
  
  return paths_to_try;
}

List updatePathList(NumericVector mpath,NumericVector &set,List &old_paths,
                    bool verbose=false){
  size_t old_size = old_paths.length();
  if (verbose) Rcout << "Size of old path list: " << old_size << std::endl;
  size_t num_paths = set.length();
  List new_paths(old_size+num_paths);
  if (verbose) {
    Rcout << "Size of new path list: ";
    Rcout << new_paths.length() << std::endl;
  }
  
  NumericVector new_path;
  String starter1 = String("Path ");
  String starter2;
  for (size_t i=0;i<old_size+num_paths;++i){
    if (i < old_size){
      new_paths[i] = old_paths[i]; 
    } else {
      new_path = clone(mpath);
      new_path.push_back(set(i-old_size));
      new_paths[i] = new_path;
    }
    starter2 = starter1;
    starter2 += std::to_string(i);
    starter2 += ": ";
    if (verbose) printVecElementsNoNames(new_paths[i],starter2,"\n");
  }
  return new_paths;
}

// Identify the minimum discriminating path from a to c for b
// Return a vector (-1) if no discriminating path is found
NumericVector Graph::minDiscPath(size_t a,size_t b,size_t c){
  std::vector<bool> visited;
  visited.assign(m_p,false);
  visited[a] = true;
  visited[b] = true;
  visited[c] = true;
  // Obtain values that are colliders (or first node) on a path from the nodes to c
  NumericVector d_vals = get_d_vals(a,visited); 
  if (d_vals.length()>0){
    List path_list = createPathList(a,d_vals,verbose);
    size_t counter = 0; // ensures that we go through each vector in the list
    size_t list_length = path_list.length();
    NumericVector mpath; // tracks the potential minimum discriminating path
    size_t m; // tracks the current length of the minimum discriminating path
    size_t d; // tracks the current last value in the path
    size_t pred; // tracks second-last
    while (counter < list_length){
      mpath = path_list[counter];
      if (verbose) printVecElementsNoNames(mpath,"mpath: ","\n");
      m = mpath.length();
      d = mpath(m-1);
      // If node d is not connected to c, then it is the first node on the path
      if (amat(c,d)==0 && amat(d,c)==0){
        NumericVector minDiscPath;
        for (int i=m-1;i>=0;--i){
          minDiscPath.push_back(mpath(i));
        }
        minDiscPath.push_back(b);
        minDiscPath.push_back(c);
        if (verbose) {
          printVecElementsNoNames(minDiscPath,
                                  "Minimum Discriminating Path: ","\n"); 
        }
        return minDiscPath;
      }
      // Otherwise, we must continue to search for a path
      pred = mpath(m-2);
      ++counter;
      // d is connected to c, but it must be an ancestor
      // we must also go back and check to make sure d is a collider on the path
      if (amat(d,c)==2 && amat(c,d)==3 && 
      amat(pred,d)==2){
        visited[d] = true;
        // Find all neighbors of d not visited yet
        NumericVector r_vals = get_r_vals(d,visited);
        if (r_vals.length()>0){
          path_list = updatePathList(mpath,r_vals,path_list,verbose);
        }
      }
      list_length = path_list.length();
    }
  }
  return NumericVector::create(-1);
}

// Determines if we have an uncovered potentially directed path of size 3
// <alpha,beta,gamma>
NumericVector Graph::uncovPdPath(size_t alpha,size_t beta,size_t gamma){
  NumericVector final_path(0);
  bool cond1 = amat(beta,gamma)==1 || 
    amat(beta,gamma)==2;
  bool cond2 = amat(gamma,beta)==1 || 
    amat(gamma,beta)==3;
  bool cond3 = amat(gamma,alpha)==0 && 
    amat(alpha,gamma)==0;
  
  if (cond1 && cond2 && cond3){
    final_path = NumericVector::create(alpha,beta,gamma);
  }
  return final_path;
}

bool Graph::isPathUncovered(NumericVector p){
  size_t n_path = p.length();
  for (size_t i=0;i<n_path-2;++i){
    if ((amat(p(i),p(i+2))!=0) || (amat(p(i+2),p(i))!=0)){
      return false;
    }
  }
  return true;
}

void Graph::checkWronglyCovered(NumericVector p){
  if (!isPathUncovered(p)){
    stop("Error in path construction.");
  }
}

// Check if alpha and beta are properly connected for a potentially directed path
bool Graph::areEdgesPotentiallyDirected(size_t alpha,size_t beta){
  // Check p.d. requirements: cannot be into alpha or out of beta
  bool cond1 = amat(alpha,beta) == 1 || amat(alpha,beta)==2;
  bool cond2 = amat(beta,alpha) == 1 || amat(beta,alpha)==3;
  return cond1 && cond2;
}

// Identify possible values of theta such that <alpha,beta,theta> is 
// uncovered p.d.
NumericVector Graph::idThetaVals(size_t alpha,size_t beta,
                                 const std::vector<bool> &visited){
  NumericVector theta_vals(0);
  bool potentially_directed; bool uncovered;
  for (size_t theta = 0; theta < m_p; ++theta){
    potentially_directed = areEdgesPotentiallyDirected(beta,theta);
    uncovered = amat(theta,alpha) == 0 && 
      amat(alpha,theta)== 0;
    if (potentially_directed && uncovered && !visited[theta]){
      if (verbose) Rcout << "Potential theta: " << theta << std::endl;
      theta_vals.push_back(theta);
    }
  }
  return theta_vals;
}

// Determines if node d is the final node necessary for the 
// uncovered p.d. path
// p = <alpha,beta,...,d_1,d,gamma>
// If so, returns that path. If not, returns empty path.
NumericVector Graph::idUncovPdPath(size_t alpha,size_t beta,size_t gamma,
                                   size_t d,NumericVector mpath){
  // Ensure that the last part of the path is uncovered
  size_t d_1 = mpath(mpath.size()-2);
  bool uncovered = amat(d_1,gamma)==0 && amat(gamma,d_1)==0;
  // Check that d fulfills p.d. requirements
  if (areEdgesPotentiallyDirected(d,gamma) && uncovered){
    if (verbose) {
      Rcout << "Found a final node on the uncovered p.d. path: ";
      Rcout << d << std::endl; 
    }
    // Final path starts with alpha
    NumericVector final_path = NumericVector::create(alpha);
    size_t m = mpath.length();
    // Add mpath after alpha to the final path
    for (size_t i=0;i<m;++i){
      final_path.push_back(mpath(i));  
    }
    // Path ends with gamma
    final_path.push_back(gamma);
    if (verbose) printVecElementsNoNames(final_path,"Path: ","\n");
    
    // Ensure the path is uncovered
    checkWronglyCovered(final_path);
    
    if (verbose) printVecElementsNoNames(final_path,"Final Path: ","\n");
    return final_path;
  } else {
    // we have to keep looking
    if (verbose) printVecElementsNoNames(mpath,
        "Current Path = <","> does not complete uncovered p.d. path.\n");
    return NumericVector(0);
  }
}

// Identify an uncovered potentially directed path <alpha,beta,...,gamma>
NumericVector Graph::minUncovPdPath(size_t alpha,size_t beta,size_t gamma){
  NumericVector final_path(0);
  // Check if conditions are met for alpha and beta
  if (!areEdgesPotentiallyDirected(alpha,beta)){
    return final_path; // This cannot be an uncovered p.d. path
  }
  
  // If <alpha,beta,gamma> is uncovered p.d., return this path
  final_path = uncovPdPath(alpha,beta,gamma);
  if (final_path.length()>0){
    if (verbose){
      Rcout << "Inputted values already form an uncovered p.d. path\n";
    }
    return final_path;
  }
  
  bool potentially_directed; bool uncovered;
  NumericVector mpath; // current path under consideration
  size_t m; // length of the current path
  List path_list; // running list of possible paths
  size_t counter; // tracks which path in the list we are considering
  size_t d; size_t d_1; // track last two nodes of path
  
  // Check for paths of 4 or more (what we are interested in for rule 9)
  if (verbose) Rcout << "Checking for paths of 4 or more\n";
  // we haven't visited any of the other nodes besides alpha,beta,gamma
  std::vector<bool> visited;
  visited.assign(m_p,false); 
  visited[alpha]=true; 
  visited[beta]=true;
  visited[gamma]=true;
  // Grab possible values in the path following beta (denoted theta)
  NumericVector theta_vals = idThetaVals(alpha,beta,visited);
  if (theta_vals.length()>0){
    // Create list of paths starting with beta and followed by each potential theta
    path_list = createPathList(beta,theta_vals,verbose);
    counter = 0;
    // Iterate through each path
    while (counter<path_list.length()){
      // Current path we are considering
      mpath = path_list[counter];
      if (verbose) printVecElementsNoNames(mpath,"mpath: ","\n");
      m = mpath.length();
      d = mpath(m-1);
      visited[d]=true;
      // Check that d fulfills p.d. requirements
      final_path = idUncovPdPath(alpha,beta,gamma,d,mpath);
      if (final_path.length()>0){
        //Rcout << "Final path first\n";
        return final_path;
      } else {
        // d and gamma are not connected or connected with a "wrong" edge
        // iteratively search for neighbors of d not yet visited
        // and add them to potential members of the path
        NumericVector r_vals(0); // track possible nodes to add to the path 
        d_1 = mpath(m-2);
        for (size_t i=0;i<m_p;++i){
          // Check p.d. requirements
          potentially_directed = areEdgesPotentiallyDirected(d,i);
          // Check uncovered requirement
          uncovered = amat(i,d_1) == 0 && amat(d_1,i) == 0;
          if (potentially_directed && uncovered && !visited[i]){
            if (verbose) Rcout << "Potential New Value in Path: " << i << std::endl;
            r_vals.push_back(i);  
          }
        }
        if (r_vals.length()>0){
          path_list = updatePathList(mpath,r_vals,path_list,verbose);
        }
      }
      ++counter;
      if (verbose) {
        Rcout << "Counter: " << counter;
        Rcout << " | Number of paths: " << path_list.length() << std::endl;
      }
    } // End while
  }
  
  return NumericVector(0);  
  
}
