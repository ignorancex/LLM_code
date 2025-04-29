#include "DAG.h"
#include "PDAG.h"

using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix check_amat_works(int nodes,StringVector node_names,NumericMatrix adj){
  Graph g(nodes,node_names,adj);
  return g.getAmat();
}

// [[Rcpp::export]]
NumericMatrix check_amat_works_onepar(int nodes){
  Graph g(nodes);
  return g.getAmat(); // returns the adjacency matrix for a complete graph
}

// [[Rcpp::export]]
StringVector check_names_works(int nodes,StringVector node_names,NumericMatrix adj){
  Graph g(nodes,node_names,adj);
  return g.getNodeNames();
}

// [[Rcpp::export]]
List check_dag_object(int nodes,StringVector node_names,
                      NumericMatrix adj,bool v = false){
  DAG g(nodes,node_names,adj);
  int i = 3;
  Rcout << "Adjacency Matrix:\n";
  g.printAmat();
  Rcout << "The neighbors of node " << i << " are " << g.getNeighbors(i,v) << std::endl;
  NumericVector t {0,2};
  Rcout << "The neighbors of nodes " << t << " are " << g.getNeighborsMultiTargets(t,v) << std::endl;
  return List::create(
    _["OneNeighbor"]=g.getNeighbors(i,v),
    _["TwoNeighbors"]=g.getNeighborsMultiTargets(t,v)
  );
}

// [[Rcpp::export]]
List check_pdag_object(int nodes,StringVector node_names,
                       NumericMatrix adj,bool v = false){
  PDAG g(nodes,node_names,adj);
  int i = 3;
  Rcout << "Adjacency Matrix:\n";
  g.printAmat();
  Rcout << "The neighbors of node " << i << " are " << g.getNeighbors(i,v) << std::endl;
  NumericVector t {0,2,7};
  Rcout << "The neighbors of nodes " << t << " are " << g.getNeighborsMultiTargets(t,v) << std::endl;
  return List::create(
    _["OneNeighbor"]=g.getNeighbors(i,v),
    _["TwoNeighbors"]=g.getNeighborsMultiTargets(t,v)
  );
}

// [[Rcpp::export]]
void check_dag_object2(int nodes){
  DAG g(nodes);
  g.printAmat();
}

// [[Rcpp::export]]
void check_pdag_object2(int nodes){
  PDAG g(nodes);
  g.printAmat();
}

// [[Rcpp::export]]
NumericVector check_neighbors_retrieval(int nodes,StringVector node_names,
                                        NumericMatrix adj,int t,bool v = false){
  DAG g(nodes,node_names,adj);
  return g.getNeighbors(t,v);
}

// [[Rcpp::export]]
NumericVector check_neighbors_retrieval_multi(int nodes,StringVector node_names,
                                              NumericMatrix adj,
                                              NumericVector t,bool v = false){
  DAG g(nodes,node_names,adj);
  return g.getNeighborsMultiTargets(t,v);
}

// [[Rcpp::export]]
NumericVector check_pdag_neighbors_retrieval(int nodes,StringVector node_names,
                                             NumericMatrix adj,int t,bool v = false){
  PDAG g(nodes,node_names,adj);
  return g.getNeighbors(t,v);
}

// [[Rcpp::export]]
int check_amat_retrieval(int nodes,StringVector node_names,NumericMatrix adj,int i,int j){
  DAG g(nodes,node_names,adj);
  return g(i,j);
}

// [[Rcpp::export]]
int check_amat_retrieval_function(int nodes,StringVector node_names,NumericMatrix adj,int i,int j){
  DAG g(nodes,node_names,adj);
  return g.getAmatVal(i,j);
}

// [[Rcpp::export]]
NumericVector check_amat_row_retrieval(int nodes,StringVector node_names,NumericMatrix adj,int i){
  DAG g(nodes,node_names,adj);
  return g.getAmatRow(i);
}

// [[Rcpp::export]]
NumericVector check_amat_col_retrieval(int nodes,StringVector node_names,NumericMatrix adj,int j){
  DAG g(nodes,node_names,adj);
  return g.getAmatCol(j);
}

// [[Rcpp::export]]
List check_adjacent_non_adjacent(int nodes,StringVector node_names,NumericMatrix adj,int i){
  DAG g(nodes,node_names,adj);
  return List::create(
    _["adj"]=g.getAdjacent(i),
    _["nonadj"]=g.getNonAdjacent(i)
  );
}

// [[Rcpp::export]]
NumericVector check_non_adjacent_solo(int nodes,StringVector node_names,NumericMatrix adj,int i){
  Graph g(nodes,node_names,adj);
  return g.getNonAdjacent(i);
}

// [[Rcpp::export]]
List check_directed_undirected(int nodes,StringVector node_names,
                                        NumericMatrix adj,int i,int j){
  Graph g(nodes,node_names,adj);
  return List::create(
    _["directed"]=g.isDirected(i,j),
    _["undirected"]=g.isUndirected(i,j)
  );
}

// [[Rcpp::export]]
List check_sizes(int nodes,StringVector node_names,
                               NumericMatrix adj){
  Graph g(nodes,node_names,adj);
  return List::create(
    _["ncol"]=g.getNCol(),
    _["nrow"]=g.getNRow()
  );
}

// [[Rcpp::export]]
int check_amat_setval(int nodes,StringVector node_names,NumericMatrix adj,int i,int j,int val){
  Graph g(nodes,node_names,adj);
  g(i,j)=val;
  return g.getAmatVal(i,j);
}

// [[Rcpp::export]]
int check_amat_setval_function(int nodes,StringVector node_names,NumericMatrix adj,int i,int j,int val){
  Graph g(nodes,node_names,adj);
  g.setAmatVal(i,j,val);
  return g.getAmatVal(i,j);
}

// [[Rcpp::export]]
bool checkIfAdjacent(int nodes,StringVector node_names,NumericMatrix adj,int i,int j){
  Graph g(nodes,node_names,adj);
  return g.areAdjacent(i,j);
}

// [[Rcpp::export]]
NumericMatrix checkEmptyGraph(int p){
  Graph g(p);
  g.printAmat();
  g.emptyGraph();
  return g.getAmat();
}

// [[Rcpp::export]]
bool checkAcyclicity(int nodes,StringVector node_names,NumericMatrix adj){
  DAG g(nodes,node_names,adj);
  return g.isAcyclic();
}

// [[Rcpp::export]]
bool checkIsAncestor(int nodes,StringVector node_names,NumericMatrix adj,
                     int desc,int anc,bool verbose=false){
  DAG g(nodes,node_names,adj,verbose);
  return g.isAncestor(desc,anc);
}

// [[Rcpp::export]]
bool checkInNeighborhood(int nodes,StringVector node_names,NumericMatrix adj,
                         int i,int j,bool verbose=false){
  DAG g(nodes,node_names,adj,verbose);
  return g.inNeighborhood(i,j);
}

// [[Rcpp::export]]
bool check_pdag_inNeighborhood(int nodes,StringVector node_names,NumericMatrix adj,
                               int i,int j,bool verbose=false){
  PDAG g(nodes,node_names,adj,verbose);
  return g.inNeighborhood(i,j);
}

// [[Rcpp::export]]
NumericMatrix check_set_amat(int nodes,StringVector node_names,NumericMatrix adj){
  Graph g(nodes);
  g.setAmat(adj);
  return g.getAmat();
}

// [[Rcpp::export]]
NumericVector check_disc_path(int nodes,StringVector node_names,NumericMatrix adj,
                              size_t c,size_t d,size_t e){
  Graph g(nodes,node_names,adj);
  g.setVerboseTrue();
  return g.minDiscPath(c,d,e);
}

// [[Rcpp::export]]
NumericVector check_upd_path(int nodes,StringVector node_names,NumericMatrix adj,
                             size_t a,size_t b,size_t e){
  Graph g(nodes,node_names,adj);
  g.setVerboseTrue();
  return g.minUncovPdPath(a,b,e);
}

// [[Rcpp::export]]
void test_checkWronglyCovered(int nodes,StringVector node_names,
                                       NumericMatrix adj,NumericVector p){
  Graph g(nodes,node_names,adj);
  g.checkWronglyCovered(p);
}

// [[Rcpp::export]]
List checkNeighborhoodId(int nodes,StringVector node_names,NumericMatrix adj,
                         int i,bool verbose=false){
  DAG g(nodes,node_names,adj,verbose);
  return List::create(
    _["parents"]=g.getParents(i),
    _["children"]=g.getChildren(i),
    _["neighborhood"]=g.getNeighbors(i)
  );
}


