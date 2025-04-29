#include <chrono>


#include "DecisionDiagram.h"



template<typename T>
std::ostream &operator<<(std::ostream &s, const std::set<T> &set) {
    s << "{";
    std::string sep;
    for(T el : set){
        s << sep << el;
        sep = ", ";
    }
    return s << "}";
}

template<typename T>
std::ostream &operator<<(std::ostream &s, const std::vector<T> &vec) {
    s << "[";
    std::string sep;
    for(T el : vec){
        s << sep << el;
        sep = ", ";
    }
    return s << "]";
}


Node::Node() = default;

Node::Node(int _layer, NodeIndex _index, std::set<unsigned int> _state_info)
        : layer(_layer), index(_index), state_info(std::move(_state_info)) {}

bool Node::equivalent(const Node &other) const {
    return this->state_info == other.state_info;
}

unsigned int num_vars(const DecisionDiagram &dd) {
    return dd.size() - 1;
}

unsigned int num_nodes(const DecisionDiagram &dd) {
    return std::accumulate(dd.begin(), dd.end(), 0,
                           [](size_t sum, const std::vector<Node> &level) { return sum + level.size(); });
}

unsigned int num_arcs(const DecisionDiagram &dd) {
    unsigned int num_arcs = 0;
    for(const auto &layer : dd){
        for(const auto &node : layer){
            if(node.one_arc != -1){
                num_arcs++;
            }
            if(node.zero_arc != -1){
                num_arcs++;
            }
        }
    }
    return num_arcs;
}

unsigned int get_width(const DecisionDiagram &dd) {
    return std::max_element(dd.begin(), dd.end(),
                            [](const std::vector<Node> &layer1, const std::vector<Node> &layer2) {
                                return layer1.size() < layer2.size();
                            })->size();
}

void print_decision_diagram(const DecisionDiagram &dd, bool use_tag) {
    for(const auto &layer : dd){
        std::cout << (use_tag ? "[DD] " : "") << "Layer: " << layer.front().layer << " of size: " << layer.size()
                  << std::endl;
        for(const auto &node : layer){
            std::cout << (use_tag ? "[DD] " : "") << "Node " << node.index << " state " << node.state_info
                      << " with 1-arc " << node.one_arc << " and 0-arc " << node.zero_arc << std::endl;
        }
    }
}

DecisionDiagram initial_decision_diagram(const Graph &g) {
    DecisionDiagram dd;
    dd.resize(g.ncount() + 1);

    StateInfo new_state;
    for(unsigned int i = 1; i <= g.ncount(); ++i){
        new_state.insert(new_state.end(), i);
    }

    dd[0].emplace_back(1, 0, new_state);
    for(unsigned int j = 0; j < g.ncount(); ++j){
        new_state.erase(j + 1);
        dd[j + 1].emplace_back(j + 2, 0, new_state);
        dd[j][0].zero_arc = 0;
        dd[j][0].one_arc = 0;
    }
    return dd;
}

DecisionDiagram exact_decision_diagram(const Graph &g, int size_limit) {
    NeighborList neighbors = g.get_neighbor_list();
    return exact_decision_diagram(g, neighbors, size_limit);
}

DecisionDiagram exact_decision_diagram(const Graph &g, const NeighborList &neighbors, int size_limit) {
    DecisionDiagram dd;
    dd.resize(g.ncount() + 1);
    int num_nodes = 0;
    int num_arcs = 0;

    StateInfo initial_state;


    // add all vertices to root state set S(r)
    for(unsigned int i = 1; i <= g.ncount(); ++i){
        initial_state.insert(initial_state.end(), i);
    }

    dd[0].emplace_back(1, 0, initial_state);
    num_nodes++;

    //maps state sizes to node indices with that state size
    std::unordered_map< unsigned long , std::vector<int> > map_state_size_to_node_indices;

    for(unsigned int layer = 0; layer < g.ncount(); ++layer) {
      dd[layer + 1].reserve(dd[layer].size() + 5);//some estimation for the size of the next layer, reserve some space for the vector
      map_state_size_to_node_indices.reserve(dd[layer].size() + 5);
      for(Node &u : dd[layer]) {
        if(num_nodes > size_limit*std::pow(10, 6)){
          throw std::runtime_error("The exact decision diagram contains more than " + std::to_string(size_limit)
                                   + " million nodes and already has width " + std::to_string(get_width(dd))
                                   +" after layer "+std::to_string(layer)+" of "+std::to_string(g.ncount())+".");
        }


        auto new_state = u.state_info;
        new_state.erase(layer + 1);

        { // Create 0-arc
          //check if there even exist nodes with state info of that size to possibly be equivalent to
          if(not map_state_size_to_node_indices[new_state.size()].empty()) {
            for(int index : map_state_size_to_node_indices[new_state.size()]) {
              //check if nodes' states are equivalent
              if(new_state == dd[layer + 1][index].state_info) {
                //redirect arc
                u.zero_arc = index;
                break;
              }
            }
          }
          //found no equivalent node so construct one
          if(u.zero_arc == -1){
            dd[layer + 1].emplace_back(layer + 2, int(dd[layer + 1].size()), new_state);
            num_nodes++;
            u.zero_arc = int(dd[layer + 1].size() - 1);
            map_state_size_to_node_indices[new_state.size()].push_back(u.zero_arc); //new node, store index under size category
          }
          num_arcs++;
        }

        //If the next vertex/layer  is in state_info, we add a 1 arc:
        if(u.state_info.count(layer + 1)) {
          // Create 1-arc
          //a 1-arc forces forces removal of neighbors
          for(unsigned int neighbor : neighbors[layer]) {
            new_state.erase(neighbor);
          }

          //check if there even exist nodes with state info of that size to possibly be equivalent to
          if(not map_state_size_to_node_indices[new_state.size()].empty()) {
            for(int index : map_state_size_to_node_indices[new_state.size()]) {
              //check if nodes' states are equivalent
              if(new_state == dd[layer + 1][index].state_info){
                //redirect arc
                u.one_arc = index;
                break;
              }
            }
          }
          //found no equivalent node so construct one
          if(u.one_arc == -1){
            dd[layer + 1].emplace_back(layer + 2, int(dd[layer + 1].size()), new_state);
            num_nodes++;
            u.one_arc = int(dd[layer + 1].size() - 1);
            map_state_size_to_node_indices[new_state.size()].push_back(u.one_arc); //new node, store index under size category
          }
          num_arcs++;
        } // end addition of 1-arc

        // At this place, we could delete the state infos of u, because they are no longer needed.
      }

      map_state_size_to_node_indices.clear();
    }

    // Is this a return w/o copy
    return dd;
}

PathLabelConflict::PathLabelConflict(Path _path, Label _label, Conflict _conflict)
        : path(std::move(_path)), label(std::move(_label)), conflict(std::move(_conflict)) {}

int intersection_size(const StateInfo &a, const StateInfo &b) {
    if(a.empty() or b.empty()) return 0;
    return int(std::count_if(a.begin(), a.end(), [&b](const unsigned int &k) { return b.count(k); }));
}

void separate_edge_conflict(DecisionDiagram &dd, const NeighborList &neighbors, const PathLabelConflict &plc,
                            RedirectArcs redirect_arcs) {
    //check assumptions of conflict being minimal. no, assume input comes from algorithm 2 and is correct

    Path path = plc.path;
    const Label &label = plc.label;
    const Conflict &conflict = plc.conflict;

    int j, k;
    std::tie(j, k) = conflict;
    for(int i = 0; i <= (k - 1) - j; ++i){
        int layer = j + i - 1;
        StateInfo new_state = dd[layer][path[i]].state_info;
        new_state.erase(layer + 1);
        if(label[i]){
            for(unsigned int n : neighbors[layer]){
                new_state.erase(n);
            }
        }

        int t = -1;
        auto search_equivalent_node = std::find_if(dd[layer + 1].begin(), dd[layer + 1].end(),
                                                   [&new_state](const Node &v) { return v.state_info == new_state; });
        if(search_equivalent_node == dd[layer + 1].end()){
            Node w(layer + 2, int(dd[layer + 1].size()), new_state);
            if(redirect_arcs == OriginalArcs){
                if(w.state_info.count(layer + 1 + 1)){
                    w.one_arc = dd[layer + 1][path[i + 1]].one_arc;
                }
                w.zero_arc = dd[layer + 1][path[i + 1]].zero_arc;
            } else if(redirect_arcs == MostSimilarNode){
                StateInfo next_state = w.state_info;
                next_state.erase(layer + 1 + 1);
                auto most_similar_node = std::max_element(dd[layer + 2].begin(), dd[layer + 2].end(),
                                                          [&next_state](const Node &u, const Node &v) {
                                                              return intersection_size(next_state, u.state_info) <
                                                                     intersection_size(next_state, v.state_info);
                                                          });
                w.zero_arc = int(most_similar_node - dd[layer + 2].begin());

                if(w.state_info.count(layer + 1 + 1)){
                    for(unsigned int n : neighbors[layer + 1]){
                        next_state.erase(n);
                    }
                    most_similar_node = std::max_element(dd[layer + 2].begin(), dd[layer + 2].end(),
                                                         [&next_state](const Node &u, const Node &v) {
                                                             return intersection_size(next_state, u.state_info) <
                                                                    intersection_size(next_state, v.state_info);
                                                         });
                    w.one_arc = int(most_similar_node - dd[layer + 2].begin());
                }

            }
            dd[layer + 1].push_back(w);
            t = int(dd[layer + 1].size() - 1);

        } else{
            t = int(search_equivalent_node - dd[layer + 1].begin());
        }

        if(label[i]){
            dd[layer][path[i]].one_arc = t;
        } else{
            dd[layer][path[i]].zero_arc = t;
        }
        path[i + 1] = t;
    }
}

std::vector<PathLabelConflict>
detect_edge_conflict(DecisionDiagram dd, const NeighborList &neighbors, double flow_val, Model model,
                     ConflictResolution find_conflicts, PathDecomposition path_decomposition) {
    unsigned int n = num_vars(dd);
    std::vector<PathLabelConflict> conflict_info;
    std::vector<double> conflict_flow;

    while((model == IP and flow_val >= (1 - double_eps)) or (model == LP and flow_val > double_eps)){
        Path path;
        Label label;
        std::vector<unsigned int> selected_vertices;
        bool path_found_conflict = false;
        path.push_back(0); //index of root node
        double path_min_flow = (double) n;

        for(unsigned int i = 0; i < n; i++){
            int u = path.back();
            if((dd[i][u].one_arc != -1) and
               (not(path_decomposition == PreferZeroArcs and (dd[i][u].zero_arc_flow >= dd[i][u].one_arc_flow))) and
               ((model == IP and (int(std::round(dd[i][u].one_arc_flow) >= 1))) or
                (model == LP and dd[i][u].one_arc_flow >= dd[i][u].zero_arc_flow))){
                path_min_flow = std::min(path_min_flow, dd[i][u].one_arc_flow);

                bool use_zero_arc_instead = false;
                if(not path_found_conflict){
                    for(auto j_it = selected_vertices.rbegin();
                        j_it != selected_vertices.rend(); j_it++){
                        unsigned int j = j_it.operator*();
                        if(std::find(neighbors[i].begin(), neighbors[i].end(), j) != neighbors[i].end()){
                            conflict_info.emplace_back(
                                    Path(path.begin() + j - 1, path.end()),
                                    Label(label.begin() + j - 1, label.end()),
                                    std::make_tuple(j, i + 1)
                            );
                            if(find_conflicts == SingleConflict){
                                return conflict_info;
                            }
                            //check if it's possible to take the 0-arc instead to avoid the conflict
                            if(dd[i][u].zero_arc_flow >= double_eps and path_decomposition == AvoidConflicts){
                                path_min_flow = std::min(path_min_flow, dd[i][u].zero_arc_flow);
                                path.push_back(dd[i][u].zero_arc);
                                label.push_back(zeroArc);
                                use_zero_arc_instead = true;
                                break;
                            }
                            //otherwise continue but remember we found a conflict on this path already
                            path_found_conflict = true;
                            break;//don't return here but continue to search for conflicts
                        }
                    }
                }
                if(not use_zero_arc_instead){
                    path.push_back(dd[i][u].one_arc);
                    label.push_back(oneArc);
                    selected_vertices.push_back(i + 1);
                }
            } else{
                path_min_flow = std::min(path_min_flow, dd[i][u].zero_arc_flow);
                path.push_back(dd[i][u].zero_arc);
                label.push_back(zeroArc);
            }
        }

        if(path_found_conflict and (find_conflicts == LargestFlowConflict)){
            conflict_flow.push_back(path_min_flow);
        }

        if(path_min_flow > 0){
            for(unsigned int i = 0; i < n; i++){
                if(label[i]){
                    dd[i][path[i]].one_arc_flow -= path_min_flow;
                } else{
                    dd[i][path[i]].zero_arc_flow -= path_min_flow;
                }
            }
            flow_val -= path_min_flow;
        } else{
            std::cout << std::setprecision(25) << "Flow value left was " << flow_val << std::endl;
            //the flow value got too small and floating errors likely occurred, return those conflicts that were found
            return conflict_info;
        }
    }
    if(conflict_info.empty()){
        return conflict_info;
    }

    if(model == IP){
        //do not use largest flow conflict in case of IP, flow is integral anyway and therefore >= 1, always large!
        return conflict_info;
    }

    if(find_conflicts == LargestFlowConflict){
        double conflict_minimum_flow = std::pow(10, -4);
        std::vector<PathLabelConflict> result;
        for(unsigned int i = 0; i < conflict_flow.size(); i++){
            if(conflict_flow[i] >= conflict_minimum_flow){
                result.push_back(conflict_info[i]);
            }
        }
        return result;
    }
    return conflict_info;
}


double compute_flow_solution(DecisionDiagram &dd, Model model, int coloring_upper_bound, Formulation formulation, int num_cores, int mip_emphasis) {

    COLORlp *flow_lp;//environment is initialised in DDColors
    if (COLORlp_init(&flow_lp, "flow_lp")) {
      throw std::runtime_error("COLORlp_init failed.");
    }
    if (COLORlp_objective_sense(flow_lp, COLORlp_MIN)) { //minimizing should be the default setting anyway
      throw std::runtime_error("COLORlp_objective_sense.");
    }

    if (COLORlp_set_emphasis(flow_lp, mip_emphasis)) {
      throw std::runtime_error("COLORlp_set_emphasis.");
    }
    // std::cout << "CPX_PARAM_MIPEMPHASIS set to " << mip_emphasis << std::endl;

    if (COLORlp_set_threads(flow_lp, num_cores)) {
      throw std::runtime_error("COLORlp_num_cores.");
    }
    // std::cout << "CPX_PARAM_THREADS set to " << num_cores << std::endl;


    unsigned int n = num_vars(dd);
    int var_bound = (coloring_upper_bound == -1) ? int(n) : coloring_upper_bound - 1;
    double used_var_bound = (formulation == VarColorBound) ? var_bound : CPX_INFBOUND; //have flow at most chi(G)-1 on edges



    //add variables for all edges in decision diagram but only set obj to 1 for outgoing edges of root
    EdgeIndex edge_index = 0;
    auto var_type = (model == IP) ? COLORlp_INTEGER : COLORlp_CONTINUOUS;
    for(unsigned int layer = 0; layer < n; layer++){
        for(Node &u : dd[layer]){
            int obj = layer ? 0 : 1;
            //First add one_arc variables and then zero_arc variables. keep this ordering in mind!
            if(u.one_arc != -1){ //bound 1-arcs by 1
              if (COLORlp_addcol(flow_lp, 0, (int *) nullptr, (double *) nullptr, obj,
                                 0.0, used_var_bound, var_type, nullptr))
                {
                  throw std::runtime_error("COLORlp_addcol.");
                }
                edge_index++;
            }
            //another variable for zero arc
            if (COLORlp_addcol(flow_lp, 0, (int *) nullptr, (double *) nullptr, obj,
                               0.0, used_var_bound, (formulation == OneArcsContinuous ? COLORlp_CONTINUOUS : var_type), nullptr))
            {
              throw std::runtime_error("COLORlp_addcol.");
            }
            edge_index++;
        }
    }
    int num_columns = edge_index;
    // introduced all variables and set objective function but have no rows/constraints

    //add rows in two phases: 1. in each level there is exactly one 1-arc with flow 1
    // 2. flow conservation and
    // 3. that variables are integer in range n is only explicitly added as constraint if options for that is given

    //1.
    edge_index = 0; //count/track edges while iterating over them
    for(unsigned int layer = 0; layer < n; layer++){
        std::vector<NodeIndex> vec_ind;
        for(Node &u : dd[layer]){
            //get index of the 1-arc of that node if it exists and set that cval to 1
            if(u.one_arc != -1){
                vec_ind.push_back(edge_index);
                edge_index++;
            }
            if(u.zero_arc != -1){
                edge_index++;
            } else{ //this should never happen since we don't consider the terminal node
                std::cout << "Error: there is no zero-arc!";
            }
        }
        std::vector<double> vec_val(vec_ind.size(), 1.0); //this is just a vector full of 1.0 the size of vec_ind
        int num_one_arcs = int(vec_ind.size());
        if (COLORlp_addrow(flow_lp, num_one_arcs, vec_ind.data(), vec_val.data(), COLORlp_GREATER_EQUAL, 1, nullptr))
        {
          throw std::runtime_error("COLORlp_addrow.");
        }
    }

    //2. flow conservation
    edge_index = 0;
    int node_index = 0;
    //map of nodes to edge index
    std::map<NodeIndex, std::vector<EdgeIndex> > old_incoming_arcs;
    Node &root = dd[0][0];
    if(root.one_arc != -1){
        old_incoming_arcs[root.one_arc].push_back(edge_index);
        edge_index++;
    }
    if(root.zero_arc != -1){
        old_incoming_arcs[root.zero_arc].push_back(edge_index);
        edge_index++;
    }
    std::map<NodeIndex, std::vector<EdgeIndex> > new_incoming_arcs;
    for(unsigned int layer = 1; layer < n; layer++){
        //find all incoming arcs of u and set cval to 1, for outgoing which are easy to find set cval to -1
        for(Node &u : dd[layer]){
            std::vector<NodeIndex> vec_ind;
            std::vector<double> vec_val;
            //incoming edges
            for(EdgeIndex e : old_incoming_arcs[u.index]){
                vec_ind.push_back(e);
                vec_val.push_back(1.0);
            }
            //outgoing edges
            if(u.one_arc != -1){
                new_incoming_arcs[u.one_arc].push_back(edge_index);
                vec_ind.push_back(edge_index);
                vec_val.push_back(-1.0);
                edge_index++;
            }
            if(u.zero_arc != -1){
                new_incoming_arcs[u.zero_arc].push_back(edge_index);
                vec_ind.push_back(edge_index);
                vec_val.push_back(-1.0);
                edge_index++;
            } else{ //this should never happen since we don't consider the terminal node
                std::cout << "Error: there is no zero-arc!" << std::endl;
            }

            int node_degree = int(vec_ind.size());
            if (COLORlp_addrow(flow_lp, node_degree, vec_ind.data(), vec_val.data(), COLORlp_EQUAL, 0, nullptr))
            {
              throw std::runtime_error("COLORlp_addrow.");
            }
            node_index++;
        }
        old_incoming_arcs = new_incoming_arcs;
        new_incoming_arcs.clear();
    }


    if (COLORlp_optimize(flow_lp)) {
      throw std::runtime_error("COLORlp_optimize failed.");
    }
    double flow_val = 0;
    if (COLORlp_objval(flow_lp, &flow_val)) {
      throw std::runtime_error("COLORlp_objval failed. ");
    }

    std::vector<double> solution;
    solution.reserve(num_columns);

    if (COLORlp_x(flow_lp, solution.data())) {
      throw std::runtime_error("COLORlp_x failed. ");
    }
    solution.assign(solution.data(), solution.data() + num_columns);

    double double_safe_bound = 0.0;
    if(model == LP){
        //Neumaier's method to obtain safe bounds with directed rounding if LP and strict bounds on variables
        const int originalRounding = fegetround();

        std::vector<double> dual_solution;
        unsigned int dd_size = num_nodes(dd);
        // dual solution consists of n variables for the constraints of each layer + the variables for each node except r,t
        // + possibly dual variables for the bound constraints, if those are used
        dual_solution.reserve(n + dd_size - 2);
        if (COLORlp_pi(flow_lp, dual_solution.data())) {
          throw std::runtime_error("COLORlp_pi failed. ");
        }

        dual_solution.assign(dual_solution.data(),
                             dual_solution.data() + n + dd_size - 2);

        fesetround(FE_DOWNWARD);
        long double dual_obj = std::accumulate(dual_solution.begin(), dual_solution.begin() + n,
                                               (long double) (0.0),
                                               [](long double sum, double var) { return sum + var; });

        fesetround(FE_UPWARD);

        //save total index of first node on each level
        //! shifted by one since root node has no dual variable
        std::vector<int> shifted_node_indices;
        shifted_node_indices.push_back(-1); // -1 to shift the indices
        for(int i = 1; i < int(dd.size()); i++){
            int next_level_index = shifted_node_indices[i - 1] + int(dd[i - 1].size());
            shifted_node_indices.push_back(next_level_index);
        }

        std::vector<long double> residual;
        residual.reserve(num_columns);
        for(unsigned int layer = 0; layer < n; layer++){
            for(int index = 0; index < int(dd[layer].size()); index++){
                Node &u = dd[layer][index];
                if(u.one_arc != -1){
                    long double res_val;
                    if(layer == 0){
                        res_val = (long double) 1 * dual_solution[layer] - 1 +
                                  1 * dual_solution[n + shifted_node_indices[layer + 1] + u.one_arc];
                    } else if(layer == n - 1){
                        res_val = (long double) 1 * dual_solution[layer] -
                                  1 * dual_solution[n + shifted_node_indices[layer] + index] + 0;
                    } else{
                        res_val = (long double) 1 * dual_solution[layer] -
                                  1 * dual_solution[n + shifted_node_indices[layer] + index] +
                                  1 * dual_solution[n + shifted_node_indices[layer + 1] + u.one_arc];
                    }
                    residual.push_back(res_val);
                }
                if(u.zero_arc != -1){
                    long double res_val;
                    if(layer == 0){
                        res_val = (long double) 0 * dual_solution[layer] - 1 +
                                  1 * dual_solution[n + shifted_node_indices[layer + 1] + u.zero_arc];
                    } else if(layer == n - 1){
                        res_val = (long double) 0 * dual_solution[layer] -
                                  1 * dual_solution[n + shifted_node_indices[layer] + index] + 0;
                    } else{
                        res_val = (long double) 0 * dual_solution[layer] -
                                  1 * dual_solution[n + shifted_node_indices[layer] + index] +
                                  1 * dual_solution[n + shifted_node_indices[layer + 1] + u.zero_arc];
                    }
                    residual.push_back(res_val);
                }
            }
        }

        if(num_columns != int(residual.size())){
            throw std::runtime_error("Not the correct sizes: " + std::to_string(num_columns) + " and " +
                                     std::to_string(residual.size()));
        }

        //compute delta with known upper bounds for primal variables
        long double delta = 0;
        for(long double &res_val : residual){
            delta += var_bound * std::max(res_val, (long double) (0.0));
        }

        fesetround(FE_DOWNWARD);

        long double safe_bound = dual_obj - delta;
        double_safe_bound = (double) safe_bound;//return this double as the safe lower bound

        fesetround(originalRounding);

        if(delta > std::pow(10, -5)){
            std::cout << "Warning: delta is quite big, larger than 10^-5: " << std::setprecision(25) << delta
                      << std::endl;
//            throw std::runtime_error("Delta");
        }
        if(std::ceil(double_safe_bound) != std::ceil(flow_val - std::pow(10, -5))){
            std::stringstream message;
            message << std::setprecision(25) << "Ceiling of z^* - 10^5 and double_safe_bound computed from Naumeier "
                            "were different: " +std::to_string(flow_val - std::pow(10, -5))
                            + " and " + std::to_string(double_safe_bound);
            std::cout << message.str() << std::endl;
//            throw std::runtime_error(message.str());
        }
    }

    //assign the flow to the edges of the decision diagram
    edge_index = 0;
    for(unsigned int layer = 0; layer < n; layer++){
        for(Node &u : dd[layer]){
            if(u.one_arc != -1){
                u.one_arc_flow = solution[edge_index];
                edge_index++;
            }
            if(u.zero_arc != -1){
                u.zero_arc_flow = solution[edge_index];
                edge_index++;
            }
        }
    }

    COLORlp_free(&flow_lp);

    if(model == LP){
        return double_safe_bound;
    }
    return flow_val;
}

void find_longest_path(const DecisionDiagram &dd, Path &path, Label &label) {
    int n = int(dd.size());
    //use that dd is dag and find longest path (longest in regards to 1-arcs) using topological order of the dd
    std::vector<std::vector<int> > prev(n);
    std::vector<std::vector<int> > dist(n);
    for(int layer = 0; layer < n; layer++){
        dist[layer] = std::vector<int>(dd[layer].size(), std::numeric_limits<int>::min());
        prev[layer].resize(dd[layer].size(), -1);
    }
    dist[0][0] = 0; //source vertex
    for(int layer = 0; layer < n - 1; layer++){
        for(int node = 0; node < int(dd[layer].size()); node++){
            if((dd[layer][node].one_arc != -1) and (dist[layer + 1][dd[layer][node].one_arc] < dist[layer][node] + 1)){
                dist[layer + 1][dd[layer][node].one_arc] = dist[layer][node] + 1;
                prev[layer + 1][dd[layer][node].one_arc] = node;
            }
            if(dist[layer + 1][dd[layer][node].zero_arc] < dist[layer][node]){
                dist[layer + 1][dd[layer][node].zero_arc] = dist[layer][node];
                prev[layer + 1][dd[layer][node].zero_arc] = node;
            }
        }
    }
    path.clear();
    path.reserve(n);
    label.clear();
    label.reserve(n);

    int prev_node = 0;
    path.push_back(prev_node);
    for(int layer = n - 1; layer > 0; layer--){
        prev_node = prev[layer][prev_node];
        if(dd[layer - 1][prev_node].one_arc == path.back()){
            label.push_back(oneArc);
        } else{
            label.push_back(zeroArc);
        }
        path.push_back(prev_node);
    }

    std::reverse(path.begin(), path.end());
    std::reverse(label.begin(), label.end());
}

PathLabelConflict conflict_on_longest_path(const DecisionDiagram &dd, const NeighborList &neighbors) {
    Path path;
    Label label;
    find_longest_path(dd, path, label);

    //look for a conflict on said longest path
    for(int j = 1; j < int(path.size()-1); j++){
       if(not label[j]) {
            continue;
        }
        //iterating this way we find the first smallest conflict, important assumption for Algorithm 1
        for(int i = j - 1; i >= 0; i--){
            if(not label[i]){
                continue;
            }
            //both vertex i and j of original graph are chosen. check if that is a conflict or not, i.e. if they are adj
            if(std::find(neighbors[i].begin(), neighbors[i].end(), (j + 1)) != neighbors[i].end()){
                return PathLabelConflict(Path(path.begin() + i, path.begin() + j + 1),
                                         Label(label.begin() + i, label.begin() + j), std::make_tuple(i + 1, j + 1));
            }
        }
    }
    return {{}, {}, std::make_tuple(-1, -1)};
}
