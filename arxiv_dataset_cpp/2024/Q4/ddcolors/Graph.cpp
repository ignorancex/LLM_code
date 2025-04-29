#include "Graph.h"


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

template<typename T>
T random_element(const std::vector<T> &v) {
    auto it = v.cbegin();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, v.size() - 1);
    int random = distr(gen);
    std::advance(it, random);
    return *it;
}

Permutation identity(size_t size) {
    Permutation identity(size);
    for(unsigned int i = 1; i <= size; i++)
        identity[i - 1] = i;
    return identity;
}

Permutation perm_inverse(const Permutation &perm) {
    Permutation inverse(perm.size());
    for(unsigned int i = 1; i <= perm.size(); i++){
        inverse[perm[i - 1] - 1] = i;
    }
    return inverse;
}


NeighborList Graph::get_neighbor_list() const {
    NeighborList list;
    list.resize(_ncount);
    for(unsigned int i = 0; i < _ecount; i++){
        list[_elist[2 * i] - 1].push_back( _elist[2 * i + 1]);
        list[_elist[2 * i + 1] - 1].push_back( _elist[2 * i]);
    }
    for(unsigned int i = 0; i < _ncount; i++){//like it better if the neighbors are sorted
        std::sort(list[i].begin(), list[i].end());
    }
    return list;
}

Graph::Graph(unsigned int vertex_count, unsigned int edge_count, std::vector<Vertex> edge_list) :
        _ncount(vertex_count), _ecount(edge_count), _elist(std::move(edge_list)) {}

unsigned int Graph::ncount() const {
    return _ncount;
}

unsigned int Graph::ecount() const {
    return _ecount;
}

void Graph::print() const {
    std::cout << "Graph has " << _ncount << " vertices and " << _ecount << " edges." << std::endl;
    NeighborList neighbors = get_neighbor_list();
    for(auto &adj : neighbors){
        std::cout << adj << std::endl;
    }
}

std::vector<Vertex> Graph::elist() const {
    return _elist;
}

std::vector<int> Graph::shifted_elist() const {
    std::vector<int> shifted_edge_list(2 * _ecount);
    std::transform(_elist.begin(), _elist.end(), shifted_edge_list.begin(),
                   [](Vertex v) {
                       if(v == 0) throw std::runtime_error("Vertex is 0, cannot shift it.");
                       return (int) (v - 1);
                   });
    return shifted_edge_list;
}

std::vector<int> Graph::complement_shifted_elist() const {
    NeighborList neighbors = get_neighbor_list();
    int complement_ecount = int((_ncount * (_ncount - 1))) / 2 - int(_ecount);
    std::vector<int> complement_edge_list(2 * complement_ecount);
    int cecount = 0;
    for(int v = 0; v < _ncount; v++){
        int a = -1;
        int a_i  = 0;
        for (int na = v + 1; na < _ncount; ++ na) {
            while (a_i < neighbors[v].size() && a < na) {
                a = neighbors[v][a_i] - 1;
                ++a_i;
            }
            if (na != a) {
                complement_edge_list[2*cecount]   = v;
                complement_edge_list[2*cecount+1] = na;
                ++cecount;
            }
        }
    }
    if(cecount != complement_ecount){
        throw std::runtime_error("Did not produce the right amount of complement edges.");
    }
    return complement_edge_list;
}



Graph::Graph(const char *filename) : _ncount(0), _ecount(0) {
    std::string string_name = filename;
    if(string_name.substr(string_name.find_last_of('.') + 1) == "col"){
        read_dimacs(filename);
    } else if(string_name.substr(string_name.find_last_of('.') + 1) == "g6"){
        read_graph6(filename);
    } else if(string_name.substr(string_name.find_last_of('.') + 1) == "b"){
        throw std::runtime_error("Binary dimacs format is currently not supported, please translate it yourself.");
    } else{
        throw std::runtime_error("File format is not accepted.");
    }
    simplify_graph();
}

Graph::Graph(std::string g6_string) : _ncount(0), _ecount(0) {
    read_graph6_string(std::move(g6_string));
}


void Graph::read_dimacs(const char *filename) {
    std::ifstream file(filename);
    if(not file){
        throw std::runtime_error("Cannot open file.");
    }

    std::string line;
    std::getline(file, line);//read in the file line by line

    while(line[0] == 'c' or line[0] == '\r' or line[0] == '\0'){          //special character to account for empty lines
        std::getline(file, line);                                                               //ignore comments
    }
    if(line[0] != 'p'){
        throw std::runtime_error("File is not in correct dimacs format.");          //a different case was expected
    }
    std::string p, edge;
    int n, e;
    std::stringstream ss(line);
    ss >> p >> edge >> n >> e;                                                       //read in number of nodes and edges

    _ncount = n;
    _ecount = e;
    _elist.reserve(2 * _ecount);

    std::getline(file, line);
    while(line[0] == 'n'){
        std::cout << "The dimacs file is specifying colors of vertices, these will be ignored for this problem."
                  << std::endl;
        std::getline(file, line);
    }

    char first;
    int head, tail;
    do{
        while(line[0] == 'c' or line[0] == '\r' or
              line[0] == '\0'){          //special character to account for empty lines
            std::getline(file, line);                                                           //ignore comments
        }
        ss.clear();
        ss.str(std::string());
        ss << line;
        ss >> first >> head >> tail;
        _elist.push_back(head);
        _elist.push_back(tail);
    } while(std::getline(file, line));
}

void Graph::read_graph6(const char *filename) {
    std::ifstream file(filename);
    if(not file){
        throw std::runtime_error("Cannot open file.");
    }
    std::cout << "Warning: if the graph6 file contains more than one graph, the first is read in" << std::endl;
    std::string line;
    std::getline(file, line);
    read_graph6_string(line);
}

//this function of reading in the graph6 format was taken from treedecomposition.com
void Graph::read_graph6_string(std::string g6_string) {// Extract vertex count from graph6
    if(g6_string.substr(0, 1) == ":"){
        g6_string = g6_string.substr(1, g6_string.size());
    }

    int vertexCount = -1;

    int r0 = g6_string[0];
    int adjIdx = 1;

    // single byte expansion (0 <= n <= 62)
    if(r0 >= 0 + 63 and r0 <= 63 + 62){
        vertexCount = r0 - 63;
        adjIdx = 1;
    }
        // four or eight byte expansions
    else if(r0 == 126){
        char r1 = g6_string[1];

        // eight byte expansions (258048 <= n <= 68719476735)
        if(r1 == 126){
            adjIdx = 8;
            throw std::runtime_error("Terribly sorry, but we don\t do 8-byte expansions yet :(");
        }
            // four byte expansion (63 <= n <= 258047)
        else{
            if(g6_string.size() < 4){
                throw std::runtime_error("This input seems to be too short for a 4-byte expansion :(");
            } else{
                vertexCount = 0;
                adjIdx = 4;

                for(int i = 0; i < 3; i++){
                    vertexCount = vertexCount | ((g6_string[1 + i] - 63) << ((2 - i) * 6));
                }
            }
        }
    }

    // Only start working when we actually have a vertex count
    if(vertexCount > -1){
        _ncount = vertexCount;
        int edgeCount = 0;
        // Extract adjacency list from graph6,
        for(int n = 1; n < vertexCount; n++){
            // Natural numbers sum n*(n+1)/2, but only need previous n so (n-1)*(n+1-1)/2
            int nPos = n * (n - 1) / 2;
            for(int m = 0; m < n; m++){
                int bitPos = nPos + m;
                int charIndex = adjIdx + int(std::floor(bitPos / 6));
                int charVal = g6_string[charIndex] - 63;

                int bitIndex = bitPos % 6;
                int bitShift = (5 - bitIndex);
                int bitMask = (1 << bitShift);

                int bitVal = ((charVal & bitMask) >> bitShift);

                // If the bit for this particular position is 1 then {m, n} is an edge
                if(bitVal == 1){
                    _elist.push_back(n + 1);
                    _elist.push_back(m + 1);
                    edgeCount++;
                }
            }
        }
        _ecount = edgeCount;
    } else{
        throw std::runtime_error("Was not able to get vertex count from graph6 file.");
    }
}


void Graph::simplify_graph(){

    NeighborList neighbors = get_neighbor_list();
    std::vector<Vertex> simplified_elist;
    simplified_elist.reserve(2*_ecount);
    int simplified_ecount = 0;
    for(Vertex v = 1; v <= _ncount; v++){
        if(neighbors[v-1].empty())
            continue;
        Vertex w;
        //add first edge if it is not a loop and and edge from smaller to larger vertex
        if(v < neighbors[v - 1][0]){
            simplified_elist.push_back(v);
            simplified_elist.push_back(neighbors[v - 1][0]);
            simplified_ecount++;
        }
        for(unsigned int j = 1; j < neighbors[v-1].size(); j++ ){
            //test for loops or duplicate edges, uses that the adjacency lists from get_neighbor_list are sorted
            //only add edge from smalle to larger vertex, also skip otherwise
            if( (v < neighbors[v - 1][j] ) and (neighbors[v - 1][j - 1] != neighbors[v - 1][j]) ){
                simplified_elist.push_back(v);
                simplified_elist.push_back(neighbors[v - 1][j]);
                simplified_ecount++;
            }
        }
    }
    if(simplified_ecount != (simplified_elist.size() / 2)){
        throw std::runtime_error("Did not simplify the graph correctly.");
    }
    _elist = simplified_elist;
    _elist.shrink_to_fit();
    _ecount = simplified_ecount;
}


Graph Graph::perm_graph(const Permutation &perm) const {
    if(perm.size() != _ncount){
        throw std::runtime_error("Size of graph and permutation do not match.");
    }
    Graph permuted_graph(_ncount, _ecount, {});
    permuted_graph._elist.resize(2 * _ecount);
    for(size_t i = 0; i < 2 * _ecount; i++){
        permuted_graph._elist[i] = perm[_elist[i] - 1];
    }
    return permuted_graph;
}

Coloring Graph::dsatur(Permutation &ordering, const std::vector<Vertex> &clique) const {
    NeighborList neighbors = get_neighbor_list();
    return dsatur(ordering, neighbors, clique);
}

Coloring Graph::dsatur(Permutation &ordering, const NeighborList &neighbors, const std::vector<Vertex> &clique) const {
    ordering.clear();
    ordering.reserve(_ncount);
    Coloring coloring;
    //initialise all vertices to be uncolored and have zero saturation
    std::vector<int> vertex_color(_ncount, -1);
    //record for an uncolored vertex to how many colored vertices it is connected to
    std::vector<int> saturation_level(_ncount, 0);

    int nclique_colors = 0;
    if(clique.size() > 2){//ignore trivial or empty clique
        //fix color of the vertices of supplied clique and set saturation levels
        int color_index = 0;
        for(Vertex v : clique){
            ordering.push_back(v);
            coloring.push_back({}); //new color class
            coloring[color_index].insert(v); //assign first color to max degree vertex
            vertex_color[v - 1] = color_index;
            color_index++;
            //update saturation levels
            for(Vertex neighbor : neighbors[v - 1]){
                saturation_level[neighbor - 1]++;
            }
            saturation_level[v - 1] = std::numeric_limits<int>::min();
        }
        nclique_colors = int(clique.size());
    }

    //main loop looking for the next vertex according to its saturation level
   for(int iteration = 0; iteration < _ncount - nclique_colors; iteration++){
        Vertex max_saturated_vertex = 0;
        int max_saturated = -1;
        //keep set of tied vertices to later choose one at random
        std::vector<Vertex> saturated_tied_max_degrees;
        for(Vertex vertex = 1; vertex <= _ncount; vertex++){
            if(vertex_color[vertex - 1] != -1)
                continue;

            if(saturation_level[vertex - 1] > max_saturated){
                //found vertex with higher saturation, update
                max_saturated = saturation_level[vertex - 1];
                max_saturated_vertex = vertex;

                saturated_tied_max_degrees.clear();
                saturated_tied_max_degrees.push_back(vertex);
            } else if(saturation_level[vertex - 1] == max_saturated){
                if(neighbors[vertex - 1].size() > neighbors[max_saturated_vertex - 1].size()){
                    //found vertex with equal saturation but higher degree, update
                    max_saturated_vertex = vertex;

                    saturated_tied_max_degrees.clear();
                    saturated_tied_max_degrees.push_back(vertex);
                } else if(neighbors[vertex - 1].size() == neighbors[max_saturated_vertex - 1].size()){
                    //same saturation level, same degree, add to set of possible candidates
                    saturated_tied_max_degrees.push_back(vertex);
                }
            }
        }
        //select max saturated vertex, at random if there exist multiple
        if(saturated_tied_max_degrees.size() > 1 and random_tiebreaks){
            max_saturated_vertex = random_element(saturated_tied_max_degrees);
        }
        ordering.push_back(max_saturated_vertex);

       int lowest_color = 0;
       int failed;
       do {
           failed = 0;
           for (Vertex neighbor : neighbors[max_saturated_vertex - 1]) {
               if (vertex_color[neighbor - 1] == lowest_color) {
                   ++lowest_color;
                   failed = 1;
                   break;
               }
           }
       } while (failed);


        if(lowest_color < int(coloring.size())){
            //we can add chosen vertex to some existing color class
            coloring[lowest_color].insert(max_saturated_vertex);
            vertex_color[max_saturated_vertex - 1] = lowest_color;
        } else if(lowest_color == int(coloring.size())){
            //new color class is created for max_saturated_vertex
            //before that we try to recolor two vertices to not have to use a new color
            bool recolored = try_color_swap(max_saturated_vertex, neighbors, coloring, vertex_color);
            if(not recolored){
                //not able to recolor, add new color class
                coloring.push_back({});
                coloring[lowest_color].insert(max_saturated_vertex);
                vertex_color[max_saturated_vertex - 1] = lowest_color;
            }
        } else{
            std::cout << "Error: lowest color is too high, something must have gone wrong." << std::endl;
        }

        //new vertex has been selected so update the saturation levels of all its neighbors if it's a different color
        int msv_color = vertex_color[max_saturated_vertex - 1];
        for(Vertex vertex : neighbors[max_saturated_vertex - 1]){
            bool is_new_color = true;
            //possibly check that neighbor has been selected, i.e. it's level is equal to numeric_limits<int>::min()
            if(saturation_level[vertex - 1] != std::numeric_limits<int>::min()){
                for(Vertex adj_vertex : neighbors[vertex - 1]){
                    if(adj_vertex != max_saturated_vertex and vertex_color[adj_vertex - 1] == msv_color){
                        //found neighbor of neighbor that uses msv color so we don't increase the saturation level
                        is_new_color = false;
                        break;
                    }
                }
                if(is_new_color)
                    saturation_level[vertex - 1]++;
            }
        }
        //remove selected vertex from any further consideration
        saturation_level[max_saturated_vertex - 1] = std::numeric_limits<int>::min();
    }
    //colored all vertices
    return coloring;
}


Coloring Graph::dsatur_original(Permutation &ordering, const std::vector<Vertex> &clique) const {
    NeighborList neighbors = get_neighbor_list();
    return dsatur_original(ordering, neighbors, clique);
}


Coloring
Graph::dsatur_original(Permutation &ordering, const NeighborList &neighbors, const std::vector<Vertex> &clique) const {
    ordering.clear();
    ordering.reserve(_ncount);
    Coloring coloring;
    //initialise all vertices to be uncolored
    std::vector<int> vertex_color(_ncount, -1);
    //record for an uncolored vertex to how many colored vertices it is connected
    //also track degree of degree in uncolored subgraph
    std::vector<int> saturation_level(_ncount, 0);
    std::vector<int> uncolored_subgraph_degree(_ncount, 0);
    for(Vertex v = 1; v <= _ncount; v++)
        uncolored_subgraph_degree[v - 1] = int(neighbors[v - 1].size());

    int nclique_colors = 0;
    if(clique.size() > 2){//ignore trivial or empty clique
        //fix color of the vertices of supplied clique and set saturation levels
        int color_index = 0;
        for(Vertex v : clique){
            ordering.push_back(v);
            coloring.push_back({}); //new color class
            coloring[color_index].insert(v); //assign first color to max degree vertex
            vertex_color[v - 1] = color_index;
            color_index++;
            //update saturation levels
            for(Vertex neighbor : neighbors[v - 1]){
                saturation_level[neighbor - 1]++;
                uncolored_subgraph_degree[neighbor - 1]--;
            }
            saturation_level[v - 1] = std::numeric_limits<int>::min();
            uncolored_subgraph_degree[v - 1] = std::numeric_limits<int>::min();
        }
        nclique_colors = int(clique.size());
    }


    //main loop looking for the next vertex according to its saturation level
    for(int iteration = 0; iteration < _ncount - nclique_colors; iteration++){
        Vertex max_saturated_vertex = 0;
        int max_saturated = -1;
        //keep set of tied vertices to later choose one at random
        std::vector<Vertex> saturated_tied_max_degrees;
        for(Vertex vertex = 1; vertex <= _ncount; vertex++){
            if(vertex_color[vertex - 1] != -1)
                continue;
            if(saturation_level[vertex - 1] > max_saturated){
                //found vertex with higher saturation, update
                max_saturated = saturation_level[vertex - 1];
                max_saturated_vertex = vertex;

                saturated_tied_max_degrees.clear();
                saturated_tied_max_degrees.push_back(vertex);
            } else if(saturation_level[vertex - 1] == max_saturated){
                if(uncolored_subgraph_degree[vertex - 1] > uncolored_subgraph_degree[max_saturated_vertex - 1]){
                    //found vertex with equal saturation but higher degree, update
                    max_saturated_vertex = vertex;

                    saturated_tied_max_degrees.clear();
                    saturated_tied_max_degrees.push_back(vertex);
                } else if(neighbors[vertex - 1].size() == neighbors[max_saturated_vertex - 1].size()){
                    //same saturation level, same degree, add to set of possible candidates
                    saturated_tied_max_degrees.push_back(vertex);
                }
            }
        }
        //select max saturated vertex, at random if there exist multiple
        if(saturated_tied_max_degrees.size() > 1 and random_tiebreaks){
            max_saturated_vertex = random_element(saturated_tied_max_degrees);
        }
        ordering.push_back(max_saturated_vertex);

        int lowest_color = 0;
        int failed;
        do {
            failed = 0;
            for (Vertex neighbor : neighbors[max_saturated_vertex - 1]) {
                if (vertex_color[neighbor - 1] == lowest_color) {
                    ++lowest_color;
                    failed = 1;
                    break;
                }
            }
        } while (failed);

        if(lowest_color < int(coloring.size())){
            //we can add chosen vertex to some existing color class
            coloring[lowest_color].insert(max_saturated_vertex);
            vertex_color[max_saturated_vertex - 1] = lowest_color;
        } else if(lowest_color == int(coloring.size())){
            //new color class is created for max_saturated_vertex
            //before that we try to recolor two vertices to not have to use a new color
            bool recolored = try_color_swap(max_saturated_vertex, neighbors, coloring, vertex_color);
            if(not recolored){
                //not able to recolor, add new color class
                coloring.push_back({});
                coloring[lowest_color].insert(max_saturated_vertex);
                vertex_color[max_saturated_vertex - 1] = lowest_color;
            }
        } else{
            std::cout << "Error: lowest color is too high, something must have gone wrong." << std::endl;
        }

        //new vertex has been selected so update the saturation levels of all its neighbors if it's a different color
        int msv_color = vertex_color[max_saturated_vertex - 1];
        for(Vertex vertex : neighbors[max_saturated_vertex - 1]){
            bool is_new_color = true;
            //possibly check that neighbor has been selected, i.e. it's level is equal to numeric_limits<int>::min()
            if(saturation_level[vertex - 1] != std::numeric_limits<int>::min()){
                for(Vertex adj_vertex : neighbors[vertex - 1]){
                    if(adj_vertex != max_saturated_vertex and vertex_color[adj_vertex - 1] == msv_color){
                        //found neighbor of neighbor that uses msv color so we don't increase the saturation level
                        is_new_color = false;
                        break;
                    }
                }
                if(is_new_color){
                    saturation_level[vertex - 1]++;
                }
                //also update degree of neighbor in uncolored subgraph
                uncolored_subgraph_degree[vertex - 1]--;
            }
        }
        //remove selected vertex from any further consideration
        saturation_level[max_saturated_vertex - 1] = std::numeric_limits<int>::min();
        uncolored_subgraph_degree[max_saturated_vertex - 1] = std::numeric_limits<int>::min();
    }
    //colored all vertices
    return coloring;
}


Coloring Graph::max_connected_degree_coloring(Permutation &ordering, const std::vector<Vertex> &clique) const {
    NeighborList neighbors = get_neighbor_list();
    return max_connected_degree_coloring(ordering, neighbors, clique);
}

Coloring Graph::max_connected_degree_coloring(Permutation &ordering, const NeighborList &neighbors,
                                              const std::vector<Vertex> &clique) const {
    ordering.clear();
    ordering.reserve(_ncount);
    Coloring coloring;
    //initialise all vertices to be uncolored
    std::vector<int> vertex_color(_ncount, -1);
    //record for an uncolored vertex to how many colored vertices it is connected
    std::vector<int> num_selected_neighbors(_ncount, 0);

    int nclique_colors = 0;
    if(clique.size() > 2){//ignore trivial or empty clique
        //fix color of the vertices of supplied clique and set saturation levels
        int color_index = 0;
        for(Vertex v : clique){
            ordering.push_back(v);
            coloring.push_back({}); //new color class
            coloring[color_index].insert(v); //assign first color to max degree vertex
            vertex_color[v - 1] = color_index;
            color_index++;
            //update saturation levels
            for(Vertex neighbor : neighbors[v - 1]){
                num_selected_neighbors[neighbor - 1]++;
            }
            num_selected_neighbors[v - 1] = std::numeric_limits<int>::min();
        }
        nclique_colors = int(clique.size());
    }


    //main loop looking for the next vertex according to its saturation level
    for(int iteration = 0; iteration < _ncount - nclique_colors; iteration++){
        Vertex max_saturated_vertex = 0;
        int max_saturated = -1;
        //keep set of tied vertices to later choose one at random
        std::vector<Vertex> saturated_tied_max_degrees;
        for(Vertex vertex = 1; vertex <= _ncount; vertex++){
            if(vertex_color[vertex - 1] != -1)
                continue;

            if(num_selected_neighbors[vertex - 1] > max_saturated){
                //found vertex with higher saturation, update
                max_saturated = num_selected_neighbors[vertex - 1];
                max_saturated_vertex = vertex;

                saturated_tied_max_degrees.clear();
                saturated_tied_max_degrees.push_back(vertex);
            } else if(num_selected_neighbors[vertex - 1] == max_saturated){
                if(neighbors[vertex - 1].size() > neighbors[max_saturated_vertex - 1].size()){
                    //found vertex with equal saturation but higher degree, update
                    max_saturated_vertex = vertex;

                    saturated_tied_max_degrees.clear();
                    saturated_tied_max_degrees.push_back(vertex);
                } else if(neighbors[vertex - 1].size() == neighbors[max_saturated_vertex - 1].size()){
                    //same saturation level, same degree, add to set of possible candidates
                    saturated_tied_max_degrees.push_back(vertex);
                }
            }
        }
        //select max saturated vertex, at random if there exist multiple
        if(saturated_tied_max_degrees.size() > 1 and random_tiebreaks){
            max_saturated_vertex = random_element(saturated_tied_max_degrees);
        }
        ordering.push_back(max_saturated_vertex);

        int lowest_color = 0;
        int failed;
        do {
            failed = 0;
            for (Vertex neighbor : neighbors[max_saturated_vertex - 1]) {
                if (vertex_color[neighbor - 1] == lowest_color) {
                    ++lowest_color;
                    failed = 1;
                    break;
                }
            }
        } while (failed);

        if(lowest_color < int(coloring.size())){
            //we can add chosen vertex to some existing color class
            coloring[lowest_color].insert(max_saturated_vertex);
            vertex_color[max_saturated_vertex - 1] = lowest_color;
        } else if(lowest_color == int(coloring.size())){
            //new color class is created for max_saturated_vertex
            //before that we try to recolor two vertices to not have to use a new color
            bool recolored = try_color_swap(max_saturated_vertex, neighbors, coloring, vertex_color);
            if(not recolored){
                //not able to recolor, add new color class
                coloring.push_back({});
                coloring[lowest_color].insert(max_saturated_vertex);
                vertex_color[max_saturated_vertex - 1] = lowest_color;
            }
        } else{
            std::cout << "Error: lowest color is too high, something must have gone wrong." << std::endl;
        }

        //new vertex has been selected so update the the number of colored neighbors for that vertex's neighbors
        for(Vertex vertex : neighbors[max_saturated_vertex - 1]){
            //possibly check that neighbor has been selected, i.e. it's level is equal to numeric_limits<int>::min()
            if(num_selected_neighbors[vertex - 1] != std::numeric_limits<int>::min()){
                num_selected_neighbors[vertex - 1]++;
            }
        }
        //remove selected vertex from any further consideration
        num_selected_neighbors[max_saturated_vertex - 1] = std::numeric_limits<int>::min();
    }
    return coloring;
}


Permutation Graph::max_connected_degree_ordering(const std::vector<Vertex> &clique) const {
    NeighborList neighbors = get_neighbor_list();
    return max_connected_degree_ordering(neighbors, clique);
}


Permutation Graph::max_connected_degree_ordering(const NeighborList &neighbors, const std::vector<Vertex> &clique) const {
    Permutation ordering;
    ordering.reserve(_ncount);
    //record for an unselected vertex to how many selected vertices it is connected
    std::vector<int> num_selected_neighbors(_ncount);
    std::vector<int> vertex_is_colored(_ncount, -1);

    int nclique_colors = 0;
    if(clique.size() > 2){//ignore trivial or empty clique
        //fix color of the vertices of supplied clique and set saturation levels
        for(Vertex v : clique){
            ordering.push_back(v);
            vertex_is_colored[v - 1] = 1;
            //update saturation levels
            for(Vertex neighbor : neighbors[v - 1]){
                num_selected_neighbors[neighbor - 1]++;
            }
            num_selected_neighbors[v - 1] = std::numeric_limits<int>::min();
        }
        nclique_colors = int(clique.size());
    }

    for(int iteration = 0; iteration < _ncount - nclique_colors; iteration++){
        Vertex max_connections_vertex = 0;
        int max_connections = -1;
        for(Vertex vertex = 1; vertex <= _ncount; vertex++){
            if(vertex_is_colored[vertex - 1] != -1)
                continue;
            if(num_selected_neighbors[vertex - 1] > max_connections){
                max_connections = num_selected_neighbors[vertex - 1];
                max_connections_vertex = vertex;
            } else if(num_selected_neighbors[vertex - 1] == max_connections and
                      (neighbors[vertex - 1].size() > neighbors[max_connections_vertex - 1].size())){
                max_connections_vertex = vertex; //saturation is the same but vertex has a higher degree
            }
        }
        //select max conneceted vertex
        ordering.push_back(max_connections_vertex);
        vertex_is_colored[max_connections_vertex - 1] = 1;

        for(Vertex neighbor : neighbors[max_connections_vertex - 1]){
            if(num_selected_neighbors[neighbor - 1] != std::numeric_limits<int>::min()){
                num_selected_neighbors[neighbor - 1]++;
            }
        }
        //remove selected vertex from any further consideration
        num_selected_neighbors[max_connections_vertex - 1] = std::numeric_limits<int>::min();
    }
    return ordering;
}


int Graph::constraint_graph_width() {
    Graph g = *this;
    g.peel_graph(1); //remove isolated vertices
    NeighborList neighbors = g.get_neighbor_list();
    //begin peeling starting from the minimum degree
    int k = std::min_element(neighbors.begin(), neighbors.end(),
                             [](const std::vector<Vertex> &d_1, const std::vector<Vertex> &d_2) {
                                 return d_1.size() < d_2.size();
                             })->size();
    while(g.ncount()){
        k++;
        unsigned int current_node_count = g.ncount();
        while(true){
            g.peel_graph(k);
            if(current_node_count == g.ncount() or g.ncount() == 0){
                break;
            }
            current_node_count = g.ncount();
        }
    }
    return k;
}

Permutation Graph::constraint_graph_ordering() {
    int width = constraint_graph_width();
    int n = int(_ncount);
    Permutation ordering(n);
    std::set<Vertex> unselected_vertices;
    NeighborList neighbors = get_neighbor_list();
    std::vector<int> degree(_ncount);
    for(Vertex i = 1; i <= _ncount; ++i){
        unselected_vertices.insert(unselected_vertices.end(), i);
        degree[i - 1] = int(neighbors[i - 1].size());
    }

    for(int i = n; i >= 1; i--){
        for(auto v_it = unselected_vertices.rbegin(); v_it != unselected_vertices.rend(); v_it++){
            Vertex v = *v_it;
            if(degree[v - 1] <= width + 1){
                ordering[i - 1] = v;
                unselected_vertices.erase(v);
                for(auto neighbor : neighbors[v - 1]){
                    degree[neighbor - 1]--;
                }
                break;
            }
        }
    }
    return ordering;
}


bool Graph::try_color_swap(Vertex max_saturated_vertex, const NeighborList &neighbors, Coloring &coloring,
                           std::vector<int> &vertex_color) {
    for(unsigned int j = 0; j < coloring.size(); j++){
        for(unsigned int k = j + 1; k < coloring.size(); k++){
            std::set<Vertex> neighbors_v_intersect_color_j;
            std::set_intersection(neighbors[max_saturated_vertex - 1].begin(),
                                  neighbors[max_saturated_vertex - 1].end(),
                                  coloring[j].begin(), coloring[j].end(),
                                  std::inserter(neighbors_v_intersect_color_j, neighbors_v_intersect_color_j.end()));

            if(neighbors_v_intersect_color_j.size() == 1){
                Vertex u = *neighbors_v_intersect_color_j.begin();
                std::set<Vertex> neighbors_u_intersect_color_k;
                std::set_intersection(neighbors[u - 1].begin(), neighbors[u - 1].end(),
                                      coloring[k].begin(), coloring[k].end(),
                                      std::inserter(neighbors_u_intersect_color_k,
                                                    neighbors_u_intersect_color_k.end()));

                if(neighbors_u_intersect_color_k.empty()){
                    //we can swap colors around, i.e. color max_saturated vertex and u with j and k respectively
                    coloring[j].erase(u);
                    coloring[j].insert(max_saturated_vertex);
                    coloring[k].insert(u);
                    vertex_color[max_saturated_vertex - 1] = int(j);
                    vertex_color[u - 1] = int(k);
                    k = coloring.size();
                    j = coloring.size();//exit loops, but exiting anyways
                    return true;//was able to swap colors
                }
            }
        }
    }
    return false; //unable to swap two colors
}


Permutation Graph::max_degree_ordering() const {
    NeighborList neighbors = get_neighbor_list();
    return max_degree_ordering(neighbors);
}


Permutation Graph::max_degree_ordering(const NeighborList &neighbors) const {
    Permutation ordering;
    ordering.reserve(_ncount);
    for(Vertex i = 1; i <= _ncount; i++){
        ordering.push_back(i);
    }
    std::sort(ordering.begin(), ordering.end(),
              [&neighbors](Vertex v, Vertex w) { return neighbors[v - 1].size() < neighbors[w - 1].size(); });
    std::reverse(ordering.begin(), ordering.end());

    return ordering;
}


void Graph::peel_graph(int peeling_degree) {
    NeighborList neighbors = get_neighbor_list();
    peel_graph(peeling_degree, neighbors);
}

void Graph::peel_graph(int peeling_degree, const NeighborList &neighbors) {
    std::set<Vertex> small_degree_vertices;
    for(Vertex i = 1; i <= _ncount; i++){
        if(int(neighbors[i - 1].size()) < peeling_degree){
            small_degree_vertices.insert(small_degree_vertices.end(), i);
        }
    }
    remove_vertices_together(small_degree_vertices);
}

void Graph::remove_vertex(Vertex remove) {
    for(int edge_index = int(_ecount) - 1; edge_index >= 0; edge_index--){
        if(remove == _elist[2 * edge_index] or remove == _elist[2 * edge_index + 1]){
            _elist.erase(std::next(_elist.begin(), 2 * edge_index + 1));
            _elist.erase(std::next(_elist.begin(), 2 * edge_index));
            _ecount--;
        } else{
            if(_elist[2 * edge_index + 1] > remove) _elist[2 * edge_index + 1]--;
            if(_elist[2 * edge_index] > remove) _elist[2 * edge_index]--;
        }
    }
    _ncount--;
}


void Graph::remove_vertices(const std::set<Vertex> &to_remove) {
    //order of removing vertices is important here, by removing later vertices first, we keep the labeling intact
    for(auto v_it = to_remove.rbegin(); v_it != to_remove.rend(); v_it++){
        remove_vertex(*v_it);
    }
}

//just a tiny helper function to remove vertices all at once
int Graph::num_larger(Vertex v, const std::set<Vertex>& set){
    if(set.empty())
        return 0;
    auto num = std::count_if(set.begin(), set.end(), [&v](Vertex set_vertex){return v > set_vertex;});
    return int(num);
}

void Graph::remove_vertices_together(const std::set<Vertex> &to_remove) {
    for(int edge_index = int(_ecount) - 1; edge_index >= 0; edge_index--){
        if(to_remove.count(_elist[2 * edge_index]) or to_remove.count(_elist[2 * edge_index + 1])){
            _elist.erase(std::next(_elist.begin(), 2 * edge_index + 1));
            _elist.erase(std::next(_elist.begin(), 2 * edge_index));
            _ecount--;
        } else{
            _elist[2 * edge_index + 1] -= num_larger(_elist[2 * edge_index + 1], to_remove);
            _elist[2 * edge_index] -= num_larger(_elist[2 * edge_index], to_remove);
        }
    }
    _ncount -= to_remove.size();
}


void Graph::remove_dominated_vertices() {
    NeighborList neighbors = get_neighbor_list();
    remove_dominated_vertices(neighbors);
}


void Graph::remove_dominated_vertices(const NeighborList &neighbors) {
    std::set<Vertex> dominated_vertices;
    for(Vertex v = 1; v <= _ncount; v++){
        for(Vertex w = v + 1; w <= _ncount; w++){
            //vertices are not neighbors
            if(std::find(neighbors[v - 1].begin(), neighbors[v - 1].end(), w) != neighbors[v - 1].end())
                continue;
            //vertices have not been marked as dominated
            if(dominated_vertices.count(v) or dominated_vertices.count(w)){
                continue;
            }

            std::vector<Vertex> intersection;
            intersection.reserve(std::min(neighbors[v - 1].size(), neighbors[w - 1].size()));
            std::set_intersection(neighbors[v - 1].begin(), neighbors[v - 1].end(),
                                  neighbors[w - 1].begin(), neighbors[w - 1].end(),
                    std::back_inserter(intersection));
            if(intersection == neighbors[v - 1]){
                //neighbors of v included in neighbors of w
                dominated_vertices.insert(dominated_vertices.end(), v);
                break;//v has been marked as vertex to be removed
            }
            else if(intersection == neighbors[w - 1]){
                //neighbors of w included in neighbors of v
                dominated_vertices.insert(dominated_vertices.end(), w);
            }
        }
    }
    remove_vertices_together(dominated_vertices);
}


std::vector<Vertex> Graph::find_clique(int nrbranches) const {

    //dense graph, or force using heuristic; or force using ostergard
    if( ( nrbranches <= 0) and ((nrbranches == -1) or ((_ncount * (_ncount - 1) / 2.0) / _ecount > 0.3) ) ){
        MWISls_env* env = (MWISls_env*) NULL;
        COLORset *cliques = (COLORset *) NULL;
        int ncliques = 0;
        std::vector<int> one_weights(_ncount, 1);
        int rval = COLORstable_LS(&env, &cliques, &ncliques, int(_ncount), (int((_ncount * (_ncount - 1))) / 2 - int(_ecount)),
                                  complement_shifted_elist().data(), one_weights.data(), COLORNWT_MIN);
        if(rval or not ncliques){
            std::cout << "Failed in COLORstable_LS." << std::endl;
            return {};
        }
        std::vector<Vertex> clique;
        std::transform(cliques[ncliques - 1].members, cliques[ncliques - 1].members + cliques[ncliques - 1].count, std::inserter(clique, clique.end()),
                       [](int v) { return (Vertex) v + 1; });
        return clique;
    }else{
        //find size of some (optimally maximal) clique
        COLORset *cliques = (COLORset *) NULL;
        int ncliques = 0;
        int pval = 0;
        if(nrbranches == 0)
            nrbranches = std::max(int(_ncount) / 10 + 10, 100);
        std::vector<int> one_weights(_ncount, 1);
        int rval = COLORclique_ostergard(&cliques, &ncliques, int(_ncount), int(_ecount),
                                         shifted_elist().data(), one_weights.data(), COLOR_MAXINT, &pval, nrbranches);
        if(rval or not ncliques) {
            std::cout << "Failed in COLORclique_ostergard." << std::endl;
            return {};
        }
        std::vector<Vertex> clique;
        std::transform(cliques[0].members, cliques[0].members + cliques[0].count, std::inserter(clique, clique.end()),
                       [](int v) { return (Vertex) v + 1; });
        return clique;
    }
    return {};
}







