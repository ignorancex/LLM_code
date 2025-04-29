#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <bitset>
#include <cassert>
#include <random>

#include "vertex.cpp"
#include "simplicial_complex.cpp"
#include "graph.cpp"
#include "localpicture.cpp"
#include "localpicture_homfinder.cpp"



const std::string localpictures_folder = "localpictures/";
const std::array<std::string, 3> localpictures_files{
    "univ_5_1_T10.txt",
    "univ_5_1_T12_B.txt",
    "univ_5_1_T12_A.txt",
};
constexpr int n_localpicture = localpictures_files.size();



// Device for pseudo-random number generation
std::mt19937 random_box;
void PrepareRandomBox(int seed)
{
    std::clog << "(PrepareRandomBox) seed = " << seed << '\n';
    random_box = std::mt19937(seed);
}
void PrepareRandomBox()
{
    std::random_device rd;
    int seed = rd();
    PrepareRandomBox(seed);
}



// The following declarations and methods allow to store/read a history
// of analyzed triangulations.
// The history structure stores two different things:
// - A set of pairwise-nonisomorphic triangulations (actually, of their 
//   "encodings"; two encodings are equal if and only if their corresponding
//   triangulations are isomorphic);
// - For each triangulation, an Information object that stores an identificative
//   number of the triangulation (its ID), whether or not the triangulation has
//   been analyzed, and the result of the analysis, which is a bitset telling
//   for each of the "target localpictures" whether or not the triangulation
//   has a "special" map with image that localpicture.
// - A vector of "events" telling what triangulation has been constructed at
//   each iteration of the algorithm.
// 
// Use the following functions to populate an history object:
// - SetId(Enc, ID): associate the id ID to the encoding Enc;
// - AddEvent(ID): add an event corresponding to "visiting" the triangulation with id ID;
// - UpdateInfo(ID, Dom): set the "domination bitset" associated to id ID.
// They return a Handle object; which is a reference to a pair (Enc, Info), where
// Enc is an encoding and Info is the associated information.
// Also the following versions can be used:
// - AddEvent(Enc, ID): equivalent to SetID(Enc, ID) + AddEvent(ID);
// - AddEvent(Enc): automatically associate a new id and add the event;
// - UpdateInfo(Handle, Dom);
// - UpdateInfo(Information&, Dom), this doesn't return a Handle.
// The assignment of an ID is always successful, even if it was previously 
// assigned to another triangulation; in this case the other triangulation "loses"
// the id.
struct Information
{
    int id;
    bool analyzed;
    std::bitset<n_localpicture> dominance;
    Information(): analyzed(false) {}
};
struct History
{
    typedef std::map<SimplicialComplex::Encoding, Information> Imap;
    typedef Imap::iterator Handle;
    
    Imap t_info; // Encoding -> Information
    std::map<int, Handle> i_data; // ID -> Corresponding encoding and associated information.
    std::vector<Handle> event; // The list of events.
    
    std::set<int> temporary_ids;
    int next_proposed_id = 1;
    int TakeNextId(){
        while(i_data.count(next_proposed_id) or temporary_ids.count(next_proposed_id)) next_proposed_id++;
        next_proposed_id++;
        return next_proposed_id - 1;
    }
    
    Handle AddEvent(const SimplicialComplex::Encoding &E, int id = 0)
    {
        const auto [it, novel] = t_info.emplace(E, Information{});
        
        if(novel){
            if(id == 0) id = TakeNextId();
            if(i_data.count(id)){
                int other_id = TakeNextId();
                i_data[other_id] = i_data.at(id);
                i_data.at(other_id)->second.id = other_id;
                temporary_ids.insert(other_id);
            }
            
            i_data[id] = it;
            event.emplace_back(it);
            it->second.id = id;
        }else{
            if(id == 0) id = it->second.id;
            else if(id != it->second.id){
                if(i_data.count(id)){
                    int other_id = TakeNextId();
                    i_data[other_id] = i_data.at(id);
                    i_data.at(other_id)->second.id = other_id;
                    temporary_ids.insert(other_id);
                }
                
                i_data.erase(it->second.id);
                it->second.id = id;
                i_data[id] = it;
            }
            event.emplace_back(it);
        }
        
        if(temporary_ids.count(id)){
            temporary_ids.erase(id);
        }
        
        return it;
    }
    
    Handle AddEvent(int id){
        auto it = i_data.at(id);
        event.emplace_back(it);
        if(temporary_ids.count(id)){
            throw std::runtime_error("(History::AddEvent) Called with a temporary id");
            temporary_ids.erase(id);
        }
        return it;
    }
    
    void UpdateInfo(Information& info, const std::bitset<n_localpicture> &dominance)
    {
        info.analyzed = true;
        info.dominance = dominance;
        if(temporary_ids.count(info.id)){
            temporary_ids.erase(info.id);
        }
    }
    
    Handle UpdateInfo(const Imap::iterator &it, const std::bitset<n_localpicture> &dominance)
    {
        UpdateInfo(it->second, dominance);
        return it;
    }
    
    Handle UpdateInfo(const SimplicialComplex::Encoding &E, const std::bitset<n_localpicture> &dominance)
    {
        return UpdateInfo(t_info.find(E), dominance);
    }
    
    Handle UpdateInfo(int id, const std::bitset<n_localpicture> &dominance)
    {
        if(temporary_ids.count(id)){
            throw std::runtime_error("(History::UpdateInfo) Called with a temporary id");
            temporary_ids.erase(id);
        }
        return UpdateInfo(i_data.at(id), dominance);
    }
    
    Handle SetId(const SimplicialComplex::Encoding &E, int id)
    {
        if(id == 0) throw std::runtime_error("(History::SetId) id cannot be 0");
        const auto [it, novel] = t_info.emplace(E, Information{});
        
        int prev_id = 0;
        if(not novel) prev_id = it->second.id;
        
        if(prev_id != id and i_data.count(id)){
            int other_id = TakeNextId();
            i_data[other_id] = i_data.at(id);
            i_data.at(other_id)->second.id = other_id;
            temporary_ids.insert(other_id);
        }
        
        i_data[id] = it;
        it->second.id = id;
        return it;
    }
};



void ReadLogLine(std::istream &in, char line_type, History &hist)
{
    int id;
    std::bitset<n_localpicture> dominance;
    SimplicialComplex::Encoding E;
        
    switch(line_type){
        case 'A': // Associate ID.
            in >> id >> E;
            hist.SetId(E, id);
            break;
        case 'B': // Add event.
            in >> id;
            hist.AddEvent(id);
            break;
        case 'C': // Update analysis.
            in >> dominance >> id;
            hist.UpdateInfo(id, dominance);
            break;
        case 'D': // Associate ID and add event.
            in >> id >> E;
            hist.AddEvent(E, id);
            break;
        case 'E': // Add event and update analysis.
            in >> dominance >> id;
            hist.AddEvent(id);
            hist.UpdateInfo(id, dominance);
            break;
        case 'F': // Associate ID and update analysis.
            in >> dominance >> id >> E;
            hist.SetId(E, id);
            hist.UpdateInfo(id, dominance);
            break;
        case 'G': // Associate ID, update analysis and add event.
            in >> dominance >> id >> E;
            hist.AddEvent(E, id);
            hist.UpdateInfo(id, dominance);
            break;
        default: throw std::runtime_error("Error reading log line");
    }
}



void ReadLogFile(History &hist, std::string s)
{
    std::ifstream fin(s);
    if(fin.fail()) throw std::runtime_error("File " + s + " has some problems");
    
    std::clog << "Reading logfile " << s << '\n';
    DiscardComments(fin);
    char line_type;
    while(fin >> line_type){
        ReadLogLine(fin, line_type, hist);
        DiscardComments(fin);
    }
}



// Read a graph from a file.
// If the same file has been already used to read a graph, the file is not 
// reopened, because already read graphs are kept in memory.
Graph GraphFromFile(std::string filename)
{
    static std::map<std::string, Graph> memo;
    if(memo.count(filename)) return memo.at(filename);
    
    std::ifstream fin(filename);
    if(fin.fail()) throw std::runtime_error("(GraphFromFile) File " + filename + " has some problems");
    
    std::clog << "(GraphFromFile) Reading from " << filename << '\n';
    
    Graph G;
    DiscardComments(fin);
    fin >> G;
    
    memo.emplace(filename, G);
    return memo.at(filename);
}



// A struct containing a graph, two vertices forming an edge in it, and the 
// LocalPicture constructed from them.
struct GraphLocalPicture
{
    const Graph graph;
    const LocalPicture locpic;
    const Vertex u, v;
    
    std::map<std::string, std::map<Vertex,Vertex>> reductions;
    
    GraphLocalPicture(const Graph &G, const Vertex &u, const Vertex &v):
        graph(G),
        locpic(G, u, v),
        u(u),
        v(v),
        reductions()
    {}
};



// Extracts a GraphLocalPicture from a file containing a graph and two vertices
// forming an edge.
GraphLocalPicture LocalPictureFromFile(std::string filename)
{
    static std::map<std::string, GraphLocalPicture> memo;
    if(memo.count(filename)) return memo.at(filename);
    
    std::ifstream fin(filename);
    if(fin.fail()) throw std::runtime_error("(LocalPictureFromFile) File " + filename + " has some problems");
    
    std::clog << "(LocalPictureFromFile) Reading from " << filename << '\n';
    
    Graph G;
    DiscardComments(fin);
    fin >> G;
    
    Vertex x, y;
    DiscardComments(fin);
    fin >> x >> y;
    assert(x != y);
    
    GraphLocalPicture PG(G, x, y);
    DiscardComments(fin);
    fin >> PG.reductions;
    
    memo.emplace(filename, PG);
    return memo.at(filename);
}



// Checks if there is a map between local pictures coming from two graphs,
// specifying whether there should be a reflection and/or inversion.
// The target local picture (B) is assumed to be defined around an almost-omniscent edge.
// If there is a map, a map between the GRAPHS is returned.
// Otherwise, an empty map is returned.
std::map<Vertex,Vertex> SpecialHomomorphism(const GraphLocalPicture &A, const GraphLocalPicture &B, bool reflect, bool invert)
{
    LocalPicture::HomFinder F(A.locpic, B.locpic, reflect, invert);
    if(F.HomExists()){
        auto FF = F.Homomorphism();
        if(reflect){
            FF[A.u] = B.v;
            FF[A.v] = B.u;
        }else{
            FF[A.u] = B.u;
            FF[A.v] = B.v;
        }
        
        Vertex out = -1;
        for(const Vertex &x: B.graph.Vertices()) if(x != B.u and x != B.v and not B.locpic.IsVertex(x)){
            assert(out == -1);
            out = x;
        }
        for(const Vertex &x: A.graph.Vertices()) if(x != A.u and x != A.v and not A.locpic.IsVertex(x)){
            assert(out >= 0);
            FF[x] = out;
        }
        
        for(auto &p: FF) assert(A.graph.IsVertex(p.first));
        if(not Graph::IsHomomorphism(A.graph, B.graph, FF))
            throw std::runtime_error("(SpecialHomomorphism) The map does not respect the graph structure");
        return FF;
    }
    return {};
}



// Checks if there is a map between local pictures coming from two graphs,
// trying all reflections/inversions.
// The target local picture (B) is assumed to be defined around an almost-omniscent edge.
// If there is a map, a map between the GRAPHS is returned.
// Otherwise, an empty map is returned.
std::map<Vertex,Vertex> SpecialHomomorphism(const GraphLocalPicture &A, const GraphLocalPicture &B)
{
    for(int k=0; k<2; k++){ //invert A?
        for(int j=0; j<2; j++){ //reflect A?
            auto FF = SpecialHomomorphism(A, B, j > 0, k > 0);
            if(not FF.empty()) return FF;
        }
    }
    return {};
}



// Checks if there is a map between local pictures coming from two graphs,
// trying the specified reflections/inversions.
// The target local picture (B) is assumed to be defined around an almost-omniscent edge.
// If there is a map, a map between the GRAPHS is returned.
// Otherwise, an empty map is returned.
std::map<Vertex,Vertex> SpecialHomomorphism(const GraphLocalPicture &A, const GraphLocalPicture &B, const std::vector<std::bitset<2>> &rip)
{
    for(const auto &p: rip){
        auto FF = SpecialHomomorphism(A, B, p.test(0), p.test(1));
        if(not FF.empty()) return FF;
    }
    return {};
}



// Check if the clique complex of G dominates one of the target
// triangulations via a homomorphism of local pictures.
// Assumption: G is the one-skeleton of a flag triangulation of S^3.
std::bitset<n_localpicture> Check(const Graph &G){
    static std::vector<std::vector<std::bitset<2>>> reps;
    if(reps.empty()){
        std::clog << "(Check) Computing reps\n";
        for(int i=0; i<n_localpicture; i++){
            reps.emplace_back(LocalPictureFromFile(localpictures_folder + localpictures_files[i]).locpic.ReflectInvertRepresentatives());
        }
    }
    
    const int gamma2 = 16 - 5 * G.NumVertices() + G.NumEdges();
    std::bitset<n_localpicture> result{};
    if(gamma2 == 0) assert(G.NumVertices() == 8);
    if(gamma2 > 0){
        for(int i=0; i<n_localpicture; i++){
            const GraphLocalPicture TP = LocalPictureFromFile(localpictures_folder + localpictures_files[i]);
            
            bool dominates_s = false;
            for(auto e: G.Edges()){
                const GraphLocalPicture PG(G, e.first, e.second);
                const auto f = SpecialHomomorphism(PG, TP, reps[i]);
                if(not f.empty()){
                    dominates_s = true;
                    break;
                }
            }
            if(dominates_s){
                result[i] = true;
            }
        }
    }
    return result;
}



// Check if S dominates one of the target
// triangulations via a homomorphism of local pictures.
// Assumption: G is the one-skeleton of S, which is a flag triangulation of S^3.
// This version of the function performs additional checks.
std::bitset<n_localpicture> Check(const Graph& G, const SimplicialComplex& S){
    static std::vector<SimplicialComplex> tloc;
    if(tloc.empty()){
        std::clog << "(Check) Computing tloc\n";
        for(int i=0; i<n_localpicture; i++){
            const GraphLocalPicture &TP = LocalPictureFromFile(localpictures_folder + localpictures_files[i]);
            tloc.emplace_back(TP.graph.CliqueCollection());
        }
    }
    
    static std::vector<std::vector<std::bitset<2>>> reps;
    if(reps.empty()){
        std::clog << "(Check) Computing reps\n";
        for(int i=0; i<n_localpicture; i++){
            reps.emplace_back(LocalPictureFromFile(localpictures_folder + localpictures_files[i]).locpic.ReflectInvertRepresentatives());
        }
    }
    
    const int gamma2 = 16 - 5 * G.NumVertices() + G.NumEdges();
    std::bitset<n_localpicture> result{};
    if(gamma2 == 0) assert(G.NumVertices() == 8);
    if(gamma2 > 0){
        for(int i=0; i<n_localpicture; i++){
            const GraphLocalPicture &TP = LocalPictureFromFile(localpictures_folder + localpictures_files[i]);
            
            bool dominates_s = false;
            for(auto e: G.Edges()){
                const GraphLocalPicture PG(G, e.first, e.second);
                const auto hom = SpecialHomomorphism(PG, TP, reps[i]);
                if(not hom.empty()){
                    if(not IsSimplicial(hom, S, tloc[i])){
                        std::cout << "# NONSIMPLICIAL MAP:\n" << hom << '\n' << G << '\n' << TP.graph << '\n' << std::endl;
                        throw std::runtime_error("(Check) The map is not simplicial");
                    }else if(DegreeMod2(hom, S, tloc[i]) == 0){
                        std::cout << "# ZERO DEGREE (MOD 2):\n" << hom << '\n' << G << '\n' << TP.graph << '\n' << std::endl;
                        throw std::runtime_error("(Check) The map has degree == 0 mod 2");
                    }
                    dominates_s = true;
                    break;
                }
            }
            if(dominates_s){
                result[i] = true;
            }
        }
    }
    return result;
}



// Changes the graph G by collapsing edges not contained in squares,
// until every edge is contained in a square.
// Selects randomly the edges to collapse.
void CollapseToSquares(Graph &G)
{
    auto e = G.Edges();
    std::vector<std::pair<Vertex,Vertex>> es;
    
    for(auto p: e) if(not G.InSquare(p.first, p.second)) es.emplace_back(p.first, p.second);
    
    if(es.size() == 0) return;
    
    std::uniform_int_distribution<> distrib(0, es.size()-1);
    int u = distrib(random_box);
    
    G.IdentifyVertices(es[u].first, es[u].second);
    CollapseToSquares(G);
}



// Changes a graph G by performing, in order:
// - sub edge-subdivisions;
// - col edge-collapses (of edges not contained in squares).
// The edges on which to apply these operations are chosen randomly.
// If bl == true (stands for BigLinks) the edge to collapse is (at every step)
// chosen among those having the biggest link size.
// If at some point all edges are contained in squares, the procedure ends
// (without errors) even if the number of performed collapses is less than col.
void NextRandomTriangulation(Graph &G, int sub, int col, bool bl)
{
    for(int j=0; j<sub; j++){ //Perform edge subdivisions
        auto e = G.Edges();
        std::uniform_int_distribution<> distrib(0, e.size()-1);
        int u = distrib(random_box);
        G.SubdivideEdge(e[u].first, e[u].second);
    }
    for(int j=0; j<col; j++){ //Perform edge collapses
        auto e = G.Edges();
        int cn = 0;
        std::vector<std::pair<Vertex,Vertex>> es;
        for(auto p: e) if(not G.InSquare(p.first, p.second)){
            int ncn = G.CommonNeighbors({p.first,p.second}).size();
            if(bl and ncn > cn){
                cn = ncn;
                es.clear();
            }
            if(ncn >= cn or not bl) es.emplace_back(p.first, p.second);
        }
        if(es.empty()) break;
        std::uniform_int_distribution<> distrib(0, es.size()-1);
        int u = distrib(random_box);
        G.IdentifyVertices(es[u].first, es[u].second);
    }
}



// Process the triangulation given by the clique complex of G.
// Assumption: G is the one-skeleton of a flag triangulation of S^3.
// This adds an event in the history and prints the corresponding 
// instructions on the stdout stream.
void ProcessTriangulation(const Graph &G, History &hist){
    const SimplicialComplex S(G.CliqueCollection());
    const SimplicialComplex::Encoding E = S.StandardEncoding();
    
    History::Handle z = hist.AddEvent(E);
    if(not z->second.analyzed){
        auto res = Check(G, S);
        hist.UpdateInfo(z, res);
        std::cout << "G " << res << " " << z->second.id << " " << E << std::endl;
    }else{
        std::cout << "B " << z->second.id << std::endl;
    }
}



// This is the procedure that implements a "random walk" on the space of 
// flag triangulations of S^3.
void Explore(Graph G, int iterations, History &hist)
{
    int sub;
    int col;
    bool bl = false;
    int num_v = G.NumVertices();
    
    int iteration_mod = 0;
    for(int i=0; i<iterations; i++){
        sub = 30; // How many subdivisions in one iteration of the process.
        col = sub-3; // How many collapses in one iteration of the process.
        if(num_v > 12) col += 3;
        if(num_v > 30) col += 2;
        col++;
        
        if(i > 0) NextRandomTriangulation(G, sub, col, bl);
        
        Graph H = G;
        CollapseToSquares(H);
        // int gamma2 = 16 - 5 * H.NumVertices() + H.NumEdges();
        num_v = H.NumVertices();
        
        if(num_v > 25) bl = true; // Decide whether to use the "Big Links" strategy in the next iteration.
        if(num_v <= 15) bl = false;
                
        ProcessTriangulation(H, hist);
        
        iteration_mod++;
        if(iteration_mod == 100){
            iteration_mod = 0;
            std::clog << "(Explore) " << i+1 << " iterations, " << hist.t_info.size() << " triangulations\n";
        }
    }
    
    int count_tot = hist.t_info.size();
    std::clog << "(Explore) Checked " << count_tot << " pairwise-nonisomorphic triangulations" << std::endl;
}



int main(){
    PrepareRandomBox(1081506); // Put any seed for pseudo-random number generation.
    
    
    History hist;
    // Possibly, read a history of previously found triangulations:
    // ReadLogFile(hist, "log1.txt");
    // ReadLogFile(hist, "log2.txt");
    
    Graph G = Graph::Circle(4).Join(Graph::Circle(4));
    Explore(G, 1000, hist);
    
    std::clog << "##########################" << '\n';
    std::clog << "# Number of triangulations: " << hist.t_info.size() << '\n';
    
    std::map<int,int> q_vert;
    std::map<int, int> q_dom;
    
    for(const auto &p: hist.t_info){
        assert(p.second.analyzed);
        q_vert[p.first.NumVertices()]++;
        q_dom[p.second.dominance.to_ulong()]++;
    }
    
    std::clog << "\n# Vert\t# Triang\n";
    for(auto p:q_vert) std::clog << "# " << p.first << '\t' << p.second << '\n';
    
    std::clog << "\n# DOM\t# Triang\n";
    for(auto p:q_dom) std::clog << "# " << std::bitset<3>(p.first) << '\t' << p.second << '\n';
    
    return 0;
}
