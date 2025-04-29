#ifndef SIMPLICIAL_COMPLEX_CPP
#define SIMPLICIAL_COMPLEX_CPP

#include "simplicial_complex.h"
#include "vertex.h"
#include "simplex.h"
#include "simplex_collection.h"

#include "vertex.cpp"
#include "simplex.cpp"
#include "simplex_collection.cpp"

#include <set>
#include <map>
#include <queue>
#include <cassert>
#include <stdexcept>



SimplicialComplex::SimplicialComplex(const SimplexCollection &collection){
    if(collection.GetSimplices().empty()){
        throw std::runtime_error("(SimplicialComplex::SimplicialComplex) The provided collection is empty");
    }
    if(not collection.IsSimplicialComplex()){
        throw std::runtime_error("(SimplicialComplex::SimplicialComplex) The provided collection does not form a simplicial complex");
    }
    
    const std::set<Vertex> &set_v = collection.Vertices();
    label = std::vector<Vertex>(set_v.begin(), set_v.end());
    
    rank = static_cast<unsigned int>(collection.Rank());
    simplex_count.resize(rank+2);
    
    simplices.reserve(collection.GetSimplices().size());
    const SimplexCollection &rc = collection.Relabelled();
    for(const Simplex &s: rc.GetSimplices()){
        simplices.push_back(s);
        simplex_count[s.Rank()]++;
    }
    
    unsigned int sm = 0;
    for(unsigned int i=0; i<=rank+1; i++){
        unsigned int t = simplex_count[i];
        simplex_count[i] = sm;
        sm += t;
    }
    
    Precomputations();
}



void SimplicialComplex::Precomputations(){
    ComputeFacets();
    ComputeLinksAndStars();
}



void SimplicialComplex::ComputeFacets(){
    facet = std::vector<std::vector<unsigned int>>(simplex_count[rank+1]);
    for(unsigned int i=0; i<simplex_count[rank+1]; i++){
        const std::vector<Simplex> &fc = simplices[i].Facets();
        facet[i].reserve(simplices[i].Rank());
        for(const Simplex &s: fc){
            facet[i].push_back(GetSimplexId(s));
        }
    }
}



void SimplicialComplex::ComputeLinksAndStars(){
    link = std::vector<std::vector<unsigned int>>(simplex_count[rank+1]);
    open_star = std::vector<std::vector<unsigned int>>(simplex_count[rank+1]);
    for(unsigned int i=0; i<simplex_count[rank+1]; i++){
        const std::vector<Simplex> &z = simplices[i].Faces();
        for(const Simplex &s: z){
            link[GetSimplexId(s)].push_back(GetSimplexId(simplices[i].OppositeFace(s)));
            open_star[GetSimplexId(s)].push_back(i);
        }
    }
    for(unsigned int i=0; i<simplex_count[rank+1]; i++)
        std::sort(link[i].begin(), link[i].end());
}



std::string SimplicialComplex::ToString() const
{
    std::string a = "SimplicialComplex{";
    for(const Simplex &s: simplices){
        a += s.ToString();
        a += " ";
    }
    a.pop_back();
    a += "}";
    return a;
}



SimplexCollection SimplicialComplex::Collection() const
{
    return SimplexCollection(std::set<Simplex>(simplices.begin(), simplices.end()));
}



unsigned int SimplicialComplex::Rank() const
{
    return rank;
}



int SimplicialComplex::Dimension() const
{
    return static_cast<int>(rank) - 1;
}



int SimplicialComplex::NumVertices() const
{
    return simplex_count[2] - simplex_count[1];
}



// Returns the SimplexId corresponding to the given simplex s, if the latter is part of the complex;
// otherwise, returns an invalid SimplexId.
unsigned int SimplicialComplex::GetSimplexId(const Simplex &s) const
{
    if(s.Rank() > rank) return simplex_count[rank+1];
    unsigned int l = simplex_count[s.Rank()];
    unsigned int r = simplex_count[s.Rank()+1]-1;
    while(l < r){
        unsigned int m = (l + r) / 2;
        if(simplices[m] < s) l = m + 1;
        else r = m;
    }
    if(not(simplices[l] == s)) l = simplex_count[rank+1];
    return l;
}



bool SimplicialComplex::ValidSimplexId(const unsigned int &id) const
{
    return id < simplex_count[rank+1];
}



// Returns the link of a given simplex.
const std::vector<unsigned int>& SimplicialComplex::GetSimplexLink(const unsigned int &id) const
{
    return link[id];
}



// Given a simplex and one of its facets (the i-th one), gives the simplex sharing this facet, assuming it is unique.
unsigned int SimplicialComplex::OppositeFacet(const unsigned int &id, const unsigned int &i) const
{
    unsigned int wall = facet[id][i];
    if(open_star[wall].size() != 3) throw std::runtime_error("(SimplicialComplex::OppositeFacet) The link of the wall is not of the expected type");
    return open_star[wall][1] + open_star[wall][2] - id;
}



// Determines if f:S->T is a simplicial map.
bool IsSimplicial(const std::map<Vertex,Vertex> &f, const SimplicialComplex &S, const SimplicialComplex &T)
{
    std::map<Vertex,unsigned int> T_vert, S_vert;
    for(int i=0; i<S.NumVertices(); i++) S_vert[S.label.at(i)] = i;
    for(int i=0; i<T.NumVertices(); i++) T_vert[T.label.at(i)] = i;
    
    for(const auto &p: f) if(S_vert.count(p.first) == 0 or T_vert.count(p.second) == 0) return false;
    
    for(unsigned int i=1; i<S.simplex_count.at(S.rank+1); i++){
        const Simplex &s = S.simplices.at(i);
        std::set<Vertex> fs;
        for(Vertex v: s.GetVertices()) fs.insert(T_vert.at(f.at(S.label.at(v))));
        if(not T.ValidSimplexId(T.GetSimplexId(Simplex(fs)))) return false;
    }
    return true;
}



// Given a simplicial map between orientable strongly connected pseudomanifold of the same dimension, compute its degree modulo 2.
int DegreeMod2(const std::map<Vertex,Vertex> &f, const SimplicialComplex &S, const SimplicialComplex &T)
{
    if(S.rank != T.rank) throw std::runtime_error("(SimplicialComplex::DegreeMod2) The complexes have different dimensions");
    
    std::map<Vertex,unsigned int> T_vert, S_vert;
    for(int i=0; i<S.NumVertices(); i++) S_vert[S.label.at(i)] = i;
    for(int i=0; i<T.NumVertices(); i++) T_vert[T.label.at(i)] = i;
    
    for(const auto &p: f) if(S_vert.count(p.first) == 0 or T_vert.count(p.second) == 0)
        throw std::runtime_error("(SimplicialComplex::DegreeMod2) The map f has wrong domain or codomain");

    int deg = 0;
    const Simplex &t = T.simplices.at(T.simplex_count[T.rank]);
    for(unsigned int i=S.simplex_count.at(S.rank); i<S.simplex_count.at(S.rank+1); i++){
        const Simplex &s = S.simplices.at(i);
        std::vector<Vertex> fs;
        for(Vertex v: s.GetVertices()) fs.emplace_back(T_vert.at(f.at(S.label.at(v))));
        std::sort(fs.begin(), fs.end());
        if(fs == t.GetVertices()) deg++;
    }
    return deg%2;
}



// Given a simplex and an order on its vertices (as a permutation of {0,...,rank-1}),
// it assigns a number f[v] to every vertex v, 
// telling the order of "discovery" during a "visit" of the complex (starting from 0).
// Assumption: the simplicial complex is a strongly-connected pseudomanifold.
std::vector<int> SimplicialComplex::VisitOrder(const unsigned int &s_id, const std::vector<int> &s_order) const{
    std::vector<bool> visited(simplex_count[rank+1]-simplex_count[rank], false);
    std::vector<int> f(simplex_count[2]-simplex_count[1], -1);
    
    const Simplex &s = simplices[s_id];
    
    for(unsigned int i=0; i<rank; i++){
        f[s.GetVertices()[s_order[i]]] = i;
    }
    unsigned int counter = rank;
    
    visited[s_id - simplex_count[rank]] = true;
    
    std::queue<std::pair<unsigned int,unsigned int>> moves; // Simplex s, vertex to remove from s.
    for(unsigned int i=0; i<rank; i++) moves.emplace(s_id, s_order[i]);
    
    while(not moves.empty()){
        const auto [s_id, vs] = moves.front();
        moves.pop();
        
        unsigned int so_id = OppositeFacet(s_id, vs);
        if(visited[so_id - simplex_count[rank]]) continue;
        
        unsigned int vso = 0;
        while(facet[so_id][vso] != facet[s_id][vs]) vso++;
        
        const Vertex &vx = simplices[so_id].GetVertices()[vso];
        if(f[vx] < 0) f[vx] = counter++;
        
        visited[so_id - simplex_count[rank]] = true;
        
        std::vector<int> new_moves;
        for(unsigned int i=0; i<rank; i++) if(i != vso) new_moves.emplace_back(i);
        sort(new_moves.begin(), new_moves.end(), [&](int i, int j){
            return f[simplices[so_id].GetVertices()[i]] < f[simplices[so_id].GetVertices()[j]];
        });
        
        for(int u:new_moves) moves.emplace(so_id, u);
    }
    
    // The following check should be superflous.
    assert(counter == simplex_count[2] - simplex_count[1]);
    
    return f;
}



SimplicialComplex::Encoding::Encoding(std::vector<int> enc): enc(enc) {}



SimplicialComplex::Encoding::Encoding(): enc({}) {}



bool SimplicialComplex::Encoding::operator == (const Encoding &other) const{
    return enc == other.enc;
}



bool SimplicialComplex::Encoding::operator < (const Encoding &other) const{
    return enc < other.enc;
}



std::ostream& operator << (std::ostream &o, const SimplicialComplex::Encoding &E){
    o << "(";
    for(auto it=E.enc.begin(); it!=E.enc.end(); ){
        o << *it;
        it++;
        if(it != E.enc.end()) o << ",";
    }
    o << ")";
    return o;
}



std::istream& operator >> (std::istream &s, SimplicialComplex::Encoding &E){
    s >> E.enc;
    return s;
}



// Computes the encoding of the simplicial complex.
// Assumption: the complex is a flag strongly-connected pseudomanifold.
SimplicialComplex::Encoding SimplicialComplex::StandardEncoding() const{
    int n_ver = simplex_count[2]-simplex_count[1]; // Number of vertices.
    std::vector<int> current_best;
    for(unsigned int s_id = simplex_count[rank]; s_id < simplex_count[rank+1]; s_id++){
        std::vector<int> s_order(rank);
        for(unsigned int i=0; i<rank; i++) s_order[i] = i;
        do{
            auto f = VisitOrder(s_id, s_order);
            std::vector<int> attempt;
            for(unsigned int ei = simplex_count[2]; ei < simplex_count[3]; ei++){ // Iterate through 1-simplices.
                if(f[facet[ei][0]-1] < f[facet[ei][1]-1]){
                    attempt.emplace_back(n_ver * f[facet[ei][0]-1] + f[facet[ei][1]-1]);
                }else{
                    attempt.emplace_back(n_ver * f[facet[ei][1]-1] + f[facet[ei][0]-1]);
                }
            }
            std::sort(attempt.begin(), attempt.end());
            if(current_best.empty() or attempt < current_best) current_best = attempt;
        }while(next_permutation(s_order.begin(), s_order.end()));
    }
    
    std::vector<int> vec;
    vec.emplace_back(n_ver);
    vec.insert(vec.end(), current_best.begin(), current_best.end());
    
    return Encoding(vec);
}



// The 1-skeleton of a triangulation whith the given encoding.
Graph SimplicialComplex::Encoding::OneSkeleton() const
{
    Graph G;
    
    int n_ver = enc.at(0);
    for(unsigned int i=1; i<enc.size(); i++){
        int u = enc[i] % n_ver;
        int v = enc[i] / n_ver;
        G.Connect(u, v);
    }
    
    return G;
};



// The number of vertices of the corresponding complex.
int SimplicialComplex::Encoding::NumVertices() const
{
    return enc.at(0);
}



SimplicialComplex::IsomorphismFinder::IsomorphismFinder(const SimplicialComplex &X, const SimplicialComplex &Y):
    X(X),
    Y(Y),
    rank(X.Rank()),
    computed(false),
    isomorphic(false),
    f(0)
{}



void SimplicialComplex::IsomorphismFinder::BasicChecks(){
    bool ok = true;
    if(Y.Rank() != rank) ok = false;
    else{
        for(unsigned int i=0; i<=rank+1; i++)
            if(X.simplex_count[i] != Y.simplex_count[i]) ok = false;
    }
    if(not ok){
        isomorphic = false;
        computed = true;
    }
}



bool SimplicialComplex::IsomorphismFinder::AttemptFromFacets(const unsigned int &sX_id, const unsigned int &sY_id, const std::vector<Vertex> &sY_order){
    std::vector<bool> visited(X.simplex_count[rank+1]-X.simplex_count[rank], false);
    f = std::vector<Vertex>(X.simplex_count[2]-X.simplex_count[1], -1);
    
    const Simplex &sX = X.simplices[sX_id];
    
    for(unsigned int i=0; i<rank; i++){
        f[sX.GetVertices()[i]] = sY_order[i];
    }
    
    visited[sX_id - X.simplex_count[rank]] = true;
    
    std::queue<std::tuple<unsigned int,unsigned int,unsigned int>> moves; // Simplex s in X, f(s), vertex to remove from s.
    for(unsigned int i=0; i<rank; i++) moves.emplace(sX_id, sY_id, i);
    
    while(not moves.empty()){
        const auto [s_id, t_id, vs] = moves.front();
        moves.pop();
        
        unsigned int so_id = X.OppositeFacet(s_id, vs);
        if(visited[so_id - X.simplex_count[rank]]) continue;
        
        unsigned int vt = 0;
        while(f[X.simplices[s_id].GetVertices()[vs]] != Y.simplices[t_id].GetVertices()[vt]) vt++;
        
        unsigned int to_id = Y.OppositeFacet(t_id, vt);
        
        unsigned int vso = 0;
        while(X.facet[so_id][vso] != X.facet[s_id][vs]) vso++;
        unsigned int vto = 0;
        while(Y.facet[to_id][vto] != Y.facet[t_id][vt]) vto++;
        
        const Vertex &vx = X.simplices[so_id].GetVertices()[vso];
        const Vertex &vy = Y.simplices[to_id].GetVertices()[vto];
        if(f[vx] >= 0 and f[vx] != vy) return false;
        if(f[vx] < 0) f[vx] = vy;
        
        visited[so_id - X.simplex_count[rank]] = true;
        for(unsigned int i=0; i<rank; i++) if(i != vso){
            moves.emplace(so_id, to_id, i);
        }
    }
    
    std::vector<bool> in_image(Y.simplex_count[2]-Y.simplex_count[1], false);
    for(unsigned int i=0; i<Y.simplex_count[2]-Y.simplex_count[1]; i++){
        assert(f[i] >= 0);
        in_image[f[i]] = true;
    }
    for(unsigned int i=0; i<Y.simplex_count[2]-Y.simplex_count[1]; i++)
        if(not in_image[f[i]]) return false;
    
    // The following check should be superflous.
    std::map<Vertex,Vertex> mp;
    for(unsigned int i=0; i<X.simplex_count[2]-X.simplex_count[1]; i++) mp[i] = f[i];
    for(const Simplex &s: X.simplices){
        Simplex fs = s.Relabel(mp);
        assert(Y.ValidSimplexId(Y.GetSimplexId(fs)));
    }
    
    return true;
}



void SimplicialComplex::IsomorphismFinder::Compute(){
    computed = false;
    BasicChecks();
    if(computed) return;
    
    unsigned int sX = X.simplex_count[rank];
    for(unsigned int sY = Y.simplex_count[rank]; sY < Y.simplex_count[rank+1]; sY++){
        std::vector<Vertex> v = Y.simplices[sY].GetVertices();
        do{
            if(AttemptFromFacets(sX,sY,v)){
                isomorphic = true;
                computed = true;
                return;
            }
        }while(next_permutation(v.begin(), v.end()));
    }
    
    isomorphic = false;
    computed = true;
}



bool SimplicialComplex::IsomorphismFinder::Isomorphic(){
    if(not computed) Compute();
    return isomorphic;
}



std::map<Vertex,Vertex> SimplicialComplex::IsomorphismFinder::Isomorphism(){
    if(not Isomorphic()) throw std::runtime_error("(SimplicialComplex::IsomorphismFinder) The complexes are not isomorphic");
    std::map<Vertex,Vertex> mp;
    for(unsigned int i=0; i<f.size(); i++) mp[X.label[i]] = Y.label[f[i]];
    return mp;
}



SimplicialComplex::NZMapFinder::NZMapFinder(const SimplicialComplex &X, const SimplicialComplex &Y):
    X(X),
    Y(Y),
    rank(X.Rank()),
    computed(false),
    found(false),
    f(0)
{}



void SimplicialComplex::NZMapFinder::BasicChecks(){
    bool ok = true;
    if(Y.Rank() != rank) ok = false;
    else{
        for(unsigned int i=0; i<=rank+1; i++)
            if(X.simplex_count[i] < Y.simplex_count[i]) ok = false;
    }
    if(not ok){
        found = false;
        computed = true;
    }
}



void SimplicialComplex::NZMapFinder::Compute(){
    computed = false;
    BasicChecks();
    if(computed) return;
    found = false;
    f = std::vector<Vertex>(X.NumVertices(), -1);
    std::vector<std::set<Vertex>> candidates(X.NumVertices());
    std::set<Vertex> VY;
    for(int v = 0; v < Y.NumVertices(); v++) VY.insert(v);
    for(int u = 0; u < X.NumVertices(); u++) candidates[u] = VY;
    Compute(candidates);
    computed = true;
}



void SimplicialComplex::NZMapFinder::Compute(std::vector<std::set<Vertex>> candidates){
    unsigned int ncand = Y.NumVertices() + 1;
    int U = -1;
    for(int u = 0; u < X.NumVertices(); u++) if(f[u] < 0){
        if(candidates[u].size() < ncand){
            ncand = candidates[u].size();
            U = u;
        }
    }
    
    if(U < 0){
        if(DegreeMod2() != 0){
            found = true;
        }
        return;
    }
    
    if(candidates[U].empty()) return;
    
    for(Vertex V: candidates[U]){
        f[U] = V;
        
        bool valid = true;
        for(int os: X.open_star[U+1]){
            const Simplex &s = X.simplices[os];
            bool complete = true;
            std::set<Vertex> t;
            for(Vertex x: s.GetVertices()) if(f[x] < 0) {complete = false; break;}
                             else t.insert(f[x]);
            if(complete){
                int iy = Y.GetSimplexId(Simplex(t));
                if(not Y.ValidSimplexId(iy)){valid = false; break;}
            }
        }
        if(not valid){
            f[U] = -1;
            continue;
        }
        
        
        std::set<Vertex> NV= {V};
        for(unsigned int os: Y.link[V+1]){
            if(os >= Y.simplex_count[2]) break;
            if(os == 0) continue;
            NV.insert(os - 1);
        }
        
        std::vector<std::pair<Vertex,Vertex>> removed_candidates;
        for(unsigned int os: X.link[U+1]){
            if(os >= X.simplex_count[2]) break;
            if(os == 0) continue;
            Vertex w = os - 1;
            if(f[w] >= 0) continue;
            for(Vertex z: candidates[w]) if(NV.count(z) == 0){
                removed_candidates.emplace_back(w, z);
            }
        }
        for(const auto &p: removed_candidates){
            candidates[p.first].erase(p.second);
        }
        
        Compute(candidates);
        if(found) return;
        
        for(const auto &p: removed_candidates){
            candidates[p.first].insert(p.second);
        }
        f[U] = -1;
    }
}



int SimplicialComplex::NZMapFinder::DegreeMod2(){
    int deg = 0;
    const Simplex &t = Y.simplices[Y.simplex_count[rank]];
    for(unsigned int i=X.simplex_count[rank]; i<X.simplex_count[rank+1]; i++){
        const Simplex &s = X.simplices[i];
        std::vector<Vertex> fs;
        for(Vertex v: s.GetVertices()) fs.emplace_back(f[v]);
        std::sort(fs.begin(), fs.end());
        if(fs == t.GetVertices()) deg++;
    }
    return deg%2;
}



bool SimplicialComplex::NZMapFinder::Exists(){
    if(not computed) Compute();
    return found;
}



std::map<Vertex,Vertex> SimplicialComplex::NZMapFinder::NZMap(){
    if(not Exists()) throw std::runtime_error("(SimplicialComplex::NZMapFinder) There is no map of nonzero degree");
    std::map<Vertex,Vertex> mp;
    for(unsigned int i=0; i<f.size(); i++) mp[X.label[i]] = Y.label[f[i]];
    return mp;
}



#endif
