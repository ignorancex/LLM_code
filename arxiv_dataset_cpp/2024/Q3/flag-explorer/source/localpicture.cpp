#ifndef LOCALPICTURE_CPP
#define LOCALPICTURE_CPP

#include "localpicture.h"
#include "vertex.h"
#include "graph.h"

#include "vertex.cpp"
#include "graph.cpp"

#include <stdexcept>
#include <set>
#include <map>
#include <algorithm>
#include <queue>
#include <cassert>



bool LocalPicture::AdjacentIdx(int x, int y) const
{
    if(x == y) return true;
    int l = 0;
    int r = graph.at(x).size() - 1;
    while(l < r){
        int m = (l + r) / 2;
        if(graph.at(x).at(m) < y) l = m + 1;
        else r = m;
    }
    return (graph.at(x).at(l) == y);
}



bool LocalPicture::Adjacent(const Vertex &v, const Vertex &w) const
{
    return idx.count(v) and idx.count(w) and AdjacentIdx(idx.at(v), idx.at(w));
}



int LocalPicture::NorthSize() const
{
    if(not reflected) return north_size;
    else return south_size;
}



int LocalPicture::SouthSize() const
{
    if(reflected) return north_size;
    else return south_size;
}



int LocalPicture::IthNorthIdx(int i) const
{
    if(not reflected) return equator_length + i;
    else return equator_length + north_size + i;
}



int LocalPicture::IthSouthIdx(int i) const
{
    if(reflected) return equator_length + i;
    else return equator_length + north_size + i;
}



Vertex LocalPicture::IthNorth(int i) const
{
    if(i < 0 or i >= NorthSize()) throw std::runtime_error("(LocalPicture::IthNorth) Invalid parameter");
    return labels.at(IthNorthIdx(i));
}



Vertex LocalPicture::IthSouth(int i) const
{
    if(i < 0 or i >= SouthSize()) throw std::runtime_error("(LocalPicture::IthSouth) Invalid parameter");
    return labels.at(IthSouthIdx(i));
}



Vertex LocalPicture::IthEquator(int i) const
{
    if(i < 0 or i > EquatorLength()) throw std::runtime_error("(LocalPicture::IthEquator) Invalid parameter");
    return labels.at(i);
}



Vertex LocalPicture::IthVertex(int i) const
{
    if(i < 0 or i > NumVertices()) throw std::runtime_error("(LocalPicture::IthVertex) Invalid parameter");
    return labels.at(i);
}



bool LocalPicture::IsNorthIdx(int i) const
{
    int l = equator_length;
    int r = equator_length + north_size;
    if(reflected){
        l += north_size;
        r += south_size;
    }
    return (i >= l and i < r);
}



bool LocalPicture::IsSouthIdx(int i) const
{
    int l = equator_length;
    int r = equator_length + north_size;
    if(not reflected){
        l += north_size;
        r += south_size;
    }
    return (i >= l and i < r);
}



bool LocalPicture::IsEquatorIdx(int i) const
{
    return (i >= 0 and i < equator_length);
}



bool LocalPicture::IsVertex(const Vertex &v) const
{
    return idx.count(v);
}



bool LocalPicture::IsNorth(const Vertex &v) const
{
    return idx.count(v) and IsNorthIdx(idx.at(v));
}



bool LocalPicture::IsSouth(const Vertex &v) const
{
    return idx.count(v) and IsSouthIdx(idx.at(v));
}



bool LocalPicture::IsEquator(const Vertex &v) const
{
    return idx.count(v) and IsEquatorIdx(idx.at(v));
}



bool LocalPicture::IsMarked(const Vertex &v) const
{
    return idx.count(v) and marked.at(idx.at(v));
}



int LocalPicture::EquatorSuccessorIdx(int i) const
{
    if(not inverted){
        i = i + 1;
        if(i >= equator_length) i = 0;
    }else{
        i = i - 1;
        if(i < 0) i = equator_length - 1;
    }
    return i;
}



int LocalPicture::EquatorPredecessorIdx(int i) const
{
    if(inverted){
        i = i + 1;
        if(i >= equator_length) i = 0;
    }else{
        i = i - 1;
        if(i < 0) i = equator_length - 1;
    }
    return i;
}



Vertex LocalPicture::EquatorSuccessor(const Vertex &v) const
{
    if(not idx.count(v)) throw std::runtime_error("(LocalPicture::EquatorSuccessor) Invalid vertex");
    return labels.at(EquatorSuccessorIdx(idx.at(v)));
}



Vertex LocalPicture::EquatorPredecessor(const Vertex &v) const
{
    if(not idx.count(v)) throw std::runtime_error("(LocalPicture::EquatorPredecessor) Invalid vertex");
    return labels.at(EquatorPredecessorIdx(idx.at(v)));
}



void LocalPicture::Init(const Graph &G,
               const std::vector<Vertex> &equator,
               const std::set<Vertex> &north,
               const std::set<Vertex> &south,
               const std::set<Vertex> &marked_vertices)
{
    int next_v = 0;
    for(const Vertex &x: equator){
        if(not G.IsVertex(x)) throw std::runtime_error("(LocalPicture::Init) Vertex " + std::to_string(x) + " is not in the graph");
        if(idx.count(x)) throw std::runtime_error("(LocalPicture::Init) Repeated vertices");
        idx[x] = next_v++;
        labels.push_back(x);
        equator_length++;
    }
    for(const Vertex &x: north){
        if(not G.IsVertex(x)) throw std::runtime_error("(LocalPicture::Init) Vertex " + std::to_string(x) + " is not in the graph");
        if(idx.count(x)){
            if(idx[x] > equator_length)
                throw std::runtime_error("(LocalPicture::Init) Repeated vertices");
        }else{
            idx[x] = next_v++;
            labels.push_back(x);
            north_size++;
        }
    }
    for(const Vertex &x: south){
        if(not G.IsVertex(x)) throw std::runtime_error("(LocalPicture::Init) Vertex " + std::to_string(x) + " is not in the graph");
        if(idx.count(x)){
            if(idx[x] > equator_length)
                throw std::runtime_error("(LocalPicture::Init) Repeated vertices");
        }else{
            idx[x] = next_v++;
            labels.push_back(x);
            south_size++;
        }
    }
    
    if(equator_length + north_size + south_size != G.NumVertices())
        throw std::runtime_error("(LocalPicture::Init) There are spare vertices");
    
    graph = std::vector<std::vector<int>>(next_v);
    for(const Vertex &v: G.Vertices()){
        std::vector<int> &nv = graph[idx.at(v)];
        for(const Vertex &w: G.Neighbors(v))
            nv.push_back(idx.at(w));
        std::sort(nv.begin(), nv.end());
    }
    
    marked = std::vector<bool>(next_v, false);
    for(const Vertex &v: marked_vertices) marked[idx.at(v)] = true;
    
    reflected = false;
    inverted = false;
}



LocalPicture::LocalPicture(const Graph &G,
               const std::vector<Vertex> &equator,
               const std::set<Vertex> &north,
               const std::set<Vertex> &south,
               const std::set<Vertex> &marked_vertices)
{
    Init(G, equator, north, south, marked_vertices);
}



// The following constructor takes a graph G and two vertices x,y.
// Assumptions:
// - x and y are distinct and adjacent in G;
// - The common neighbours of x and y induce a closed path in G.
// The resulting LocalPicture has:
// - The common neighbours of x and y as equator;
// - The neighbours of x at distance >= 2 from y, as northern hemisphere;
// - The neighbours of y at distance >= 2 from x, as southern hemisphere;
// - Graph: the one induced in G by the set of vertices of the three types;
// - Marked vertices are those which are adjacent, in G, to vertices which
//     are not in the LocalPicture (that is, not adjacent to x nor y).
LocalPicture::LocalPicture(const Graph &G, const Vertex &x, const Vertex &y){
    if(x == y) throw std::runtime_error("(LocalPicture::LocalPicture) x == y");
    if(not G.Adjacent(x, y)) throw std::runtime_error("(LocalPicture::LocalPicture) x and y are not adjacent in G");
    
    std::set<Vertex> nx = G.Neighbors(x);
    std::set<Vertex> ny = G.Neighbors(y);
    
    nx.erase(y);
    ny.erase(x);
    
    std::set<Vertex> nxy = nx;
    nxy.insert(ny.begin(), ny.end());
    
    std::set<Vertex> lk = G.CommonNeighbors({x,y});
    for(const Vertex &v: lk){
        nx.erase(v);
        ny.erase(v);
    }
    
    std::set<Vertex> bd;
    for(const Vertex &v: nxy){
        bool cov = true;
        for(const Vertex &w: G.Vertices())
            if(not G.Adjacent(x,w) and not G.Adjacent(y,w))
                if(G.Adjacent(w,v)) cov = false;
        if(not cov) bd.insert(v);
    }
    
    Init(G.InducedSubgraph(nxy), G.InducedSubgraph(lk).MaximalSimplePath(), nx, ny, bd);
}



int LocalPicture::NumVertices() const
{
    return graph.size();
}



int LocalPicture::EquatorLength() const
{
    return equator_length;
}



void LocalPicture::Reflect(){
    reflected = !reflected;
}



void LocalPicture::Invert(){
    inverted = !inverted;
}



// Check if f describes a homomorphism from P to Q:
// - Its domain is the set of vertices of P;
// - It sends adjacent vertices to adjacent or equal vertices;
// - It sends equatorial vertices to equatorial vertices;
// - It sends southern vertices to southern or equatorial vertices;
// - Restricted to equators, it is a nondecreasing map of degree 1 between oriented circles.
bool LocalPicture::IsHomomorphism(const LocalPicture &P, const LocalPicture &Q, const std::map<Vertex,Vertex> &f){
    return IsHomomorphism(P, Q, f, false, false);
}



// Check if f describes a homomorphism from P to Q',
// where Q' is obtain from Q by reflecting and/or inverting as dictated by the parameters.
bool LocalPicture::IsHomomorphism(const LocalPicture &P, const LocalPicture &Q, const std::map<Vertex,Vertex> &f, bool reflect, bool invert){
    for(auto &p: f){
        if(not P.IsVertex(p.first)) return false;
        if(not Q.IsVertex(p.second)) return false;
    }
    
    if(static_cast<int>(f.size()) != P.NumVertices()) return false;
    
    for(int i=0; i<P.NumVertices(); i++){
        Vertex u = P.labels.at(i);
        for(int j: P.graph[i]){
            Vertex v = P.labels.at(j);
            if(f.at(u) != f.at(v) and not Q.Adjacent(f.at(u), f.at(v))) return false;
        }
        
        if(P.marked.at(i) and not Q.IsMarked(f.at(u))) return false;
    }
    
    if(not reflect){
        for(int i=0; i<P.NorthSize(); i++) if(Q.IsSouth(f.at(P.IthNorth(i)))) return false;
        for(int i=0; i<P.SouthSize(); i++) if(Q.IsNorth(f.at(P.IthSouth(i)))) return false;
    }else{
        for(int i=0; i<P.NorthSize(); i++) if(Q.IsNorth(f.at(P.IthNorth(i)))) return false;
        for(int i=0; i<P.SouthSize(); i++) if(Q.IsSouth(f.at(P.IthSouth(i)))) return false;
    }
    
    int changes = 0;
    for(int i=0; i<P.EquatorLength(); i++){
        Vertex u = P.labels.at(i);
        Vertex v = P.EquatorSuccessor(u);
        if(f.at(u) != f.at(v)){
            changes++;
            if(not invert){
                if(f.at(v) != Q.EquatorSuccessor(f.at(u))) return false;
            }else{
                if(f.at(v) != Q.EquatorPredecessor(f.at(u))) return false;
            }
        }
    }
    assert(changes % Q.EquatorLength() == 0);
    if(changes != Q.EquatorLength()) return false;
    
    return true;
}



// The group C_2 x C_2 acts on localpictures by reflections / inversions.
// This function computes a set of representatives for the cosets w.r.t.
// the subset acting trivially on the locapicture.
std::vector<std::bitset<2>> LocalPicture::ReflectInvertRepresentatives() const
{
    std::vector<int> isomorphic;
    for(int k=0; k<2; k++){ //invert?
        for(int j=0; j<2; j++){ //reflect?
            HomFinder F(*this, *this, j > 0, k > 0);
            if(F.HomExists()) isomorphic.emplace_back(j + 2*k);
        }
    }
    std::set<int> taken;
    std::vector<std::bitset<2>> reps;
    for(int z=0; z<4; z++){
        if(taken.count(z)) continue;
        reps.emplace_back(z);
        for(int w:isomorphic) taken.emplace(z ^ w);
    }
    return reps;
}



#endif
