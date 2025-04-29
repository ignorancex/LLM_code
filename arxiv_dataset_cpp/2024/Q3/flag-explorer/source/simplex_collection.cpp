#ifndef SIMPLEX_COLLECTION_CPP
#define SIMPLEX_COLLECTION_CPP

#include "simplex_collection.h"
#include "vertex.h"
#include "simplex.h"
#include "graph.h"

#include "vertex.cpp"
#include "simplex.cpp"
#include "graph.cpp"

#include <stdexcept>



SimplexCollection::SimplexCollection(){}



SimplexCollection::SimplexCollection(const std::set<Simplex> &simplex_set):
    simplex_set(simplex_set)
{}



const std::set<Simplex>& SimplexCollection::GetSimplices() const
{
    return simplex_set;
}



std::set<Simplex> SimplexCollection::Simplices() const
{
    return simplex_set;
}



// Returns a string describing the collection.
// Format: {S_1 S_2 ...}
std::string SimplexCollection::ToString() const
{
    if(simplex_set.empty()) return "{}";
    std::string a = "{";
    for(const Simplex &s: simplex_set){
        a += s.ToString();
        a += " ";
    }
    a.pop_back();
    a += "}";
    return a;
}



// Returns the set of vertices appearing in the collection.
std::set<Vertex> SimplexCollection::Vertices() const
{
    std::set<Vertex> v;
    for(const Simplex &s: simplex_set){
        const std::vector<Vertex> &vv = s.GetVertices();
        v.insert(vv.begin(), vv.end());
    }
    return v;
}



int SimplexCollection::Rank() const
{
    int rk = -1;
    for(const Simplex &s: simplex_set) rk = std::max(rk, static_cast<int>(s.Rank()));
    return rk;
}



// Returns true if the collection of simplices is subset-closed, false otherwise.
bool SimplexCollection::IsSimplicialComplex() const
{
    bool is_sc = true;
    
    for(const Simplex &s: simplex_set){
        for(const Simplex &t: s.Facets()){
            if(simplex_set.count(t) == 0) is_sc = false;
        }
    }
    
    return is_sc;
}



// Adds a simplex and returns:
// - A pointer to the newly inserted simplex (or an equal one already there);
// - Whether a new simplex was actually inserted.
std::pair<SimplexCollection::ptr,bool> SimplexCollection::Insert(const Simplex &s){
    return simplex_set.insert(s);
}



// Inserts all the simplices of the provided collection.
void SimplexCollection::Merge(const SimplexCollection &other){
    simplex_set.insert(other.simplex_set.begin(), other.simplex_set.end());
}



// Returns a collection whose vertices are relabelled as prescribed by f.
// f is not required to preserve the ordering.
SimplexCollection SimplexCollection::Relabelled(const std::map<Vertex,Vertex> &f) const
{
    SimplexCollection C;
    for(const Simplex &s: simplex_set) C.Insert(s.Relabel(f));
    return C;
}



// Returns a collection whose vertices are relabelled starting from 0, preserving the vertex ordering.
SimplexCollection SimplexCollection::Relabelled() const
{
    std::map<Vertex,Vertex> f;
    Vertex i = 0;
    for(const Vertex &v: Vertices()){
        f[v] = i;
        i++;
    }
    return Relabelled(f);
}



// Returns the join with another collection C.
// Vertices of C are relabelled if needed.
SimplexCollection SimplexCollection::Join(const SimplexCollection &C) const
{
    if(C.simplex_set.empty() or simplex_set.empty()) return SimplexCollection();
    
    std::set<Vertex> vs = Vertices();
    if(vs.empty()) return SimplexCollection(C.simplex_set);
    
    std::map<Vertex,Vertex> nn;
    Vertex i = 0;
    for(const Vertex &v: C.Vertices()){
        if(vs.count(v)){
            while(vs.count(i)) i++;
            nn[v] = i;
            vs.insert(i);
        }else{
            vs.insert(v);
        }
    }
    
    SimplexCollection ans;
    for(const Simplex &s: simplex_set){
        for(const Simplex &t: C.simplex_set){
            ans.Insert(s.Merge(t.Relabel(nn)));
        }
    }
    
    return ans;
}



// Identifies the vertices U and V. U is kept and V is discarded.
SimplexCollection SimplexCollection::Identify(const Vertex &U, const Vertex &V){
    std::set<Simplex> nsc;
    for(const Simplex &s: simplex_set){
        std::set<Vertex> ss(s.GetVertices().begin(), s.GetVertices().end());
        if(ss.count(V)){
            ss.erase(V);
            ss.insert(U);
        }
        nsc.insert(Simplex(ss));
    }
    return SimplexCollection(nsc);
}



// Triangulation of the circle, with n vertices, numbered from 0 to n-1 in order.
SimplexCollection SimplexCollection::Circle(int n){
    if(n <= 2) throw std::runtime_error("(SimplexCollection::Circle) The number of vertices must be at least 3");
    SimplexCollection ans;
    ans.Insert(Simplex());
    for(int i=0; i<n; i++) ans.Insert(Simplex(std::vector<Vertex>{i}));
    for(int i=1; i<n; i++) ans.Insert(Simplex(std::vector<Vertex>{i-1, i}));
    ans.Insert(Simplex(std::vector<Vertex>{0,n-1}));
    return ans;
}



// Returns the one-skeleton, as a graph.
Graph SimplexCollection::OneSkeleton() const
{
    Graph G;
    for(const Simplex &s: simplex_set){
        if(s.Rank() > 2) break;
        if(s.Rank() == 0) continue;
        if(s.Rank() == 1) G.AddVertex(s.GetVertices()[0]);
        if(s.Rank() == 2){
            G.Connect(s.GetVertices()[0], s.GetVertices()[1]);
        }
    }
    return G;
}

#endif
