#ifndef SIMPLEX_CPP
#define SIMPLEX_CPP

#include "simplex.h"
#include "vertex.h"

#include "vertex.cpp"

#include <algorithm>
#include <stdexcept>



Simplex::Simplex():
    vertex_vector(0),
    rank(0)
{}



// Creates a Simplex from a vector of vertices.
// The vector must be strictly increasingly ordered.
Simplex::Simplex(const std::vector<Vertex> &vertex_vector):
    vertex_vector(vertex_vector),
    rank(vertex_vector.size())
{
    if(rank > 1) for(unsigned int i=0; i<rank-1; i++){
        if(not (vertex_vector[i]<vertex_vector[i+1]))
            throw std::runtime_error("(Simplex::Simplex) The provided vector of vertices is not strictly increasing");
    }
}



Simplex::Simplex(const std::set<Vertex> &vertex_set):
    vertex_vector(vertex_set.begin(), vertex_set.end()),
    rank(vertex_set.size())
{}



bool Simplex::operator <(const Simplex &other) const
{
    if(rank < other.rank) return true;
    if(other.rank < rank) return false;
    return (vertex_vector < other.vertex_vector);
}



bool Simplex::operator ==(const Simplex &other) const
{
    return (vertex_vector == other.vertex_vector);
}



unsigned int Simplex::Rank() const
{
    return rank;
}



const std::vector<Vertex>& Simplex::GetVertices() const
{
    return vertex_vector;
}



std::vector<Vertex> Simplex::Vertices() const
{
    return vertex_vector;
}



// Returns a string describing the simplex.
// Format: [v_1 v_2 ... ]
std::string Simplex::ToString() const
{
    if(rank == 0) return "[]";
    std::string a = "[";
    for(const Vertex &v: vertex_vector){
        a += std::to_string(v);
        a += " ";
    }
    a.pop_back();
    a += "]";
    return a;
}



// Returns the codimension-1 face opposite to the given vertex.
// If v is not a vertex, returns a simplex identical to the current one.
Simplex Simplex::Facet(const Vertex &v) const
{
    Simplex f;
    f.vertex_vector.reserve(rank-1);
    for(const Vertex &w: vertex_vector) if(w != v) f.vertex_vector.emplace_back(w);
    f.rank = f.vertex_vector.size();
    return f;
}



//Returns a vector containing all the codimension-1 faces of the simplex.
std::vector<Simplex> Simplex::Facets() const
{
    std::vector<Simplex> fs;
    fs.reserve(rank);
    for(const Vertex &v: vertex_vector) fs.emplace_back(Facet(v));
    return fs;
}



// Returns the subsimplex whose vertices are those NOT contained in s.
Simplex Simplex::OppositeFace(const Simplex &s) const
{
    Simplex f;
    unsigned int i = 0;
    for(const Vertex &v: vertex_vector){
        if(i < s.rank and v == s.vertex_vector[i]){
            i++;
            continue;
        }else{
            f.vertex_vector.emplace_back(v);
        }
    }
    f.rank = f.vertex_vector.size();
    return f;
}



// Returns a vector storing all the faces of the simplex, of any dimension.
std::vector<Simplex> Simplex::Faces() const
{
    std::vector<Simplex> f;
    f.reserve(1<<rank);
    for(int z = 0; z<(1<<rank); z++){
        Simplex sub;
        for(unsigned int i=0; i<rank; i++) if(z&(1<<i))
            sub.vertex_vector.emplace_back(vertex_vector[i]);
        sub.rank = sub.vertex_vector.size();
        f.emplace_back(sub);
    }
    return f;
}



// Returns a new simplex with vertices relabelled as prescribed by f.
// f is not required to preserve order.
Simplex Simplex::Relabel(const std::map<Vertex,Vertex> &f) const
{
    std::vector<Vertex> a;
    a.reserve(rank);
    for(const Vertex &v: vertex_vector)
        if(f.count(v)) a.emplace_back(f.at(v));
        else a.emplace_back(v);
    std::sort(a.begin(), a.end());
    return Simplex(a);
}



// Returns a simplex by merging the current one with another.
Simplex Simplex::Merge(const Simplex &other) const
{
    std::vector<Vertex> a;
    a.reserve(rank + other.rank);
    
    unsigned int i = 0;
    for(unsigned int j = 0; j < other.rank; j++){
        while(i < rank and vertex_vector[i] < other.vertex_vector[j])
            a.push_back(vertex_vector[i++]);
        a.push_back(other.vertex_vector[j]);
    }
    for(; i < rank; i++) a.push_back(vertex_vector[i]);
    
    return Simplex(a);
}

#endif
