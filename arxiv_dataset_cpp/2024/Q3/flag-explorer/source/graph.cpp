#ifndef GRAPH_CPP
#define GRAPH_CPP

#include "graph.h"
#include "vertex.h"
#include "simplex.h"
#include "simplex_collection.h"

#include "vertex.cpp"
#include "simplex.cpp"
#include "simplex_collection.cpp"
#include "input_output.cpp"

#include <stdexcept>
#include <cassert>
#include <algorithm>



Graph::Graph(){}
Graph::Graph(const std::map<Vertex,std::set<Vertex>> &adj): adj(adj) {}



// Reads a graph from an input stream
// Format: {x_1 y_1 x_2 y_2 ... x_M y_M}
// where x_i are vertices and y_i are either vertices or sets of vertices,
// meaning that, for every i, x_i is adjacent to every vertex in y_i.
// If y_i is a vertex (not a set), then an optional ":" between x_i and y_i is allowed.
// The delimiters (first and last chars) can be {}, [] or ().
// As separators between x_i, y_i commas and colons are allowed.
std::istream& operator >> (std::istream &s, Graph &G)
{
    char end_delim;
    char c;
    
    if(not (s >> c)) return s;
    switch(c){
        case '(': end_delim = ')'; break;
        case '[': end_delim = ']'; break;
        case '{': end_delim = '}'; break;
        default: throw std::runtime_error("Error while reading a graph");
    }
    
    G = {};
    
    while(s >> c){
        if(c == end_delim) return s;
        if(c != ',' and c != ';'){
            s.unget();
            Vertex x;
            s >> x;
            G.AddVertex(x);
            
            if(not(s >> c)) throw std::runtime_error("Error while reading a graph");
            if(c != ':') s.unget();
            switch(c){
                case '(':
                case '[':
                case '{':
                {
                    std::set<Vertex> sy;
                    s >> sy;
                    for(const Vertex &y: sy) G.Connect(x, y);
                    break;
                }
                default:
                {
                    Vertex y;
                    s >> y;
                    G.Connect(x, y);
                }
            }
        }
    }
    throw std::runtime_error("Error while reading a graph");
    return s;
}



// Writes a graph on an output stream
std::ostream& operator << (std::ostream &o, const Graph &G)
{
    o << "{\n";
    for(const auto &p: G.adj){
        o << p.first << "   [ ";
        for(const Vertex &v: p.second) o << v << " ";
        o << "]\n";
    }
    o << "}" << std::flush;
    return o;
}



int Graph::NumVertices() const
{
    return adj.size();
}



int Graph::NumEdges() const
{
    int ans = 0;
    for(const auto &p: adj) ans += p.second.size();
    return ans/2;
}



// Does the graph have vertex v?
bool Graph::IsVertex(const Vertex &v) const
{
    return adj.count(v);
}



// Adds an edge between u and v (if not present already, and u != v).
// It is not required that the vertices be already part of the graph:
// if they are missing, they are added.
void Graph::Connect(const Vertex &v, const Vertex &u)
{
    if(u == v) adj[v];
    else{
        adj[v].insert(u);
        adj[u].insert(v);
    }
}



void Graph::AddVertex(const Vertex &v)
{
    adj[v];
}



void Graph::RemoveVertex(const Vertex &v)
{
    for(const Vertex &u: adj[v]){
        adj[u].erase(v);
    }
    adj.erase(v);
}



// Returns true if u and v are in the graph and are equal or connected by an edge.
bool Graph::Adjacent(const Vertex &v, const Vertex &u) const
{
    if(not adj.count(v)) return false;
    return (v == u) or adj.at(v).count(u);
}



// Returns the set of vertices of G
std::set<Vertex> Graph::Vertices() const
{
    std::set<Vertex> s;
    for(const auto &p: adj) s.insert(p.first);
    return s;
}



// Returns the set of vertices adjacent to v.
// v must be a vertex of the graph.
const std::set<Vertex>& Graph::Neighbors(Vertex v) const
{
    return adj.at(v);
}



// Edges; only one among {u,v} and {v,u} is returned, for u and v distinct adjacent vertices.
std::vector<std::pair<Vertex,Vertex>> Graph::Edges() const
{
    std::vector<std::pair<Vertex,Vertex>> e;
    for(const auto &x: adj){
        const Vertex &v = x.first;
        for(const Vertex &w: x.second) if(v < w) e.emplace_back(v, w);
    }
    return e;
}



// Returns the set of vertices x such that x is adjacent (and distinct) to every vertex in V.
std::set<Vertex> Graph::CommonNeighbors(const std::set<Vertex> &V) const
{
    if(V.size() == 0){
        return Vertices();
    }else{
        std::set<Vertex> s;
        for(const Vertex &v: adj.at(*(V.begin()))){
            bool ok = true;
            for(const Vertex &w: V){
                if(not adj.at(w).count(v)){
                    ok = false;
                    break;
                }
            }
            if(ok) s.insert(v);
        }
        return s;
    }
}



// Every vertex adjacent to y becomes adjacent to x, and y is deleted (if != x).
// x and y must be vertices of the graph.
void Graph::IdentifyVertices(const Vertex &x, const Vertex &y)
{
    if(adj.count(x) == 0 or adj.count(y) == 0) throw std::runtime_error("(Graph::IdentityVertices) The graph does not contain the given vertices");
    if(x == y) return;
    for(const Vertex &v: adj[y]) if(v != x){
        if(adj[v].count(x) == 0){
            adj[x].insert(v);
            adj[v].insert(x);
        }
        adj[v].erase(y);
    }
    if(adj[x].count(y)) adj[x].erase(y);
    adj.erase(y);
}



// Adds a new vertex subdividing the edge [x,y]. It becomes adjacent to common neighs. of x and y.
// x and y must be distinc adjacent vertices.
void Graph::SubdivideEdge(const Vertex &x, const Vertex &y)
{
    if(x == y) throw std::runtime_error("(Graph::SubdivideEdge) The two vertices coincide");
    if(adj.count(x) == 0 or adj.count(y) == 0) throw std::runtime_error("(Graph::SubdivideEdge) The graph does not contain the given vertices");
    if(adj[x].count(y) == 0) throw std::runtime_error("(Graph::SubdivideEdge) The two vertices are not adjacent");
    Vertex z = 0;
    while(adj.count(z)) z++;
    Connect(x, z);
    Connect(y, z);
    for(const Vertex &v: CommonNeighbors({x,y})) Connect(v, z);
    adj[x].erase(y);
    adj[y].erase(x);
}



// Determines whether x and y are distinct vertices which are part of a square (an induced subgraph isomorphic to a circle of length 4).
bool Graph::InSquare(const Vertex &x, const Vertex &y) const
{
    if(x == y) return false;
    if(not adj.count(x) or not adj.count(y)) return false;
    if(adj.at(x).count(y)){
        for(const Vertex &vx: adj.at(x)) if(vx != y and not adj.at(vx).count(y))
            for(const Vertex &vy: adj.at(y)) if(vy != x and not adj.at(vy).count(x))
                if(adj.at(vx).count(vy)) return true;
        return false;
    }else{
        for(const Vertex &v: adj.at(x)) if(adj.at(v).count(y))
            for(const Vertex &w: adj.at(x)) if(w != v and adj.at(w).count(y))
                if(not adj.at(v).count(w)) return true;
        return false;
    }
}



// A circular graph with n vertices, numbered from 0 to n-1 in order
Graph Graph::Circle(int n)
{
    if(n <= 2) throw std::runtime_error("(Graph::Circle) The number of vertices must be at least 3");
    Graph ans;
    for(int i=1; i<n; i++) ans.Connect(i-1, i);
    ans.Connect(0, n-1);
    return ans;
}



// Returns the join with another graph.
// Vertices of the other graph are relabelled as needed.
Graph Graph::Join(const Graph &other) const
{
    if(other.adj.empty() or adj.empty()) return Graph();
    
    std::set<Vertex> vs = Vertices();
    if(vs.empty()) return other;
    
    const std::set<Vertex> &other_v = other.Vertices();
    
    std::map<Vertex,Vertex> nn;
    Vertex i = 0;
    for(const Vertex &v: other_v){
        if(vs.count(v)){
            while(vs.count(i)) i++;
            nn[v] = i;
            vs.insert(i);
        }else{
            nn[v] = v;
            vs.insert(v);
        }
    }
    
    Graph ans;
    for(const auto &p: other.adj){
        for(const Vertex &v: p.second) ans.Connect(nn[p.first], nn[v]);
    }
    
    for(const auto &p: adj){
        for(const Vertex &v: p.second) ans.Connect(p.first, v);
        for(const Vertex &v: other_v) ans.Connect(p.first, nn[v]);
    }
    
    return ans;
}



//Takes two copies of G-v, identifying the neighbous of the two copies of v.
Graph Graph::VertexDouble(const Vertex &v) const
{
    const std::set<Vertex> &vert = Vertices();
    std::set<Vertex> far;
    for(const Vertex &w: vert) if(v != w and not Adjacent(v, w)) far.insert(w);
    
    std::map<Vertex,Vertex> nn;
    Vertex i = 0;
    for(const Vertex &w: far){
        while(i != v and vert.count(i)) i++;
        nn[w] = i;
        i++;
    }
    
    Graph H(adj);
    H.RemoveVertex(v);
    for(const Vertex &w: far) H.AddVertex(nn[w]);
    for(const Vertex &w: far) for(const Vertex &z: Neighbors(w)){
        if(far.count(z)) H.Connect(nn[w],nn[z]);
        else H.Connect(nn[w],z);
    }
    
    return H;
}



// The induced subgraph whose vertices are in V.
Graph Graph::InducedSubgraph(const std::set<Vertex> &V) const
{
    Graph H;
    for(const Vertex &v: V) if(adj.count(v)){
        H.AddVertex(v);
        for(const Vertex &w: Neighbors(v))
            if(V.count(w)) H.Connect(v, w);
    }
    return H;
}



// Returns a maximal simple path, meaning that it cannot be extended without repeating vertices.
std::vector<Vertex> Graph::MaximalSimplePath() const
{
    if(adj.empty()) return std::vector<Vertex>();
    const Vertex &o = adj.begin()->first;
    
    std::set<Vertex> visited;
    visited.insert(o);
    
    std::vector<Vertex> ans;
    Vertex last = o;
    bool expanding = true;
    while(expanding){
        expanding = false;
        for(const Vertex &v: Neighbors(last)) if(visited.count(v) == 0){
            expanding = true;
            ans.push_back(v);
            visited.insert(v);
            last = v;
            break;
        }
    }
    
    std::reverse(ans.begin(), ans.end());
    ans.push_back(o);
    
    last = o;
    expanding = true;
    while(expanding){
        expanding = false;
        for(const Vertex &v: Neighbors(last)) if(visited.count(v) == 0){
            expanding = true;
            ans.push_back(v);
            visited.insert(v);
            last = v;
            break;
        }
    }
    
    std::reverse(ans.begin(), ans.end());
    return ans;
}



// Input:
// - s: a set of vertices of G forming a clique;
// - cone_v: a set of vertices of G, each of them adjacent to every vertex of s, but not in s.
// Output: the collection of cliques in G containing s and contained in the union of s and cone_v.
SimplexCollection Graph::CliqueCollection(std::set<Vertex> &s, const std::set<Vertex> &cone_v) const
{
    SimplexCollection ans({Simplex(s)});
    for(const Vertex &u: cone_v){
        s.insert(u);
        std::set<Vertex> new_cone_v;
        for(const Vertex &v: cone_v) if(u < v and adj.at(u).count(v)) new_cone_v.insert(v);
        ans.Merge(CliqueCollection(s, new_cone_v));
        s.erase(u);
    }
    return ans;
}



// Input:
// - s: a set of vertices of G forming a clique;
// Output: the collection of cliques in G containing s.
SimplexCollection Graph::CliqueCollection(std::set<Vertex> &s) const
{
    std::set<Vertex> cone_v;
    
    for(auto &vv: adj){
        if(s.count(vv.first) == 0){
            bool cone = true;
            for(const Vertex &u: s) if(vv.second.count(u) == 0) cone = false;
            if(cone) cone_v.emplace(vv.first);
        }
    }
    
    return CliqueCollection(s, cone_v);
}



// Returns the collection of cliques of G.
SimplexCollection Graph::CliqueCollection() const
{
    std::set<Vertex> s = {};
    return CliqueCollection(s);
}



// Returns a graph whose vertices are relabelled as prescribed by f.
// f is not required to preserve the ordering.
Graph Graph::Relabelled(const std::map<Vertex,Vertex> &f) const
{
    Graph H;
    for(const auto &p: adj)
        for(const Vertex &v: p.second)
            H.adj[f.at(p.first)].insert(f.at(v));
    return H;
}



// Returns a graph whose vertices are relabelled starting from 0, preserving the vertex ordering.
Graph Graph::Relabelled() const
{
    std::map<Vertex,Vertex> f;
    Vertex i = 0;
    for(const auto &vv: adj){
        f[vv.first] = i;
        i++;
    }
    return Relabelled(f);
}



// Computes gamma_2 = 16 - 8V + 4E - 2F + T, where:
// - V = number of vertices;
// - E = number of edges;
// - F = number of triangles (cliques of size 3);
// - T = number of tetrahedrons (cliques of size 4);
int Graph::Gamma2() const
{
    int V = NumVertices();
    int E = NumEdges();
    int F = 0;
    int T = 0;
    
    for(auto e: Edges()){
        std::set<Vertex> l = CommonNeighbors({e.first, e.second});
        F += l.size();
        for(const Vertex &v: l){
            T += CommonNeighbors({e.first, e.second, v}).size();
        }
    }
    F /= 3;
    T /= 12;
    return 16 - 8*V + 4*E - 2*F + T;
}



// Returns true if f sends adjacent vertices of G to adjacent (or equal) vertices of H.
bool Graph::IsHomomorphism(const Graph &G, const Graph &H, const std::map<Vertex,Vertex> f)
{
    for(const auto &vv: G.adj){
        if(not f.count(vv.first)) return false;
    }
    for(const auto &vv: G.adj){
        for(const Vertex &w: vv.second){
            if(not H.Adjacent(f.at(vv.first), f.at(w))) return false;
        }
    }
    return true;
}

#endif
