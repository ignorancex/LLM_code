#ifndef LOCALPICTURE_HOMFINDER_CPP
#define LOCALPICTURE_HOMFINDER_CPP

#include "localpicture.h"
#include "vertex.h"

#include "vertex.cpp"
#include "localpicture.cpp"

#include <vector>
#include <queue>
#include <map>
#include <stdexcept>
#include <cassert>



LocalPicture::HomFinder::HomFinder(const LocalPicture &P, const LocalPicture &Q, bool reflect, bool invert):
    P(P),
    Q(Q),
    rQ(Q.NumVertices())
{
    for(int i=0; i<Q.NumVertices(); i++){
        unsigned int it = 0;
        for(int j=0; j<Q.NumVertices(); j++){
            if(it < Q.graph.at(i).size() and j == Q.graph.at(i).at(it)) it++;
            else if(i != j){
                rQ[i].push_back(j);
            }
        }
    }
    
    computed = false;
    hom_found = false;
    inverted = P.inverted xor Q.inverted xor invert;
    reflected = P.reflected xor Q.reflected xor reflect;
}



LocalPicture::HomFinder::HomFinder(const LocalPicture &P, const LocalPicture &Q):
    HomFinder(P, Q, false, false)
{}



bool LocalPicture::HomFinder::HomExists()
{
    if(not computed) Compute();
    return hom_found;
}



std::map<Vertex,Vertex> LocalPicture::HomFinder::Homomorphism()
{
    if(not computed) Compute();
    if(not HomExists()) throw std::runtime_error("(LocalPicture::HomFinder::Homomorphism) The homomorphism does not exists");
    
    std::map<Vertex,Vertex> ans;
    for(int i=0; i<P.NumVertices(); i++)
        ans[P.labels.at(i)] = Q.labels.at(f.at(i));
    
    return ans;
}



void LocalPicture::HomFinder::Compute()
{
    if(computed) return;
    
    eqP = std::vector<int>(P.equator_length);
    if(not inverted) for(int i=0; i<P.equator_length; i++) eqP[i] = i;
    else for(int i=0; i<P.equator_length; i++) eqP[i] = P.equator_length - 1 - i;
        
    for(int rot=0; rot<P.equator_length; rot++){
        std::rotate(eqP.begin(), eqP.begin()+1, eqP.end());
        
        f = std::vector<int>(P.NumVertices(), -1);
        is_allowed = std::vector<std::vector<bool>>(P.NumVertices(), std::vector<bool>(Q.NumVertices(), true));
        
        //Initialize is_allowed with the initial constraints
        for(int i=1; i<Q.equator_length; i++) is_allowed[eqP[0]][i] = false;
        for(int i=0; i<Q.equator_length-1; i++) is_allowed[eqP[P.equator_length-1]][i] = false;
        
        for(int i=0; i<P.equator_length; i++)
            for(int j=Q.equator_length; j<Q.NumVertices(); j++)
                is_allowed[i][j] = false;
        
        if(not reflected){
            for(int i=0; i<P.north_size; i++) for(int j=0; j<Q.south_size; j++)
                is_allowed[P.equator_length + i][Q.equator_length + Q.north_size + j] = false;
            for(int i=0; i<P.south_size; i++) for(int j=0; j<Q.north_size; j++)
                is_allowed[P.equator_length + P.north_size + i][Q.equator_length + j] = false;
        }else{
            for(int i=0; i<P.north_size; i++) for(int j=0; j<Q.north_size; j++)
                is_allowed[P.equator_length + i][Q.equator_length + j] = false;
            for(int i=0; i<P.south_size; i++) for(int j=0; j<Q.south_size; j++)
                is_allowed[P.equator_length + P.north_size + i][Q.equator_length + Q.north_size + j] = false;
        }
        
        for(int i=0; i<P.NumVertices(); i++) if(P.marked[i])
            for(int j=0; j<Q.NumVertices(); j++) if(not Q.marked[j])
                is_allowed[i][j] = false;
        
        count_alternatives = std::vector<int>(P.graph.size(), 0);
        for(int i=0; i<P.NumVertices(); i++){
            for(int j=0; j<Q.NumVertices(); j++)
                if(is_allowed[i][j]) count_alternatives[i]++;
        }
        
        pq = std::priority_queue<std::pair<int,int>>();
        for(int i=0; i<P.NumVertices(); i++){
            pq.emplace(-count_alternatives[i], i);
        }
        
        ComputeRecursive();
        if(hom_found){
            for(int i=0; i<P.NumVertices(); i++) if(f[i] < 0 or f[i] >= Q.NumVertices())
                throw std::runtime_error("(LocalPicture::Homfinder::Compute) Invalid map");
            // Check the restriction on the equator.
            if(f[eqP[0]] != 0 or f[eqP[P.equator_length-1]] != Q.equator_length-1)
                throw std::runtime_error("(LocalPicture::Homfinder::Compute) Failed extremal check");
            for(int i=0; i<P.equator_length-1; i++){
                if(f[eqP[i+1]] != f[eqP[i]] and f[eqP[i+1]] != f[eqP[i]] + 1)
                    throw std::runtime_error("(LocalPicture::Homfinder::Compute) Failed equator check");
            }
            break;
        }
    }
    computed = true;
}



void LocalPicture::HomFinder::ComputeRecursive()
{
    int v = -1;
    while(true){
        if(pq.empty()){
            hom_found = true;
            return;
        }
        auto p = pq.top();
        pq.pop();
        v = p.second;
        if(f[v] == -1 and count_alternatives[v] == -p.first) break;
    }
    assert(v >= 0);
    
    for(int u=0; u<Q.NumVertices(); u++) if(is_allowed[v][u]){
        std::vector<std::pair<int,int>> deleted_alternatives;
        std::vector<int> changed_vertices;
        //Assign f[v] = u, and adjust is_allowed.
        //Record added constraints in deleted_alternatives.
        f[v] = u;
        
        for(int w: P.graph[v]){
            for(int z: rQ[u])
                RemoveAlternative(w, z, deleted_alternatives, changed_vertices);
        }
        if(v < P.equator_length){
            int ev = 0;
            if(not inverted){
                if(v >= eqP[0]) ev = v - eqP[0];
                else ev = P.EquatorLength() + v - eqP[0];
            }else{
                if(v <= eqP[0]) ev = eqP[0] - v;
                else ev = P.EquatorLength() + eqP[0] - v;
            }
            assert(ev >= 0 and ev < P.EquatorLength());
            assert(eqP[ev] == v);
            
            for(int i=1; ev+i<P.equator_length and u+i+1<Q.equator_length and f[eqP[ev+i]] == -1; i++)
                for(int j=u+i+1; j<Q.equator_length; j++)
                    RemoveAlternative(eqP[ev+i], j, deleted_alternatives, changed_vertices);
            for(int i=1; ev-i>=0 and u-i>0 and f[eqP[ev-i]] == -1; i++)
                for(int j=0; j<u-i; j++)
                    RemoveAlternative(eqP[ev-i], j, deleted_alternatives, changed_vertices);
            for(int i=ev+1; i<P.equator_length and f[eqP[i]] == -1; i++)
                for(int j=0; j<u; j++)
                    RemoveAlternative(eqP[i], j, deleted_alternatives, changed_vertices);
            for(int i=ev-1; i>=0 and f[eqP[i]] == -1; i--)
                for(int j=u+1; j<Q.equator_length; j++)
                    RemoveAlternative(eqP[i], j, deleted_alternatives, changed_vertices);
        }
        
        std::sort(changed_vertices.begin(), changed_vertices.end());
        auto last = std::unique(changed_vertices.begin(), changed_vertices.end());
        changed_vertices.erase(last, changed_vertices.end());
        for(int w: changed_vertices) pq.emplace(-count_alternatives[w], w);
        
        ComputeRecursive();
        if(hom_found) return;
        
        //Revert changes
        f[v] = -1;
        for(const auto &p: deleted_alternatives){
            assert(is_allowed[p.first][p.second] == false);
            is_allowed[p.first][p.second] = true;
            count_alternatives[p.first]++;
        }
        for(int w: changed_vertices) pq.emplace(-count_alternatives[w], w);
    }
    pq.emplace(-count_alternatives[v], v);
}



void LocalPicture::HomFinder::RemoveAlternative(int v, int u, 
                                                std::vector<std::pair<int,int>> &deleted_alternatives,
                                                std::vector<int> &changed_vertices)
{
    if(not is_allowed[v][u]) return;
    is_allowed[v][u] = false;
    count_alternatives[v]--;
    deleted_alternatives.emplace_back(v, u);
    changed_vertices.push_back(v);
}



#endif
