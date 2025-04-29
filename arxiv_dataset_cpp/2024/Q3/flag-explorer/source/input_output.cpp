#ifndef INPUT_OUTPUT_CPP
#define INPUT_OUTPUT_CPP

// This file contains functions to write and read vectors, sets and maps.
// It also has a function that discard "comment lines" from a input stream.

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <map>
#include <stdexcept>
#include <limits>



// Discards lines having '#' as first non-whitespace character.
void DiscardComments(std::istream &s)
{
    char c;
    while(s >> c){
        s.unget();
        if(c == '#'){
            s.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
        }else{
            return;
        }
    }
    return;
}



// Reads a vector<T> from an input stream.
// Format: (T-object,...,T-object)
// Can use (), [], or {} as external delimiters.
// ',' ';' and whitespaces can be used to separate objects.
template<class T>
std::istream& operator >> (std::istream &s, std::vector<T> &a)
{
    char end_delim;
    char c;
    
    if(not (s >> c)) return s;
    
    switch(c){
        case '(': end_delim = ')'; break;
        case '[': end_delim = ']'; break;
        case '{': end_delim = '}'; break;
        default: throw std::runtime_error("Error while reading a vector");
    }
    
    a = {};
    
    while(s >> c){
        if(c == end_delim) return s;
        if(c != ',' and c != ';'){
            s.unget();
            T x;
            s >> x;
            a.emplace_back(x);
        }
    }
    throw std::runtime_error("Error while reading a vector");
    return s;
}



// Reads a set<T> from an input stream.
// Format: [T-object,...,T-object]
// Can use (), [], or {} as external delimiters.
// ',' ';' and whitespaces can be used to separate objects.
template<class T>
std::istream& operator >> (std::istream &s, std::set<T> &a)
{
    char end_delim;
    char c;
    
    if(not (s >> c)) return s;
    
    switch(c){
        case '(': end_delim = ')'; break;
        case '[': end_delim = ']'; break;
        case '{': end_delim = '}'; break;
        default: throw std::runtime_error("Error while reading a set");
    }
    
    a = {};
    
    while(s >> c){
        if(c == end_delim) return s;
        if(c != ',' and c != ';'){
            s.unget();
            T x;
            s >> x;
            a.insert(x);
        }
    }
    throw std::runtime_error("Error while reading a set");
    return s;
}



// Reads a map<T,S> from an input stream.
// Format: [T-object:S-object,...,T-object:S-object]
// Can use (), [], or {} as external delimiters
// ',' ';' and whitespaces can be used to separate T-S pairs.
// The elements of a pair can be separated by whitespaces and (optionally) a ':'
template<class T, class S>
std::istream& operator >> (std::istream &s, std::map<T,S> &a)
{
    char end_delim;
    char c;
    
    if(not (s >> c)) return s;
    
    switch(c){
        case '(': end_delim = ')'; break;
        case '[': end_delim = ']'; break;
        case '{': end_delim = '}'; break;
        default: throw std::runtime_error("Error while reading a map");
    }
    
    a = {};
    
    while(s >> c){
        if(c == end_delim) return s;
        if(c != ',' and c != ';'){
            s.unget();
            T x;
            s >> x;
            
            if(not(s >> c)) throw std::runtime_error("Error while reading a map");
            if(c != ':') s.unget();
            S y;
            s >> y;
            
            a[x] = y;
        }
    }
    throw std::runtime_error("Error while reading a map");
    return s;
}



// Writes a set to an output stream.
// Format: [T-object,...,T-object]
template<class T>
std::ostream& operator <<(std::ostream &o, const std::set<T> &s)
{
    o << "[";
    for(auto it=s.begin(); it!=s.end(); ){
        o << *it;
        it++;
        if(it != s.end()) o << ",";
    }
    o << "]";
    return o;
}



// Writes a vector to an output stream.
// Format: (T-object,...,T-object)
template<class T>
std::ostream& operator <<(std::ostream &o, const std::vector<T> &s)
{
    o << "(";
    for(auto it=s.begin(); it!=s.end(); ){
        o << *it;
        it++;
        if(it != s.end()) o << ",";
    }
    o << ")";
    return o;
}



// Writes a map to an output stream.
// Format: {k1: o1, ... }
template<class T,class S>
std::ostream& operator <<(std::ostream &o, const std::map<T,S> &m)
{
    o << "{";
    for(auto it=m.begin(); it!=m.end(); ){
        o << it->first << ": " << it->second;
        it++;
        if(it != m.end()) o << ", ";
    }
    o << "}";
    return o;
}

#endif
