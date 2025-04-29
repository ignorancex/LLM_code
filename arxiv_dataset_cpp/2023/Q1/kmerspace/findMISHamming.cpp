/*
 * This program is the source code for the paper "On the Maximal Independent 
 * Sets of k-mers with the Edit Distance".
 *
 * Author: Leran Ma (lkm5463@psu.edu)
 * Date:   3:38 PM, Monday, April 10, 2023
 */

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <deque>
#include <iterator>
#include <unistd.h>
#include <sys/wait.h>
#include <ios>
#include <fstream>
#include <string>

using namespace std;

/*
 * Prints a k-mer given its binary encoding according to the following 
 * binary-to-base translation: 00 -> A, 01 -> C, 10 -> G, 11 -> T
 *
 * enc: The binary encoding of the k-mer
 * k  : The length of the k-mer
 */
void printKmer( unsigned long int enc, int k )
{
    char base[4] = {'A', 'C', 'G', 'T'};
    char kmer[k + 1];
    kmer[k] = '\0';
    for (int i = k - 1; i >= 0; --i)
    {
        kmer[i] = base[enc & 3];
        enc = enc >> 2;
    }
    cerr << kmer;
}

/*
 * Calculates the Hamming distance between 2 k-mers
 *
 * s1: The encoding of the first k-mer
 * s2: The encoding of the second k-mer
 * k : The length of the two k-mers
 * d : The maximum edit distance allowed
 */
int hammingDist( const unsigned long int s1, const unsigned long int s2, 
              const int k, const int d )
{
    unsigned long int temp = s1 ^ s2;
    int count = 0;
    for (int i = 0; i < k; ++i)
    {
        count += ((temp & 3) != 0);
        temp >>= 2;
    }
    return count;
}

/*
 * Reports the time and space usage
 */
void reportPerformance()
{
    string temp;
    unsigned long int utime, stime, vmpeak, vmhwm;

    // Find time usage in "/proc/self/stat"
    ifstream time_stream("/proc/self/stat", ios_base::in);

    // Skip all irrelevant attributes
    for (int i = 0; i < 13; ++i)
    {
        time_stream >> temp;
    }

    time_stream >> utime >> stime;
    time_stream.close();

    // Find space usage in "/proc/self/status"
    ifstream space_stream("/proc/self/status", ios_base::in);
    while ( temp.compare("VmPeak:") != 0 )
    {
        space_stream >> temp;
    }
    space_stream >> vmpeak;
    while ( temp.compare("VmHWM:") != 0 )
    {
        space_stream >> temp;
    }
    space_stream >> vmhwm;
    space_stream.close();

    cerr << "Performance Report\n"
         << "Time in user mode:        " 
         << utime / sysconf(_SC_CLK_TCK) << " sec\n"
         << "Time in kernel mode:      " 
         << stime / sysconf(_SC_CLK_TCK) << " sec\n"
         << "Peak virtual memory size: " << vmpeak << " kB\n"
         << "Peak resident set size:   " << vmhwm << " kB\n\n";
}

/*
 * Implementation of the Simple Pairwise Comparison method with alphabetical
 * order of k-mer iteration
 *
 * k: The length of the k-mer
 * d: The maximum edit distance allowed
 */
void doPairwiseCmp( const int k, const int d )
{
    unsigned long int kmerSpaceSize = 1ul << (2 * k);
    vector<unsigned long int> MIS;
    bool isCovered = false;
    
    cerr << "\nList of independent nodes: " << endl;
    for ( unsigned long int i = 0; i < kmerSpaceSize; ++i )
    {
        for ( const unsigned long int &j : MIS )
        {
            if ( hammingDist(i, j, k, d) <= d )
            {
                isCovered = true;
                break;
            }
        }

        if ( isCovered )
        {
            isCovered = false;
            continue;
        }

        printKmer( i, k );
        cerr << ' ';
        MIS.push_back( i );
    }

    cerr << "\nThe graph has an independent set of size " << MIS.size() 
         << ".\n\n";
    reportPerformance();
}

/*
 * Implementation of the Simple Pairwise Comparison method with random order of
 * k-mer iteration
 *
 * k: The length of the k-mer
 * d: The maximum edit distance allowed
 */
void doRandPairwiseCmp( const int k, const int d )
{
    sleep(1);
    srand( time(nullptr) );
    unsigned long int kmerSpaceSize = 1ul << (2 * k);
    vector<unsigned long int> MIS;
    bool isCovered = false;
    unordered_set<unsigned long int> space;
    for ( unsigned long int i = 0; i < kmerSpaceSize; ++i )
    {
        space.insert(i);
    }
    
    cerr << "\nList of independent nodes: " << endl;
    while ( !space.empty() )
    {
        auto it = space.begin();
        advance(it, rand() % space.size());
        unsigned long int kmer = *it;
        space.erase(it);
        
        for ( const unsigned long int &j : MIS )
        {
            if ( hammingDist(kmer, j, k, d) <= d )
            {
                isCovered = true;
                break;
            }
        }

        if ( isCovered )
        {
            isCovered = false;
            continue;
        }

        printKmer( kmer, k );
        cerr << ' ';
        MIS.push_back( kmer );
    }

    cerr << "\nThe graph has an independent set of size " << MIS.size() 
         << ".\n\n";
    reportPerformance();
}

/*
 * A data structure to store the mapping information for kmers. The array is 
 * chopped into subarrays to avoid failures of dynamic allocation.
 */
class MappingArray
{
private:
    char **aptrs;               // An array of pointers to subarrays
    unsigned long int size;     // The capacity of the whole array in elements
    unsigned long int num_subs; // Number of subarrays
    unsigned long int sub_size; // Size of a subarray in bytes

public:
    /*
     * Constructor
     *
     * s: capacity of the array
     */
    MappingArray( unsigned long int s )
    {
        sub_size = 1ul << 30;
        num_subs = s * 8 / sub_size; // Each element occupies 8 bytes.
        if ( (s * 8) % sub_size != 0 )
        {
            num_subs++;
        }
        aptrs = (char **) calloc( num_subs, sizeof(char *) );
        for (int i = 0; i < num_subs; ++i)
        {
            aptrs[i] = (char *) calloc( sub_size, 1 );
        }
        size = s;
    }

    /*
     * Destructor
     */
    ~MappingArray()
    {
        for (int i = 0; i < num_subs; ++i)
        {
            free( aptrs[i] );
        }
        free( aptrs );
    }

    /*
     * Overload [] operator to return the mapping of the kmer indexed by sub
     *
     * sub: The index of the element to be extract
     */
    unsigned long int operator[]( const unsigned long int &sub )
    {
        // Find the byte position where the required element is located
        // Use modulo to avoid index out of range
        unsigned long int bytePos = (sub % size) * 8;

        unsigned long int m = 0;
        memcpy( &m, &aptrs[bytePos/sub_size][bytePos%sub_size], 8);
        return m;
    }
    
    /*
     * Set the mapping for an element indexed by sub
     *
     * sub: The index of the element
     * m  : The mapping of the element
     */
    void setMap( const unsigned long int sub, unsigned long int m )
    {
        unsigned long int bytePos = (sub % size) * 8;
        memcpy( &aptrs[bytePos/sub_size][bytePos%sub_size], &m, 8);
    }
};

/*
 * Asks neighbors for possible mapping. Returns true if a feasible answer is 
 * found.
 *
 * enc    : The binary encoding of the k-mer
 * k      : The length of the k-mer
 * d      : The maximum edit distance allowed
 * mapping: The mapping array
 */
bool askNeighbors( const unsigned long int enc, const int k, const int d, 
                   MappingArray &mapping )
{
    // A set to store asked neighbors
    unordered_set<unsigned long int> asked;

    // A set to store checked possibilities
    unordered_set<unsigned long int> checked;

    for ( int j = 1; j <= k; ++j )
    {
        unsigned long int head = (enc >> (2 * j)) << (2 * j);
        unsigned long int tail = (enc << 1 << (63 - 2 * (j - 1))) >> 
                                 (63 - 2 * (j - 1)) >> 1;
        for ( unsigned long int l = 0; l < 4; ++l )
        {
            unsigned long int body = l << (2 * (j - 1));
            unsigned long int node = head + body + tail;
            if ( asked.emplace(node).second )
            {
                unsigned long int temp = mapping[node];
                if ( checked.emplace(temp).second && 
                     hammingDist(temp, enc, k, d) <= d )
                {
                    mapping.setMap(enc, temp);
                    return true;
                }
            }
        }
    }
    return false;
}

/*
 * Implementation of the heuristic method with alphabetical order of k-mer
 * iteration
 *
 * k: The length of the k-mer
 * d: The maximum edit distance allowed
 */
void doHeuristic( const int k, const int d )
{
    unsigned long int kmerSpaceSize = 1ul << (2 * k);
    vector<unsigned long int> MIS;
    vector<int> da;
    vector<int> dc;
    vector<int> dg;
    vector<int> dt;

    MIS.push_back( 0 );
    da.push_back( 0 );
    dc.push_back( k );
    dg.push_back( k );
    dt.push_back( k );
    MappingArray mapping(kmerSpaceSize / 4);

    cerr << "\nList of independent nodes: " << endl;
    printKmer( 0, k );
    cerr << ' ';
    bool isCovered = false;

    for (unsigned long int i = 1; i < kmerSpaceSize; ++i)
    {
        if ( askNeighbors(i, k, d, mapping) )
        {
            continue;
        }

        int ds[] = {k, k, k, k};
        unsigned long int temp_v = i;
        for (int j = 0; j < k; ++j)
        {
            ds[temp_v & 3]--;
            temp_v = temp_v >> 2;
        }

        for (unsigned long int j = 0; j < MIS.size(); ++j)
        {
            if ( abs(da[j] - ds[0]) > d ||
                 abs(dc[j] - ds[1]) > d ||
                 abs(dg[j] - ds[2]) > d ||
                 abs(dt[j] - ds[3]) > d )
            {
                continue;
            }
            if ( da[j] + ds[0] <= d ||
                 dc[j] + ds[1] <= d ||
                 dg[j] + ds[2] <= d ||
                 dt[j] + ds[3] <= d ||
                 hammingDist(i, MIS[j], k, d) <= d)
            {
                mapping.setMap(i, MIS[j]);
                isCovered = true;
                break;
            }
        }

        if ( isCovered )
        {
            isCovered = false;
            continue;
        }

        printKmer( i, k );
        cerr << ' ';
        MIS.push_back( i );
        da.push_back( ds[0] );
        dc.push_back( ds[1] );
        dg.push_back( ds[2] );
        dt.push_back( ds[3] );
        mapping.setMap( i, i );
    }

    cerr << "\nThe graph has an independent set of size " << MIS.size() 
         << ".\n\n";
    reportPerformance();
}

/*
 * Implementation of the heuristic method with random order of k-mer iteration
 *
 * k: The length of the k-mer
 * d: The maximum edit distance allowed
 */
void doRandHeuristic( const int k, const int d )
{
    sleep(1);
    srand( time(nullptr) );
    unsigned long int kmerSpaceSize = 1ul << (2 * k);
    unsigned long int kmer = rand() % kmerSpaceSize;
    vector<unsigned long int> MIS;
    vector<int> da;
    vector<int> dc;
    vector<int> dg;
    vector<int> dt;
    MappingArray mapping(kmerSpaceSize / 4);
    for (unsigned long int i = 0; i < kmerSpaceSize; ++i)
    {
        mapping.setMap(i, kmer);
    }
    unordered_set<unsigned long int> space;
    for (unsigned long int i = 0; i < kmerSpaceSize; ++i)
    {
        space.insert(i);
    }
    space.erase(kmer);
    MIS.push_back(kmer);
    int ds[] = {k, k, k, k};
    unsigned long int temp_v = kmer;
    for (int j = 0; j < k; ++j)
    {
        ds[temp_v & 3]--;
        temp_v = temp_v >> 2;
    }
    da.push_back(ds[0]);
    dc.push_back(ds[1]);
    dg.push_back(ds[2]);
    dt.push_back(ds[3]);

    cerr << "\nList of independent nodes: " << endl;
    printKmer( kmer, k );
    cerr << ' ';
    bool isCovered = false;
    while ( !space.empty() )
    {
        auto it = space.begin();
        advance(it, rand() % space.size());
        unsigned long int kmer = *it;
        space.erase(it);
        
        if ( askNeighbors(kmer, k, d, mapping) )
        {
            continue;
        }

        ds[0] = k;
        ds[1] = k;
        ds[2] = k;
        ds[3] = k;
        temp_v = kmer;
        for (int j = 0; j < k; ++j)
        {
            ds[temp_v & 3]--;
            temp_v = temp_v >> 2;
        }

        for (unsigned long int j = 0; j < MIS.size(); ++j)
        {
            if ( abs(da[j] - ds[0]) > d ||
                 abs(dc[j] - ds[1]) > d ||
                 abs(dg[j] - ds[2]) > d ||
                 abs(dt[j] - ds[3]) > d )
            {
                continue;
            }
            if ( da[j] + ds[0] <= d ||
                 dc[j] + ds[1] <= d ||
                 dg[j] + ds[2] <= d ||
                 dt[j] + ds[3] <= d ||
                 hammingDist(kmer, MIS[j], k, d) <= d)
            {
                mapping.setMap(kmer, MIS[j]);
                isCovered = true;
                break;
            }
        }

        if ( isCovered )
        {
            isCovered = false;
            continue;
        }

        printKmer( kmer, k );
        cerr << ' ';
        MIS.push_back( kmer );
        da.push_back( ds[0] );
        dc.push_back( ds[1] );
        dg.push_back( ds[2] );
        dt.push_back( ds[3] );
        mapping.setMap( kmer, kmer );
    }

    cerr << "\nThe graph has an independent set of size " << MIS.size() 
         << ".\n\n";
    reportPerformance();
}

/*
 * A class for the dist array used in BFS. The array is chopped into subarrays
 * to avoid failures of dynamic allocation.
 */
class DistArray
{
private:
    char **aptrs;               // An array of pointers to subarrays
    unsigned long int size;     // The capacity of the whole array in elements
    unsigned long int num_subs; // Number of subarrays
    unsigned long int sub_size; // Size of a subarray in bytes
    unsigned int max_d;         // Maximum edit distance allowed

public:
    /*
     * Constructor
     *
     * s: total number of elements
     */
    DistArray( unsigned long int s, unsigned int d )
    {
        sub_size = 1;
        sub_size = sub_size << 30;
        num_subs = s / 4 / sub_size; // Each element occupies 1/4 bytes.
        if ( (s / 4) % sub_size != 0 )
        {
            num_subs++;
        }
        aptrs = (char **) calloc( num_subs, sizeof(char *) );
        for (int i = 0; i < num_subs; ++i)
        {
            aptrs[i] = (char *) calloc( sub_size, 1 );
        }
        size = s;
        max_d = d;
    }

    /*
     * Destructor
     */
    ~DistArray()
    {
        for (int i = 0; i < num_subs; ++i)
        {
            free( aptrs[i] );
        }
        free( aptrs );
    }

    /*
     * Overload [] operator to return the dist value indexed by sub
     *
     * sub: The index of the element to be extract
     */
    unsigned int operator[]( const unsigned long int &sub )
    {
        // Find the bit offset
        int offset = (sub % 4) * 2;

        // Find the byte in which the required element is located
        unsigned long int bytePos = sub / 4;

        unsigned int dist;
        memcpy( &dist, &aptrs[bytePos/sub_size][bytePos%sub_size], 1 );

        dist = (dist << (24 + offset)) >> 30;
        return dist + (max_d + 1)/2 - 1;
    }
    
    /*
     * Set the dist value for an element indexed by sub
     *
     * sub : The index of the element
     * dist: The dist value of the element
     */
    void setDist( const unsigned long int sub, int dist )
    {
        dist = dist + 1 - (max_d + 1)/2;
        if (dist < 2)
        {
            dist = 1;
        }

        // Find the byte in which the required element is located
        unsigned long int bytePos = sub / 4;

        // Find the bit offset
        int offset = (sub % 4) * 2;

        unsigned int tempByte;
        memcpy( &tempByte, &aptrs[bytePos/sub_size][bytePos%sub_size], 1 );
        unsigned int head = tempByte >> (8 - offset);
        head = head << (8 - offset);
        unsigned int tail = tempByte << 24 << (offset + 2);
        tail = tail >> 24 >> (offset + 2);
        tempByte = tempByte & (head | tail);
        unsigned int body = dist;
        body = body << (6 - offset);
        tempByte = tempByte | body;
        memcpy( &aptrs[bytePos/sub_size][bytePos%sub_size], &tempByte, 1 );
    }
};

/*
 * Gets neighbors of a vertex
 *
 * enc: The binary encoding of the k-mer
 * k  : The length of the k-mer
 * n  : An unordered_set to hold the neighbors
 */
void getNeighbor( unsigned long int enc, int k, 
                  unordered_set<unsigned long int> &n )
{
    // Handle substitution
    for (int j = 1; j <= k; ++j)
    {
        unsigned long int head = (enc >> (2 * j)) << (2 * j);
        unsigned long int tail = (enc << 1 << (63 - 2 * (j - 1))) >> 
                                 (63 - 2 * (j - 1)) >> 1;
        for (unsigned long int l = 0; l < 4; ++l)
        {
            unsigned long int body = l << (2 * (j - 1));
            unsigned long int node = head + body + tail;
            n.emplace( node );
        }
    }
    n.erase(enc);
}

/*
 * Implementation of the BFS method with alphabetical order of k-mer iteration
 *
 * k: The length of the k-mer
 * d: The maximum edit distance allowed
 */
void doBFS( const int k, const int d )
{
    // Initialize dist array for BFS
    unsigned long int num_kmers = 1ul << (2 * k);
    DistArray dist_kmer(num_kmers, d);

    unsigned long int num_indep_nodes = 0;
    cerr << "\nList of independent nodes: " << endl;
    for ( unsigned long int i = 0; i < num_kmers; ++i )
    {
        if ( dist_kmer[i] != (d + 1)/2 - 1 )
        {
            continue;
        }
        printKmer(i, k);
        cerr << ' ';
        num_indep_nodes++;

        // Do BFS
        deque<unsigned long int> Q; // Initialize an empty queue
        Q.push_back(i);

        // Keep the search history
        unordered_map<unsigned long int, unsigned int> hist;
        hist.emplace( i, 0 );

        dist_kmer.setDist(i, 0);
        while ( !Q.empty() )
        {
            auto q0 = hist.find( Q[0] );
            if ( q0->second + 1 > d )
            {
                break;
            }
            unordered_set<unsigned long int> neighbors;
            getNeighbor( Q[0], k, neighbors );

            for ( auto &j : neighbors )
            {
                if ( hist.find(j) != hist.end() )
                {
                    continue;
                }
                unsigned int targetDist;
                targetDist = dist_kmer[j];
                if ( targetDist == (d + 1)/2 - 1 ||
                     (targetDist != (d + 1)/2 &&
                      targetDist > q0->second + 1) )
                {
                    dist_kmer.setDist( j, q0->second + 1 );
                    Q.push_back(j);
                    hist.emplace( j, q0->second + 1 );
                }
            }
            Q.pop_front();
        }
    }

    cerr << "\nThe graph has an independent set of size " << num_indep_nodes 
         << ".\n\n";
    reportPerformance();
}

/*
 * Implementation of the BFS method with random order of k-mer iteration
 *
 * k: The length of the k-mer
 * d: The maximum edit distance allowed
 */
void doRandBFS( const int k, const int d )
{
    sleep(1);
    srand( time(nullptr) );

    // Initialize dist array for BFS
    unsigned long int num_kmers = 1ul << (2 * k);
    DistArray dist_kmer(num_kmers, d);
    unordered_set<unsigned long int> space;
    for ( unsigned long int i = 0; i < num_kmers; ++i )
    {
        space.insert(i);
    }

    unsigned long int num_indep_nodes = 0;
    cerr << "\nList of independent nodes: " << endl;
    while ( !space.empty() )
    {
        auto it = space.begin();
        advance(it, rand() % space.size());
        unsigned long int i = *it;
        space.erase(it);

        if ( dist_kmer[i] != (d + 1)/2 - 1 )
        {
            continue;
        }
        printKmer(i, k);
        cerr << ' ';
        num_indep_nodes++;

        // Do BFS
        deque<unsigned long int> Q; // Initialize an empty queue
        Q.push_back(i);

        // Keep the search history
        unordered_map<unsigned long int, unsigned int> hist;
        hist.emplace( i, 0 );

        dist_kmer.setDist(i, 0);
        while ( !Q.empty() )
        {
            auto q0 = hist.find( Q[0] );
            if ( q0->second + 1 > d )
            {
                break;
            }
            unordered_set<unsigned long int> neighbors;
            getNeighbor( Q[0], k, neighbors );

            for ( auto &j : neighbors )
            {
                if ( hist.find(j) != hist.end() )
                {
                    continue;
                }
                unsigned int targetDist;
                targetDist = dist_kmer[j];
                if ( targetDist == (d + 1)/2 - 1 ||
                     (targetDist != (d + 1)/2 &&
                      targetDist > q0->second + 1) )
                {
                    dist_kmer.setDist( j, q0->second + 1 );
                    Q.push_back(j);
                    hist.emplace( j, q0->second + 1 );
                }
            }
            Q.pop_front();
        }
    }

    cerr << "\nThe graph has an independent set of size " << num_indep_nodes 
         << ".\n\n";
    reportPerformance();
}

int main()
{
    cerr << "This program is used to find a MIS in a k-mer space. Valid inputs"
         << " for the integer parameters k and d should satisfy 2<=k<=30 and"
         << " 1<=d<k.\n";

    int k;
    int d;
    int method;
    int random;
    cerr << "Please enter k: ";
    cin >> k;
    cerr << k << endl;
    cerr << "Plesae enter d: ";
    cin >> d;
    cerr << d << endl;
    cerr << "Please choose an approach. Notice that the BFS approach does not " 
         << "support d>5. Enter 1 for Simple Greedy, 2 for Improved Greedy, "
         << "or 3 for BFS: ";
    cin >> method;
    cerr << method << endl;
    cerr << "The iteration order of k-mers affects the resulting MIS size and "
         << "the performance of the program.\n"
         << "Please choose the iteration order of k-mers. Enter 1 for random "
         << "order or 2 for alphabetical order: ";
    cin >> random;
    cerr << random << endl;

    if ( random == 2 )
    {
        if ( method == 1 )
        {
            doPairwiseCmp( k, d );
        }
        else if ( method == 2 )
        {
            doHeuristic( k, d );
        }
        else if ( method == 3 )
        {
            doBFS( k, d );
        }
    }
    else if ( random == 1 )
    {
        if ( method == 1 )
        {
            doRandPairwiseCmp( k, d );
        }
        else if ( method == 2 )
        {
            doRandHeuristic( k, d );
        }
        else if ( method == 3 )
        {
            doRandBFS( k, d );
        }
    }
    return 0;
}
