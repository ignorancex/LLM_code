/// @file traffic.cpp
/// @brief Nagel-Schreckenberg Traffic Model

#include <vector>
#include <iostream>
#include <sstream>
#include <random>
#include <algorithm>
#include "parameters.h"
#include "npywriter.h"

// Global pseudo random number generator based off of C++'s minstd_rand
class PRNG: public std::minstd_rand
{
  public:
    PRNG(result_type x = default_seed) : std::minstd_rand(x) {}    
    void discard(result_type z) {
        // Faster forward routine for this multiplicative linear
        // congruent generator, O(log z) instead of O(z)
        // compute az = multiplier^z mod modulus:
        result_type b = multiplier, az = 1, x;
        while (z > 0) {
            if (z % 2)
                az= (az*b) % modulus;
            b = (b*b) % modulus;
            z >>= 1;
        }
        // apply to current state:
        std::stringstream s;
        s << (*this); // get state as a string
        s >> x;       // convert back to unsigned long
        seed((x * az) % modulus); // set forwarded state
    }
};

PRNG* engine;
std::uniform_real_distribution<float>* uniformr;  ///< uniform real distribution
std::uniform_int_distribution<int>*    uniformi;  ///< uniform integer distribution


/// @brief Function to initialize internal variables need to make the
///        pseudo-random number generation work.
///
/// @param seed Random seed for the Marsenne-Twister (in).
///
/// @return void
void random_initialize(long seed)
{
    engine = new PRNG(seed);
    uniformr = new std::uniform_real_distribution<float>(0.0,1.0);
    uniformi = new std::uniform_int_distribution<int>();
    // note: uniformi cannot be initialized yet because we do not know the max integer
}


/// @brief Function to create a draw a random float between 0 (inclusive)
///        and 1 (exclusive).
///
/// @return random float
float random_next_float()
{
    return (*uniformr)(*engine);
}


/// @brief Function to create a draw a random int between 0 and max (inclusive)
///
/// @param max Maximum integer that could be returned (in)
///
/// @return random int
int random_next_int(int max)
{
    uniformi->param(std::uniform_int_distribution<int>::param_type{0,max});
    return (*uniformi)(*engine);
}


/// @brief Function to randomly place N agents on a grid of L points without overlap.
///
/// @param  N    number of agents to place (in)
/// @param  L    number of grid points (in)
///
/// @return vector of integers with the positions of the agents, in order of increasing magnitude
std::vector<int> random_placement(int N, int L)
{
    std::vector<int> result(N);
    for (int i=0; i<N; i++)
        result[i] = random_next_int(L-N-1); // allow space to shift agents
    std::sort(result.begin(), result.end());
    for (int i=0; i<N; i++)
        result[i] += i; // guaranteed to separate agents
    return result;
}


/// @brief Function to randomly draw velocities for agents on a grid
///        of L points without overlap, and such that the agents do
///        not bump into the next agents.
///
/// @param  N     number of agents to place (in)
/// @param  L     number of grid points (in)
/// @param  vmax  maximal possible velocity (in)
/// @param  x     positions of the agents; must be in order of increasing magnitude (in)
///
/// @return vector of integers with velocities of the agents
std::vector<int> random_velocity(int N, int L, int vmax, const std::vector<int>& x)
{
    std::vector<int> result(N);
    for (int i=0; i<N; i++) {
        result[i] = random_next_int(vmax);
        int d = (i<N-1)?(x[i+1]-x[i]):(x[0]-x[N-1]);
        if (d < 0)
            d += L;
        if (result[i] > std::max(0,d-1))
            result[i] = std::max(0,d-1);
    }
    return result;
}


/// @brief Update the positions and velocities of cars according to
///        the Nagel-Schreckenberg model.
///
/// @param x       vector with integer positions (in/out)
/// @param v       vector with integer velocities (in/out)
/// @param params  a struct with all parmeters, see parameters.h (in)
///
/// @return void
void one_time_step(std::vector<int>& x, std::vector<int>& v, const Parameters& params)
{
    using std::min;
    using std::max;

    const int N = min(x.size(),v.size()); ///< number of cars to process
    int x0 = x[0];           ///< store because otherwise overwritten

    for (int i = 0; i < N; i++) {

        int d; // distance to next car
        // cars remain ordered by postion, so next car is just i+1, except for the last car
        if (i < N-1)
            d = x[i+1] - x[i];
        else
            d = x0 - x[i] + params.L;

        // velocity update according to the model
        v[i] = min(min(v[i]+1, params.vmax),max(0,d-1));

        if (random_next_float() < params.p)
            if (v[i] > 0)
                v[i]--;
    }
    
    for (int i = 0; i < N; i++) {
        // position update according to the velocity
        x[i] += v[i];
    }
}


/// @brief main driver of the simulation
int main(int argc, char* argv[])
{
    // take parameter file name from command line or use defaults
    Parameters params = default_parameters();
    if (argc<2) 
        std::cout << " WARNING: No parameter file specified, using default values.\n";
    else
        params = read_parameters(argv[1]);
    // report parameters to console
    print_parameters(params);

    // prepare the random number generator and use it to randomly set
    // positions on the cars and their velocities.
    random_initialize(params.seed);
    std::vector<int> x = random_placement(params.N, params.L);
    std::vector<int> v = random_velocity(params.N, params.L, params.vmax, x);

    // prepare files into which the results will be written
    npy::Writer wrtr;
    if (params.per > 0)
        wrtr = npy::writer_open(params.outputprefix, params.L);

    for (int t = 0; t <= params.T; t++) {       
        // write current state very 'per' steps
        if ( (params.per > 0) && ((t % params.per) == 0)) 
            npy::writer_add(wrtr, t, x, v); 
        // update state
        one_time_step(x, v, params);        
    }

    // close files
    if (params.per > 0)
        npy::writer_close(wrtr);
}
