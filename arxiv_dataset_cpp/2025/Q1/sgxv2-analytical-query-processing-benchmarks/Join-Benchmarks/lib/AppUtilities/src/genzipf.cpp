/**
 * @file
 *
 * Generate Zipf-distributed input data.
 *
 * @author Jens Teubner <jens.teubner@inf.ethz.ch>
 *
 * (c) ETH Zurich, Systems Group
 *
 * $Id: genzipf.c 3017 2012-12-07 10:56:20Z bcagri $
 *
 * Overhauled by Adrian Lutsch
 */

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <random>

#include "genzipf.h"

#include <algorithm>

/**
 * Create an alphabet, an array of size @a size with randomly
 * permuted values 0..size-1.
 *
 * @param size alphabet size
 * @return an <code>item_t</code> array with @a size elements;
 *         contains values 0..size-1 in a random permutation; the
 *         return value is malloc'ed, don't forget to free it afterward.
 */
static uint32_t *
gen_alphabet(const unsigned int size) {
    /* allocate */
    const auto alphabet = static_cast<uint32_t *>(malloc(size * sizeof(uint32_t)));
    assert(alphabet);

    /* populate */
    for (unsigned int i = 0; i < size; i++)
        alphabet[i] = i + 1; /* don't let 0 be in the alphabet */

    /* permute */
    std::random_device rd;
    std::mt19937_64 gen{rd()};
    std::shuffle(alphabet, alphabet + size, gen);

    return alphabet;
}

/**
 * Generate a lookup table with the cumulated density function
 *
 * (This is derived from code originally written by Rene Mueller.)
 */
static double *
gen_zipf_lut(const double zipf_factor, const unsigned int alphabet_size) {
    const auto lut = static_cast<double *>(malloc(alphabet_size * sizeof(double))); /**< return value */
    assert(lut);

    /*
     * Compute scaling factor such that
     *
     *   sum (lut[i], i=1..alphabet_size) = 1.0
     *
     */
    double scaling_factor = 0.0;
    for (unsigned int i = 1; i <= alphabet_size; i++)
        scaling_factor += 1.0 / pow(i, zipf_factor);

    /*
     * Generate the lookup table
     */
    double sum = 0.0;
    for (unsigned int i = 1; i <= alphabet_size; i++) {
        sum += 1.0 / pow(i, zipf_factor);
        lut[i - 1] = sum / scaling_factor;
    }

    return lut;
}

/**
 * Generate a stream with Zipf-distributed content.
 */
item_t *
gen_zipf(const unsigned int stream_size, const unsigned int alphabet_size, const double zipf_factor, item_t **output) {
    item_t *ret;

    /* prepare stuff for Zipf generation */
    uint32_t *alphabet = gen_alphabet(alphabet_size);
    assert(alphabet);

    double *lut = gen_zipf_lut(zipf_factor, alphabet_size);
    assert(lut);

    if (*output == nullptr) {
        ret = static_cast<item_t *>(malloc(stream_size * sizeof(item_t)));
    } else {
        ret = *output;
    }
    assert(ret);

    std::random_device rd;
    std::mt19937_64 gen{rd()};
    std::uniform_real_distribution dist{0.0, 1.0};

    for (unsigned int i = 0; i < stream_size; i++) {
        /* take random number */
        const double r = dist(gen);

        /* binary search in lookup table to determine item */
        unsigned int left = 0;
        unsigned int right = alphabet_size - 1;
        unsigned int m;   /* middle between left and right */
        unsigned int pos; /* position to take */

        if (lut[0] >= r)
            pos = 0;
        else {
            while (right - left > 1) {
                m = (left + right) / 2;

                if (lut[m] < r)
                    left = m;
                else
                    right = m;
            }

            pos = right;
        }

        const auto dst = reinterpret_cast<uint32_t *>(&ret[i]);
        *dst = alphabet[pos];
        /* ret[i] = alphabet[pos]; */
    }

    free(lut);
    free(alphabet);

    *output = ret;

    return ret;
}
