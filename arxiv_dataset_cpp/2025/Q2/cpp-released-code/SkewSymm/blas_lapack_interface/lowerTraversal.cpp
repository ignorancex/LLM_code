#include "lowerTraversal.hpp"

void strict_lower_blk_traversal(INTE_TYPE *mat2vec, INTE_TYPE *vec2low, INTE_TYPE *vec2upp, INTE_TYPE dim)
{
    INTE_TYPE msize = dim * dim;
    INTE_TYPE lsize = ((dim + 1) * dim) / 2;
    INTE_TYPE bsize = dim / 2;

    INTE_TYPE vec_index = 0;
    for (INTE_TYPE j = 0; j < bsize; j++)
    {
        for (INTE_TYPE i = j + 1; i < bsize; i++)
        {
            // 2 x 2 block at [2i:(2i+1), 2j:(2j+1)], strict lower condition implies i > j.
            // The block is also in column major order, i.e., (1, 0) -> (2, 0) -> (m, 0) -> (2, 1) -> (2, 2) -> ... -> (m, m - 1), where m = bsize - 1
            mat2vec[2 * i + 2 * j * dim] = vec_index;
            vec2low[vec_index] = 2 * i + 2 * j * dim;
            vec2upp[vec_index] = 2 * j + 2 * i * dim;
            vec_index++;

            mat2vec[(2 * i + 1) + 2 * j * dim] = vec_index;
            vec2low[vec_index] = (2 * i + 1) + 2 * j * dim;
            vec2upp[vec_index] = 2 * j + (2 * i + 1) * dim;
            vec_index++;

            mat2vec[2 * i + (2 * j + 1) * dim] = vec_index;
            vec2low[vec_index] = 2 * i + (2 * j + 1) * dim;
            vec2upp[vec_index] = (2 * j + 1) + 2 * i * dim;
            vec_index++;

            mat2vec[(2 * i + 1) + (2 * j + 1) * dim] = vec_index;
            vec2low[vec_index] = (2 * i + 1) + (2 * j + 1) * dim;
            vec2upp[vec_index] = (2 * j + 1) + (2 * i + 1) * dim;
            vec_index++;
        }
    }

    // If there is a left-over row and column.
    if (2 * bsize != dim)
    {
        for (INTE_TYPE j = 0; j < bsize; j++)
        {
            // 1 x 2 block at [dim, 2j:(2j+1)], in the order (dim - 1, 0) -> (dim - 1, 1) -> ... -> (dim - 1, m), where m = bsize - 1.
            mat2vec[(dim - 1) + 2 * j * dim] = vec_index;
            vec2low[vec_index] = (dim - 1) + 2 * j * dim;
            vec2upp[vec_index] = 2 * j + (dim - 1) * dim;
            vec_index++;

            mat2vec[(dim - 1) + (2 * j + 1) * dim] = vec_index;
            vec2low[vec_index] = (dim - 1) + (2 * j + 1) * dim;
            vec2upp[vec_index] = (2 * j + 1) + (dim - 1) * dim;
            vec_index++;
        }
    }

    // Take care of left over entries in the lower left of the diagonal 2 x 2 blocks
    for (INTE_TYPE i = 0; i < bsize; i++)
    {
        // 2 x 2 diagonal blocks in [2i:(2i+1), 2i:(2i+1)] at entry [2i+1, 2i]
        mat2vec[(2 * i + 1) + 2 * i * dim] = vec_index;
        vec2low[vec_index] = (2 * i + 1) + 2 * i * dim;
        vec2upp[vec_index] = 2 * i + (2 * i + 1) * dim;
        vec_index++;
    }

    // Take care of the diagonals
    for (INTE_TYPE i = 0; i < dim; i++)
    {
        mat2vec[i + i * dim] = vec_index;
        vec2low[vec_index] = i + i * dim;
        vec2upp[vec_index] = i + i * dim;
        vec_index++;
    }
}

void strict_lower_col_traversal(INTE_TYPE *mat2vec, INTE_TYPE *vec2low, INTE_TYPE *vec2upp, INTE_TYPE dim)
{
    INTE_TYPE msize = dim * dim;
    INTE_TYPE lsize = ((dim + 1) * dim) / 2;

    INTE_TYPE vec_ind = 0;
    for (INTE_TYPE col_ind = 0; col_ind < dim; col_ind++)
        for (INTE_TYPE row_ind = col_ind + 1; row_ind < dim; row_ind++, vec_ind++)
        {
            mat2vec[row_ind + col_ind * dim] = vec_ind;
            mat2vec[col_ind + row_ind * dim] = -vec_ind;
            vec2low[vec_ind] = row_ind + col_ind * dim;
            vec2upp[vec_ind] = col_ind + row_ind * dim;
        }

    for (INTE_TYPE dia_ind = 0; dia_ind < dim; dia_ind++, vec_ind++)
    {
        mat2vec[dia_ind + dia_ind * dim] = vec_ind;
        vec2low[vec_ind] = dia_ind + dia_ind * dim;
        vec2upp[vec_ind] = dia_ind + dia_ind * dim;
    }
}