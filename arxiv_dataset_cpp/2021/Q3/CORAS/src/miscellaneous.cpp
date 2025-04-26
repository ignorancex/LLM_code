#include "miscellaneous.hpp"

#include <fstream>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

GSLMultiminFunctionWrapper::GSLMultiminFunctionWrapper(const std::size_t dimension, const std::function<double(const gsl_vector *minimizerArguments)> &func)
    : Function(func)
{
  n = dimension; // initialize the members of the parent gsl_multimin_function
  f = &GSLMultiminFunctionWrapper::invoke;
  params = this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

double GSLMultiminFunctionWrapper::invoke(const gsl_vector *minimizerArguments, void *parameters)
{
  return static_cast<GSLMultiminFunctionWrapper *>(parameters)->Function(minimizerArguments);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// global

void invert_matrix_svd(gsl_matrix *matrix, gsl_matrix *inverse)
{
  const std::size_t dimension = matrix->size1;

  gsl_matrix *A = gsl_matrix_alloc(dimension, dimension);
  gsl_matrix_memcpy(A, matrix);

  gsl_matrix *V = gsl_matrix_alloc(dimension, dimension);
  gsl_matrix *SV = gsl_matrix_calloc(dimension, dimension);
  gsl_vector *workspace = gsl_vector_alloc(dimension);
  gsl_vector *S = gsl_vector_alloc(dimension);

  gsl_linalg_SV_decomp(A, V, S, workspace);

  for (std::size_t i = 0; i < dimension; ++i)
  {
    const double S_i = gsl_vector_get(S, i);

    gsl_matrix_set(SV, i, i, 1.0 / S_i);
  }

  gsl_matrix *SVAt = gsl_matrix_alloc(dimension, dimension);

  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, SV, A, 0.0, SVAt);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, SVAt, 0.0, inverse);

  gsl_matrix_free(A);
  gsl_matrix_free(V);
  gsl_matrix_free(SV);
  gsl_vector_free(workspace);
  gsl_vector_free(S);
  gsl_matrix_free(SVAt);
}

bool file_exists(const std::string &fileName)
{
  std::ifstream file(fileName, std::ios::in);

  const bool fileExists = file.good();

  file.close();

  return fileExists;
}