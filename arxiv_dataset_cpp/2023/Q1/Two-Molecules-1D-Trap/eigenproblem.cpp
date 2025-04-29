#include "eigenproblem.h"

pair <double*, double*> MKLDiagonalization(Matrix& matrix, unsigned int matrix_size, bool compute_eigenvectors)
{
	char jobz;
	if (compute_eigenvectors == true)
	{
		jobz = 'V';
	}
	else
	{
		jobz = 'N';
	}

	/* Locals */
	MKL_INT n = matrix_size, lda = matrix_size, info;

	/* Local arrays */
	double* a = NULL;   			// Pointer to int, initialize to nothing.
	double* w = NULL;
	a = new double[(int64_t)matrix_size * matrix_size];  			// Allocate n ints and save ptr in a.
	w = new double[matrix_size];

	/* Matrix to array */
	for (unsigned int i = 0; i < matrix_size; i++)
	{
		for (unsigned int j = 0; j < matrix_size; j++)
		{
			a [matrix_size * i + j] = matrix(i, j);
		}
	}

	printf("\nStarting to solve the eigenproblem.\n");

	/* Solve eigenproblem */
	info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, jobz, 'U', n, a, lda, w); // 'U' stores the upper triangular part of a matrix

	/* Check for convergence */
	if (info > 0) 
	{
		printf("The algorithm failed to compute eigenvalues.\n");
		exit(1);
	}

	if (compute_eigenvectors == true)
	{
		return make_pair(w, a);
	}
	else
	{
		double dummy_array[1];
		return make_pair(w, dummy_array);
	}
}

void printEigenvalues(const char* desc, MKL_INT n, double* w)
{
	MKL_INT j;
	printf("\n %s\n", desc);
	for (j = 0; j < n; j++) {
		printf(" %3.4f", w[j]); // 3 miejsca przed i 4 miejsca po przecinku
	}
	printf("\n");
}

void printEigenvectors(const char* desc, MKL_INT n, double* v)
{
	MKL_INT i, j;
	//printf("\n %s\n", desc);
	for (i = 0; i < n; i++) 
	{
		j = 0;
		while (j < n)
		{
			printf(" %6.2f", v[i * n + j]);
			j++;
		}
		printf("\n");
	}
}