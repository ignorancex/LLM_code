#include "matrix.h"
using namespace std;

// Constructor for Any Matrix
Matrix::Matrix(unsigned rowSize, unsigned colSize, double initial) 
{
	m_rowSize = rowSize;
	m_colSize = colSize;
	m_matrix.resize(rowSize);
	for (unsigned i = 0; i < m_matrix.size(); i++)
	{
		m_matrix[i].resize(colSize, initial);
	}
}

// Addition of Two Matrices
Matrix Matrix::operator+(Matrix& B) 
{
	Matrix sum(m_colSize, m_rowSize, 0.0);
	unsigned i, j;
	for (i = 0; i < m_rowSize; i++)
	{
		for (j = 0; j < m_colSize; j++)
		{
			sum(i, j) = this->m_matrix[i][j] + B(i, j);
		}
	}
	return sum;
}

// Subtraction of Two Matrices
Matrix Matrix::operator-(Matrix& B) 
{
	Matrix diff(m_colSize, m_rowSize, 0.0);
	unsigned i, j;
	for (i = 0; i < m_rowSize; i++)
	{
		for (j = 0; j < m_colSize; j++)
		{
			diff(i, j) = this->m_matrix[i][j] - B(i, j);
		}
	}

	return diff;
}

// Multiplication of Two Matrices
Matrix Matrix::operator*(Matrix& B)
{
	Matrix multip(m_rowSize, B.getCols(), 0.0);
	if (m_colSize == B.getRows())
	{
		unsigned i, j, k;
		double temp = 0.0;
		for (i = 0; i < m_rowSize; i++)
		{
			for (j = 0; j < B.getCols(); j++)
			{
				temp = 0.0;
				for (k = 0; k < m_colSize; k++)
				{
					temp += m_matrix[i][k] * B(k, j);
				}
				multip(i, j) = temp;
				//cout << multip(i,j) << " ";
			}
			//cout << endl;
		}
		return multip;
	}
	else
	{
		return multip;
	}
}

// Overloading += with Matrix Addition
void Matrix::operator+=(Matrix& B)
{
	unsigned i, j;
	for (i = 0; i < m_rowSize; i++)
	{
		for (j = 0; j < m_colSize; j++)
		{
			this->m_matrix[i][j] += B(i, j);
		}
	}
}

// Scalar Addition
Matrix Matrix::operator+(double scalar) 
{
	Matrix result(m_rowSize, m_colSize, 0.0);
	unsigned i, j;
	for (i = 0; i < m_rowSize; i++)
	{
		for (j = 0; j < m_colSize; j++)
		{
			result(i, j) = this->m_matrix[i][j] + scalar;
		}
	}
	return result;
}

// Scalar Subtraction
Matrix Matrix::operator-(double scalar) 
{
	Matrix result(m_rowSize, m_colSize, 0.0);
	unsigned i, j;
	for (i = 0; i < m_rowSize; i++)
	{
		for (j = 0; j < m_colSize; j++)
		{
			result(i, j) = this->m_matrix[i][j] - scalar;
		}
	}
	return result;
}

// Scalar Multiplication
Matrix Matrix::operator*(double scalar) 
{
	Matrix result(m_rowSize, m_colSize, 0.0);
	unsigned i, j;
	for (i = 0; i < m_rowSize; i++)
	{
		for (j = 0; j < m_colSize; j++)
		{
			result(i, j) = this->m_matrix[i][j] * scalar;
		}
	}
	return result;
}

// Scalar Division
Matrix Matrix::operator/(double scalar) 
{
	Matrix result(m_rowSize, m_colSize, 0.0);
	unsigned i, j;
	for (i = 0; i < m_rowSize; i++)
	{
		for (j = 0; j < m_colSize; j++)
		{
			result(i, j) = this->m_matrix[i][j] / scalar;
		}
	}
	return result;
}

// Returns value of given location when asked in the form A(x,y)
double& Matrix::operator()(const unsigned& rowNo, const unsigned& colNo)
{
	return this->m_matrix[rowNo][colNo];
}

// Returns value of given location of diagonal element when asked in the form A(x)
double& Matrix::operator()(const unsigned& rowNo)
{
	return this->m_matrix[rowNo][rowNo];
}

// No brainer - returns row #
unsigned Matrix::getRows() const
{
	return this->m_rowSize;
}

// returns col #
unsigned Matrix::getCols() const
{
	return this->m_colSize;
}

// Take any given matrices transpose and returns another matrix
Matrix Matrix::transpose()
{
	Matrix Transpose(m_colSize, m_rowSize, 0.0);
	for (unsigned i = 0; i < m_colSize; i++)
	{
		for (unsigned j = 0; j < m_rowSize; j++) {
			Transpose(i, j) = this->m_matrix[j][i];
		}
	}
	return Transpose;
}

// Prints the matrix beautifully
void Matrix::print(string text) const
{
	cout << text << endl;
	for (unsigned i = 0; i < m_rowSize; i++) {
		for (unsigned j = 0; j < m_colSize; j++) {
			cout << "[" << m_matrix[i][j] << "] ";
		}
		cout << endl;
	}
}