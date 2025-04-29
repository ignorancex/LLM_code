#include <iostream>
#include <iomanip>
#include <ctime>

#include <linearc.h>

using namespace std;

/////////////////////////////////////////////////////
// Array Class //////////////////////////////////////
/////////////////////////////////////////////////////
Array::Array(){
    this->ptr = nullptr;
    this->SIZE = 0;
}

Array::Array(int SIZE){
    init(SIZE);
}

Array::Array(double* ptr,int SIZE){
    this->ptr = ptr;
    this->SIZE = SIZE;
}

void Array::init(int SIZE){
    double* arr = new double[SIZE];
    for (int i=0;i<SIZE;i++) arr[i] = 0;
    this->ptr = arr;
    this->SIZE = SIZE;
}

/////////////////////////////////////////////////////
// CLASS FUNCTIONS //////////////////////////////////

int Array::size() const{
    return this->SIZE;
}

void Array::print(bool LINEAR, ostream& stream){

    if (LINEAR){
        stream << "[ ";
        for (int i=0;i<this->size();i++){
            stream << this->ptr[i];
            if (i!=this->size()-1) stream << ", ";
        }
        stream << " ]" << endl;
    }

    else {
        for (int i=0;i<this->size();i++){
            stream << this->ptr[i] << endl;
        }
    }
}

void Array::print(bool Linear){
    print(Linear,cout);
}

void Array::print(ostream& stream){
    print(true,stream);
}

void Array::print(){
    print(true,cout);
}

/////////////////////////////////////////////////////
// Operator OVERLOADING ////////////////////////////

// This implements periodic boundaries on the array
double& Array::operator[](int i){
    if (i<this->size() && i >= 0) return this->ptr[i];
    else if (i>=this->size()) return this->ptr[i%this->size()];
    else return this->ptr[this->size()+i%(this->size())];
}

double& Array::operator[](int i) const{
    if (i<this->size() && i >= 0) return this->ptr[i];
    else if (i>=this->size()) return this->ptr[i%this->size()];
    else return this->ptr[this->size()+i%(this->size())];
}

// Multiplication by a constant
Array Array::operator*(const double& C) const{
    Array A(new double[this->SIZE],this->SIZE);
    for (int i=0; i<this->size(); i++) A[i] = C*ptr[i];

    return A;
}

// Allowing for reverse multiplication of arrays with scalars
Array operator*(double C,Array A){
    return A*C;
}


// Dot product
Array Array::operator*(const Array& arr) const{
    // Check if they have the same length
    if (this->size() != arr.size()) throw "Dot product of non-equal arrays";
    
    Array A(new double[this->SIZE],this->SIZE);
    for (int i=0; i<this->size(); i++) A[i] = ptr[i]*arr[i];

    return A;
}

// Division by a constant
Array Array::operator/(const double& C) const{
    if (!C) throw "Division of array by zero";
    return *(this)*(1/C);
}


// Array Addition
Array Array::operator+(const Array& arr) const{
    // Check if they have the same length
    if (this->size() != arr.size()) throw "Addition of non-equal arrays";
    
    Array A(new double[this->SIZE],this->SIZE);
    for (int i=0; i<this->size(); i++) A[i] = ptr[i] + arr[i];

    return A;
}

// Array subtraction
Array Array::operator-(const Array& arr) const{
    // Check if they have the same length
    if (this->size() != arr.size()) throw "Addition of non-equal arrays";
    
    Array A(new double[this->SIZE],this->SIZE);
    for (int i=0; i<this->size(); i++) A[i] = ptr[i] - arr[i];

    return A;
}

Array& Array::operator=(const Array& arr){
    if (arr.size() != this->SIZE){
        this->SIZE = arr.size();
        init(SIZE);
    }

    for (int i=0; i<SIZE; i++){
        ptr[i] = arr[i];
    }

    return *this;
}


// Array Destructor
Array::~Array(){
    if (ptr!=nullptr)delete[] ptr;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////
// Matrix Class /////////////////////////////////////
/////////////////////////////////////////////////////

// Constructors
Matrix::Matrix(){
    this->ptr = nullptr;
    this->ROW_SIZE = 0;
    this->COL_SIZE = 0;
}

// Initializes a Matrix (ROW,COL) to zeroes
Matrix::Matrix(int ROW_SIZE, int COL_SIZE){
    init(ROW_SIZE,COL_SIZE);
}

Matrix::Matrix(double** ptr,int ROW_SIZE, int COL_SIZE){
    this->ptr = ptr;
    this->ROW_SIZE = ROW_SIZE;
    this->COL_SIZE = COL_SIZE;
}

void Matrix::init(int ROW_SIZE, int COL_SIZE){
    double** mat = new double*[ROW_SIZE];
    for (int i=0;i<ROW_SIZE;i++){
        mat[i] = new double[COL_SIZE];
        for (int j=0;j<COL_SIZE;j++){
            mat[i][j] = 0;
        }
    }

    this->ptr = mat;
    this->ROW_SIZE = ROW_SIZE;
    this->COL_SIZE = COL_SIZE;
}

/////////////////////////////////////////////////////
// CLASS FUNCTIONS //////////////////////////////////

// To get the determinant of a square matrix
double Matrix::det(){
    if (!is_square()) throw "Attempted determinant of non-square matrix";
    Matrix M = lu_dcmp(*this);
    double dd = 1;
    for (int i=0;i<M.col_size();i++){
        dd *= M[i][i];
    }

    return dd;
}

// True if it's a square matrix
bool Matrix::is_square() const{
    return ROW_SIZE==COL_SIZE;
}


// Prints the matrix through the output stream given 
// i.e. M.print(cout) etc.
void Matrix::print(std::ostream& stream){
    for (int i=0;i<ROW_SIZE;i++){
        for (int j=0;j<COL_SIZE;j++){
            stream << this->ptr[i][j]; 
            if (j<COL_SIZE-1) stream << ", ";
        }
        stream << "\n";
    }
}

// Prints the matrix in the console using cout
void Matrix::print(){
    print(cout);
}

// Returns the row size
int Matrix::row_size() const{
    return ROW_SIZE;
}

// Returns the column size
int Matrix::col_size() const{
    return COL_SIZE;
}

double** Matrix::get_ptr() const{
    return ptr;
}

/////////////////////////////////////////////////////
// Operator OVERLOADING /////////////////////////////
double*& Matrix::operator[](int i){
    return ptr[i];
}

double*& Matrix::operator[](int i) const{
    return ptr[i];
}

// Multiplies matrix by a double
Matrix Matrix::operator*(const double& X) const{

    // First create a new matrix object identical to M
    double** ptr = new double*[this->row_size()];
    Matrix mat(ptr,this->row_size(),this->col_size());

    // Copy M to mat 
    for (int i=0;i<mat.row_size();i++){
        ptr[i] = new double[mat.col_size()];
        for (int j=0; j < mat.col_size();j++){
            mat[i][j] = this->ptr[i][j]*X;
        }
    }

    return mat;
}

// Define the orher way for multiplication by a scalar
Matrix operator*(double C, Matrix& M){
    return M*C;
}

// Divides matrix by a double
Matrix Matrix::operator/(const double& X) const{
    if (!X) throw "Matrix division by 0";
    return this->operator*(1/X);
}

// Adds a matrix to a matrix
Matrix Matrix::operator+(const Matrix& M) const{

    if (!((M.col_size() == col_size()) && (M.row_size() == row_size()))){
        throw "Adding matrices with unequal dimensions";
    }

    // First create a new matrix object identical to M
    double** ptr = new double*[this->row_size()];
    Matrix mat(ptr,this->row_size(),this->col_size());

    // Do the addition
    for (int i=0;i<mat.row_size();i++){
        ptr[i] = new double[mat.col_size()];
        for (int j=0; j < mat.col_size();j++){
            mat[i][j] = this->ptr[i][j]+M[i][j];
        }
    }

    return mat;
}

// Subtracting matrices
Matrix Matrix::operator-(const Matrix &M) const{
    if (!((M.col_size() == col_size()) && (M.row_size() == row_size()))){
        throw "Adding matrices with unequal dimensions";
    }

    // First create a new matrix object identical to M
    double** ptr = new double*[this->row_size()];
    Matrix mat(ptr,this->row_size(),this->col_size());

    // Do the addition
    for (int i=0;i<mat.row_size();i++){
        ptr[i] = new double[mat.col_size()];
        for (int j=0; j < mat.col_size();j++){
            mat[i][j] = this->ptr[i][j]-M[i][j];
        }
    }

    return mat;
}


// Matrix multiplication
Matrix Matrix::operator*(const Matrix& M) const{
    if (!((this->col_size() == M.row_size()))){
        throw "Multiplying incompatible matrices";
    }

    // First create a new matrix object identical to M
    double** ptr = new double*[this->row_size()];
    Matrix mat(ptr,this->row_size(),M.col_size());

    // Do the product
    for (int i=0;i<mat.row_size();i++){
        ptr[i] = new double[mat.col_size()];
        for (int j=0; j < mat.col_size();j++){
            double sum = 0;
            for (int n=0;n<this->col_size();n++) sum += this->ptr[i][n]*M[n][j];
            mat[i][j] = sum;
        }
    }

    return mat;
}

// Matrix multiplication with vector
Array Matrix::operator*(const Array& M) const{
    if (!((this->col_size() == M.size()))){
        throw "Multiplying incompatible matrices";
    }

    // First create a new matrix object identical to M
    double* ptr = new double[this->row_size()];
    Array mat(ptr,this->row_size());

    // Do the product
    for (int i=0;i<mat.size();i++){
        double sum = 0;
        for (int n=0;n<this->col_size();n++) sum += this->ptr[i][n]*M[n];
        mat[i] = sum;
    }

    return mat;
}


// Adds one matrix to another
Matrix& Matrix::operator+=(const Matrix& M){
    if (!((M.col_size() == col_size()) && (M.row_size() == row_size()))){
        throw "Adding matrices with unequal dimensions";
    }

    for (int i=0;i<this->row_size();i++){
        for (int j=0; j < this->col_size();j++){
            this->ptr[i][j]+=M[i][j];
        }
    }

    return *this;
}


// Subtracts one matrix from another
Matrix& Matrix::operator-=(const Matrix& M){
    if (!((M.col_size() == col_size()) && (M.row_size() == row_size()))){
        throw "Adding matrices with unequal dimensions";
    }

    for (int i=0;i<this->row_size();i++){
        for (int j=0; j < this->col_size();j++){
            this->ptr[i][j]-=M[i][j];
        }
    }

    return *this;
}


// Kind of important assignment operator (N^2) : (
Matrix& Matrix::operator=(const Matrix& M){
    // Take all the elements of M and assign them onto the current one
    
    this->ROW_SIZE = M.row_size();
    this->COL_SIZE = M.col_size();
    init(ROW_SIZE,COL_SIZE);

    for (int i=0;i<ROW_SIZE; i++){
        for (int j=0; j<COL_SIZE; j++){
            ptr[i][j] = M[i][j];
        }
    }

    return *this;
}


// Destructor
Matrix::~Matrix(){
    if (ptr != nullptr){
        for(int i=0; i<this->row_size(); i++){
            if(ptr[i] != nullptr) delete[] ptr[i];
        }

        delete[] ptr;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////
// LU Decomposition /////////////////////////////////
/////////////////////////////////////////////////////
// Uses Crout's Algorithm to compute LU decomposition in O(n^3) for nxn
/*
   The returned matrix is a factorisation of two matrices: M = AB
   but placed on the same matrix, like so:

        |1  0  0  0|
   A =  |a  1  0  0|
        |a  a  1  0|
        |a  a  a  1|

        |b  b  b  b|
   B =  |0  b  b  b|
        |0  0  b  b|
        |0  0  0  b|
 
   So M becomes:

        |b  b  b  b|
   M =  |a  b  b  b|
        |a  a  b  b|
        |a  a  a  b|
*/
void lu_dcmp_replace(Matrix& M){

    for(int j=0; j<M.col_size(); j++){

        // Forward
        for(int i=0; i<=j; i++){
            double sum = 0;
            for (int n=0;n<i;n++){
                sum += M[i][n] * M[n][j];
            }

            M[i][j] -= sum;
        }

        // Backward
        for(int i=j+1; i<M.col_size(); i++){
            double sum = 0;
            for (int n=0;n<j;n++){
                sum += M[i][n] * M[n][j];
            }

            M[i][j] = (M[i][j] - sum)/(M[j][j]);
        }
    }
}


//Decomposes M into a new matrix without replacing M
Matrix lu_dcmp(const Matrix& M){
    // First create a new matrix object identical to M
    double** ptr = new double*[M.row_size()];
    Matrix mat(ptr,M.row_size(),M.col_size());

    // Copy M to mat 
    for (int i=0;i<mat.row_size();i++){
        ptr[i] = new double[M.col_size()];
        for (int j=0; j < mat.col_size();j++){
            mat[i][j] = M[i][j];
        }
    }

    // Replace mat with it's decomposition
    lu_dcmp_replace(mat);
    return mat;
}

//Decomposes M into a new matrix and places it to N
void lu_dcmp(const Matrix& M, Matrix& N){
    for(int j=0; j<M.col_size(); j++){

        // Forward
        for(int i=0; i<=j; i++){
            double sum = 0;
            for (int n=0;n<i;n++){
                sum += M[i][n] * M[n][j];
            }

            N[i][j] = M[i][j] - sum;
        }

        // Backward
        for(int i=j+1; i<M.col_size(); i++){
            double sum = 0;
            for (int n=0;n<j;n++){
                sum += M[i][n] * M[n][j];
            }

            N[i][j] = (N[i][j] - sum)/(M[j][j]);
        }
    }
}





//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////
// Various Solvers //////////////////////////////////
/////////////////////////////////////////////////////

// Solve equation A X = B where X,B are either matrices or vectors, using LU decomposition.
Array solve(const Matrix& A,const Array& b,bool decomposed){
    
    // If the matrix is not in LU decompose it to LU
    Matrix M(A.col_size(),A.row_size());
    if (!decomposed)lu_dcmp(A,M);
    else M = A;
    
    // Create the output vector
    Array x(b.size());

    // Forward Substitution
    for(int i=0;i<x.size();i++){
        double sum = 0;
        for (int j=0;j<i-1;j++) sum+=M[i][j]*x[j];
        x[i] = b[i] - sum;
    }

    //backward subsitution
    for(int i=x.size()-1;i>=0;i--){
        double sum = 0;
        for (int j=i+1;j<x.size();j++) sum+=M[i][j]*x[j];
        x[i] = (x[i]-sum)/M[i][i];
    }

    return x;
}

// Assuming the matrix is not decomposed
Array solve(const Matrix& A,Array& b){
    return solve(A,b,false);
}


// Solve equation A X = B where X,B are either matrices or vectors, using LU decomposition.
void solve(const Matrix& A, const Array& b, Array& x,bool decomposed){
    
    // If the matrix is not in LU decompose it to LU
    Matrix M(A.col_size(),A.row_size());
    if (!decomposed)lu_dcmp(A,M);
    else M = A;

    // Forward Substitution
    for(int i=0;i<x.size();i++){
        double sum = 0;
        for (int j=0;j<i-1;j++) sum+=M[i][j]*x[j];
        x[i] = b[i] - sum;
    }

    //backward subsitution
    for(int i=x.size()-1;i>=0;i--){
        double sum = 0;
        for (int j=i+1;j<x.size();j++) sum+=M[i][j]*x[j];
        x[i] = (x[i]-sum)/M[i][i];
    }
}

// Assuming hte matrix is not decomposed
void solve(const Matrix& A, const Array& b, Array& x){
    solve(A,b,x,false);
}


// To do the same thing but with X, B matrices:
Matrix solve(const Matrix& A, const Matrix& b, bool decomposed){
    
    // If the matrix is not in LU decompose it to LU
    // Matrix M = !decomposed ? lu_dcmp(A) : A; 
    Matrix M(A.col_size(),A.row_size());
    if (!decomposed)lu_dcmp(A,M);
    else M = A;

    //Create the output matrix
    Matrix x(M.row_size(),b.col_size()); 

    //Solve the equivalent vecotr equtions
    Array xx(M.row_size());
    for (int i=0; i < b.col_size(); i++){
        for (int j=0; j < xx.size(); j++) xx[j] = b[j][i];
        Array tmp = solve(M,xx,true);
        for (int j=0; j < xx.size(); j++)x[j][i] = tmp[j];
    }

    return x;
}


// To do the same thing but with X, B matrices assuming decomposed = false:
Matrix solve(const Matrix& A, const Matrix& b){
    return solve(A,b,false);
}


// To do the same thing but with X, B matrices:
void solve(const Matrix& A,const Matrix& b, Matrix& x, bool decomposed){
    
    // If the matrix is not in LU decompose it to LU
    if(!decomposed){
        Matrix M(A.col_size(),A.row_size());
        if (!decomposed)lu_dcmp(A,M);
        else M = A;

        //Solve the equivalent vector equtions
        Array xx(M.col_size());
        for (int i=0; i < b.col_size(); i++){
            for (int j=0; j < xx.size(); j++) xx[j] = b[j][i];
            Array tmp(xx.size());
            solve(M,xx,tmp,true);
            for (int j=0; j < xx.size(); j++)x[j][i] = tmp[j];
        }
        return;
    }

    // if it is decpomposed already
    else {
        Matrix M(A.col_size(),A.row_size());
        if (!decomposed)lu_dcmp(A,M);
        else M = A;

        //Solve the equivalent vector equations
        Array xx(A.col_size());
        for (int i=0; i < b.col_size(); i++){
            for (int j=0; j < xx.size(); j++) xx[j] = b[j][i];
            Array tmp(xx.size());
            solve(A,xx,tmp,true);



            // Forward Substitution
            for(int j=0;j<x.size();j++){
                double sum = 0;
                for (int k=0;k<j-1;k++) sum+=M[j][k]*x[k];
                x[i] = b[j][i] - sum;
            }

            //backward subsitution
            for(int i=x.size()-1;i>=0;i--){
                double sum = 0;
                for (int j=i+1;j<x.size();j++) sum+=M[i][j]*x[j];
                x[i] = (x[i]-sum)/M[i][i];
            }
            



            for (int j=0; j < xx.size(); j++)x[j][i] = tmp[j];
        }
    }
}


// To do the same thing but with X, B matrices assuming decomposed = false:
void solve(const Matrix& A,const Matrix& b, Matrix& x){
    solve(A,b,x,false);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////
// Tridiagonal Solver ///////////////////////////////
/////////////////////////////////////////////////////

// Solves a tridiagonal system of equations in O(N) time
// Specifically we solve equation Mx = b where:
//      - A is the upper diagonal
//      - B is the lower diagonal
//      - D is the diagonal 
//      - b is the equal vector
Array tridiag(const Array& A, const Array& B, const Array& D, const Array& b){

    // We do two passes, a forward and a backward pass
    // Forward pass
    double* gamma = new double[A.size()];
    double* beta  = new double[D.size()];

    if(!D[0]) throw "Tridiag division by 0";
    gamma[0] = A[0]/D[0];
    beta[0]  = b[0]/D[0];
    for (int i=1; i<D.size(); i++){
        if(!(D[i] - B[i-1]*gamma[i-1])) throw "Tridiag division by 0";

        if (i<A.size()) gamma[i] = A[i]/(D[i] - B[i-1]*gamma[i-1]);
        beta[i] = (b[i]-beta[i-1]*B[i-1])/(D[i] - B[i-1]*gamma[i-1]);
}

    // Backward Pass
    Array x(D.size());
    x[x.size()-1] = beta[x.size()-1];
    for (int i=x.size()-2; i>=0;i--){
        x[i] = beta[i] - gamma[i] * x[i+1];
    }

    // Return the final vector
    return x;
}

// Same thing with return in input
void tridiag(const Array& A, const Array& B, const Array& D, const Array& b, Array& x){

    // We do two passes, a forward and a backward pass
    // Forward pass
    double* gamma = new double[A.size()];
    double* beta  = new double[D.size()];

    if(!D[0]) throw "Tridiag division by 0";
    gamma[0] = A[0]/D[0];
    beta[0]  = b[0]/D[0];
    for (int i=1; i<D.size(); i++){
        if(!(D[i] - B[i-1]*gamma[i-1])) throw "Tridiag division by 0";

        if (i<A.size()) gamma[i] = A[i]/(D[i] - B[i-1]*gamma[i-1]);
        beta[i] = (b[i]-beta[i-1]*B[i-1])/(D[i] - B[i-1]*gamma[i-1]);
}

    // Backward Pass
    x[x.size()-1] = beta[x.size()-1];
    for (int i=x.size()-2; i>=0;i--){
        x[i] = beta[i] - gamma[i] * x[i+1];
    }
}

// Solves tridiagonal matrix equation Mx = b in O(n) 
Array tridiag(const Matrix& M, const Array& b){
    // Check for the matrix being square
    if(!M.is_square()) throw "Tridiagonal solving of non-square matrix";
    
    // obtain the arrays for the tridiagonal matrix
    Array D(M.row_size()), A(M.row_size()-1), B(M.row_size()-1);
    for (int i=0;i<M.row_size();i++){
        if(i<A.size()){
            A[i] = M[i][i+1];
            B[i] = M[i+1][i];
        } 
        D[i] = M[i][i];
    }

    return tridiag(A,B,D,b);
}


// Solves tridiagonal matrix equation Mx = b in O(n) 
void tridiag(const Matrix& M, const Array& b,Array& x){
    // Check for the matrix being square
    if(!M.is_square()) throw "Tridiagonal solving of non-square matrix";
    
    // obtain the arrays for the tridiagonal matrix
    Array D(M.row_size()), A(M.row_size()-1), B(M.row_size()-1);
    for (int i=0;i<M.row_size();i++){
        if(i<A.size()){
            A[i] = M[i][i+1];
            B[i] = M[i+1][i];
        } 
        D[i] = M[i][i];
    }

    tridiag(A,B,D,b,x);
}






//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////
// Block Tridiagonal Solver /////////////////////////
/////////////////////////////////////////////////////

// Solves a block tridiagonal system
// We are slightly modifying thomas algorithm
Array* block_tridiag(const Matrix* A, int A_SIZE, const Matrix* B,int B_SIZE, const Matrix* D,int D_SIZE, const Array* b,int b_SIZE){

    Matrix* Gamma = new Matrix[A_SIZE];
    for (int i=0; i<A_SIZE; i++) Gamma[i].init(A[0].row_size(),A[0].col_size());
    Array* Beta = new Array[D_SIZE];
    for (int i=0; i<D_SIZE; i++) Beta[i].init(D[0].row_size());

    // Decompose the original matrix
    Matrix lu(D[0].row_size(),D[0].col_size());
    lu_dcmp(D[0],lu);
    solve(lu,A[0],Gamma[0],true);
    solve(lu,b[0],Beta[0],true);

    // Forward propagation
    for(int i=1; i<D_SIZE; i++){    
        // get the decomposition of the divisor
        lu_dcmp(D[i] - B[i-1]*Gamma[i-1],lu);

        // Calculate the Beta and Gammas
        if (i < A_SIZE)solve(lu,A[i],Gamma[i],true);
        Array arr = b[i]-(B[i-1]*Beta[i-1]);
        solve(lu,arr,Beta[i]);
    }

    // Backpropagation
    Array* x = new Array[D_SIZE];
    x[D_SIZE-1] = Beta[D_SIZE-1];
    for (int i=D_SIZE-2; i>=0;i--){
        x[i] = Beta[i] - Gamma[i] * x[i+1];
    }

    return x;
}

// Solves a block tridiagonal system
// We are slightly modifying thomas algorithm
// This is replacing the values in the output array x 
void block_tridiag(const Matrix* A,int A_SIZE,const Matrix* B,int B_SIZE,const Matrix* D,int D_SIZE,const Array* b,int b_SIZE, Array* x){

    clock_t c_start = std::clock();
    std::cout << "\n\t\t Creating Gamma  ...  ";
    Matrix* Gamma = new Matrix[A_SIZE];
    for (int i=0; i<A_SIZE; i++) Gamma[i].init(A[0].row_size(),A[0].col_size());
    std::cout << "Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - c_start) / CLOCKS_PER_SEC << " ms" << std::endl; 

    c_start = std::clock();
    std::cout << "\n\t\t Solving for the first entry  ...  ";
    // Decompose the original matrix
    Matrix lu(D[0].row_size(),D[0].col_size());

    clock_t ct_start = std::clock();
    std::cout << "\n\t\t\t lu_dcmp  ...  ";
    lu_dcmp(D[0],lu);
    std::cout << "Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - ct_start) / CLOCKS_PER_SEC << " ms" << std::endl;

    ct_start = std::clock();
    std::cout << "\n\t\t\t solve  ...  "; // This is taking forever... FIX IT!
    solve(lu,A[0],Gamma[0],true);
    solve(lu,b[0],x[0],true);
    std::cout << "Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - ct_start) / CLOCKS_PER_SEC << " ms" << std::endl;

    std::cout << "\t\t Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;

    c_start = std::clock();
    std::cout << "\n\t\t Forward Propagation  ...  ";
    // Forward propagation
    for(int i=1; i<D_SIZE; i++){    
        // get the decomposition of the divisor
        lu_dcmp((D[i] - B[i-1]*Gamma[i-1]),lu);

        // Calculate the Beta and Gammas
        if (i < A_SIZE) solve(lu,A[i],Gamma[i],true);

        Array arr = (b[i]-(B[i-1]*x[i-1]));
        solve(lu,arr,x[i]);
    }
    std::cout << "Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;

    c_start = std::clock();
    std::cout << "\n\t\t Backpropagation  ...  ";
    // Backpropagation
    for (int i=D_SIZE-2; i>=0; i--){
        x[i] = x[i] - Gamma[i] * x[i+1];
    }
    std::cout << "Completed after: " << std::fixed << std::setprecision(2) << 1000. * (std::clock() - c_start) / CLOCKS_PER_SEC << " ms" << std::endl;
}

// Solves block tridiagonal system by passing the total matrix and the dimension size
Array block_tridiag(const Matrix& MAT,const Array&arr, int N, int M){
    // First we separate the matrix to the submatrices

    // Definition of component matrices
    Matrix* A = new Matrix[M-1];
    Matrix* B = new Matrix[M-1];
    Matrix* D = new Matrix[M];
    Array* b = new Array[M];

    for (int i=0;i<M;i++){
        A[i].init(N,N);
        B[i].init(N,N);
        D[i].init(N,N);
        b[i].init(N);
    }


    // To calculate the diagonals
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){

            // Main Diagonal Elements
            if (j>0)    D[i][j][j-1] = MAT[j+i*N][j-1+i*N];
            if (j<N-1)  D[i][j][j+1] = MAT[j+i*N][j+1+i*N];
            D[i][j][j] = MAT[j+i*N][j+i*N];

            // Upper diagonal elements
            if (i<M-1){
                A[i][j][j] = MAT[i*N+N+j][i*N+N+j]; 
            }

            // Lower diagonal elements
            if (i>0){
               B[i-1][j][j] = MAT[i*N-N+j][i*N-N+j];
            }

            // Equal to vector
            b[i][j] = arr[j+i*N];
        }
    }

    // Now solve
    Array* xx = block_tridiag(A,M-1,B,M-1,D,M,b,M);
    
    // Put everything to a vector
    Array x(N*M);
    for (int i=0;i<M;i++){
        for (int j=0;j<N;j++){
            x[i*N+j] = xx[i][j];
        }
    }

    return x;
}

// Uses tridiagonal solving on matrix MAT
// Finds the M by N dimension by assuming the tridiagonal shape
Array block_tridiag(const Matrix& MAT, const Array& arr){

    // Find M and N
    int N(-1),M;
    for (int i=2;i<MAT.col_size();i++){
        if (MAT[0][i]){
            N = i+1;
            break;
        }
    }

    if (N==-1) throw "Block tridiagonal of incorrectly formatted matrix";

    M = int(MAT.col_size()/N);

    return block_tridiag(MAT,arr,N,M);

}