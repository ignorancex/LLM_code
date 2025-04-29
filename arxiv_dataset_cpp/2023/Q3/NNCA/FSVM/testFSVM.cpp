#include "svm.hpp"
using namespace std;
#include <fstream>

double dec_fun(const ptsnD& x1){
    double temp = 0.0;
    for(size_t i=0; i<NDIM; ++i){
        temp += (x1.x[i] * x1.x[i]);
    }
    return exp(-1.0 * sqrt(temp));
}

double dec_fun2d(const ptsnD& x1) {
    double temp = 0.0;
    double a1 = 1.1;
    double b1 = 0.8;
    double a2 = 0.6;
    double b2 = 0.3;
    double a3 = (a1+a2)/2;
    double b3 = (b1+b2)/2;

    if (x1.x[0] * x1.x[0]/a3/a3 + x1.x[1] * x1.x[1]/b3/b3 < 1) {
      return 1; // class 1
    }
    else {
      return -1; // class 2
    }
}

void set_random_Nodes(std::vector<ptsnD> &points) {
  for (size_t i = 0; i < N; i++) {
    ptsnD temp;
    for (size_t j = 0; j < NDIM; j++) {
      temp.x[j] = (double(rand())/double(RAND_MAX)-0.5)*2;
    }
    points.push_back(temp);
  }
}

void set_random_Nodes2d(std::vector<ptsnD> &points) {
  double a = 1.1;
  double b = 0.8;
  for (size_t i = 0; i < N/2; i++) {
    ptsnD temp;
    double theta = double(rand())/double(RAND_MAX)*2*PI;
    double err1 = (double(rand())/double(RAND_MAX)-0.5)/0.5*0.2;
    double err2 = (double(rand())/double(RAND_MAX)-0.5)/0.5*0.2;//-0.2---0.2
    if (i%20 == 0) {
      temp.x[0] = (a-0.4)*cos(theta)+err1;
      temp.x[1] = (b-0.4)*sin(theta)+err2;
    }
    else {
      temp.x[0] = a*cos(theta)+err1;
      temp.x[1] = b*sin(theta)+err2;
    }
    points.push_back(temp);
  }
  a = 0.6;
  b = 0.3;
  for (size_t i = 0; i < N/2; i++) {
    ptsnD temp;
    double theta = double(rand())/double(RAND_MAX)*2*PI;
    double err1 = (double(rand())/double(RAND_MAX)-0.5)/0.5*0.2;
    double err2 = (double(rand())/double(RAND_MAX)-0.5)/0.5*0.2;
    if (i%20 == 0) {
      temp.x[0] = (a+0.4)*cos(theta)+err1;
      temp.x[1] = (b+0.4)*sin(theta)+err2;
    }
    else {
      temp.x[0] = a*cos(theta)+err1;
      temp.x[1] = b*sin(theta)+err2;
    }
    points.push_back(temp);
  }
}

void print_vec_of_ptsnD(std::vector<ptsnD> x, std::string filename) {
  std::ofstream outFile(filename);
  for(size_t j = 0; j < x.size(); j++){
    ptsnD temp = x[j];
    for (size_t i = 0; i < NDIM; i++) {
      outFile << temp.x[i] << " ";
    }
    outFile << "\n";
  }
}

void print_vec_of_ptsnD(std::vector<ptsnD>* x, std::string filename) {
  std::ofstream outFile(filename);
  for(size_t j = 0; j < x->size(); j++){
    ptsnD temp = x->at(j);
    for (size_t i = 0; i < NDIM; i++) {
      outFile << temp.x[i] << " ";
    }
    outFile << "\n";
  }
}

void print_vec_of_ptsnD(std::vector<double>* x, std::string filename) {
  std::ofstream outFile(filename);
  for(size_t j = 0; j < x->size(); j++){
    outFile << x->at(j) << "\n";
  }
}

void print_vec_of_ptsnD(std::vector<int>* x, std::string filename) {
  std::ofstream outFile(filename);
  for(size_t j = 0; j < x->size(); j++){
    outFile << x->at(j) << "\n";
  }
}

// std::vector<double> cheb_nodes(double a, double b, int n)
// {
//     std::vector<double> X(n);
//     double l, l1, param;
//     l = 0.5 * (a + b);
//     l1 = 0.5 * (b - a);
//     for (int k = 0; k < n; k++)
//     {
//         param = (double)(k + 0.5) / n;
//         X[k] = l - l1 * cos(param * 3.1412);
//     }
//     return X;
// }

// std::vector<double> random_nodes(double a, double b, int n)
// {
//     std::vector<double> X(n);
//     Eigen::VectorXd Points = Eigen::VectorXd::Random(n);
//     double l, l1;
//     l = 0.5 * (a + b);
//     l1 = 0.5 * (b - a);
//     for (int k = 0; k < n; k++)
//     {
//         //param = (double)(k + 0.5) / n;
//         X[k] = l - l1 * Points(k);
//     }
//     return X;
// }

int main(int argc, char* argv[]) {
  numPoints = atoi(argv[1]); // in one dimension
  N = pow(numPoints, NDIM);
// N    = atoi(argv[1]);
// NDIM = atoi(argv[2]);
int Qchoice;
std::cout << "N: " << N << std::endl;
std::cout << "NDIM: " << NDIM << std::endl;
int nLevels = 4;
std::vector<ptsnD> gridPoints;

#ifdef USE_Matern
set_random_Nodes(gridPoints);
Qchoice = 0;
#endif

#ifdef USE_Gaussian
set_random_Nodes2d(gridPoints);
Qchoice = 1;
#endif

std::vector<ptsnD> data_set = gridPoints;

int class1=0, class2=0;
std::vector<ptsnD> data_class1;
std::vector<ptsnD> data_class2;

std::vector<ptsnD> train_class1_data;
std::vector<ptsnD> train_class2_data;

std::vector<ptsnD> test_class1_data;
std::vector<ptsnD> test_class2_data;

#ifdef USE_Gaussian
for(auto& itr :data_set){
    double temp = dec_fun2d(itr);
    if(temp > 0){
        data_class1.push_back(itr);
        ++class1;
    }
    else{
        data_class2.push_back(itr);
        ++class2;
    }
}
#endif

#ifdef USE_Matern
for(auto& itr :data_set){
    double temp = dec_fun(itr);
    #ifdef USE_DIM2
      double cflag = 0.6;
    #elif USE_DIM4
      double cflag = 0.4;
    #endif
    if(temp < itr.x[NDIM-2]+cflag){ // 2D
        data_class1.push_back(itr);
        ++class1;
    }
    else{
        data_class2.push_back(itr);
        ++class2;
    }
}
#endif

int len1 = data_class1.size(), len2 = data_class2.size();
for(int i=0; i<=floor(0.85 * len1); ++i){
    train_class1_data.push_back(data_class1[i]);
}
for(int i=0; i<=floor(0.85 * len2); ++i){
    train_class2_data.push_back(data_class2[i]);
}

for(int i=ceil(0.85 * len1); i<len1; ++i){
    test_class1_data.push_back(data_class1[i]);
}
for(int i=ceil(0.85 * len2); i<len2; ++i){
    test_class2_data.push_back(data_class2[i]);
}


/////////////// SVM ////////////////
svm* T = new svm(0.5, 0.05, 0.5, Qchoice, nLevels);//(0.5, 0.05, 0.1)
// std::cout << "N: " << N << std::endl;
std::cout << "train_class1_data: " << train_class1_data.size() << std::endl;
std::cout << "train_class2_data: " << train_class2_data.size() << std::endl;
std::cout << "test_class1_data: " << test_class1_data.size() << std::endl;
std::cout << "test_class2_data: " << test_class2_data.size() << std::endl;
T->set_training_params(train_class1_data,train_class2_data);
// std::cout << "N: " << N << std::endl;
// T->assemble_hodlr();
double start = omp_get_wtime();
// std::cout << "before train" << std::endl;
// std::cout << "N: " << N << std::endl;
T->train();
double train_time = omp_get_wtime() - start;
// std::cout << "here" << std::endl;
T->test(test_class1_data,test_class2_data);

T->print_details();


cout << "Training Class1 size: " << train_class1_data.size() << endl;
cout << "Training Class2 size: " << train_class2_data.size() << endl;

cout << "Test Class1 size: " << test_class1_data.size() << endl;
cout << "Test Class2 size: " << test_class2_data.size() << endl;
cout << "Training time: " << train_time << endl;
cout << "Training time per iteration: " << train_time/T->iter << endl;

cout << "---------------------------------------------" << endl;

print_vec_of_ptsnD(T->xs, "matlab/xs.txt");
print_vec_of_ptsnD(T->xs_in, "matlab/xs_in.txt");
print_vec_of_ptsnD(T->ys, "matlab/ys.txt");
print_vec_of_ptsnD(T->ys_in, "matlab/ys_in.txt");
print_vec_of_ptsnD(T->alpha_s, "matlab/alpha_s.txt");
print_vec_of_ptsnD(T->alpha_s_in, "matlab/alpha_s_in.txt");

std::ofstream outFile("matlab/b.txt");
outFile << T->b << " ";

print_vec_of_ptsnD(train_class1_data, "matlab/train_class1_data.txt");
print_vec_of_ptsnD(train_class2_data, "matlab/train_class2_data.txt");
print_vec_of_ptsnD(test_class1_data, "matlab/test_class1_data.txt");
print_vec_of_ptsnD(test_class2_data, "matlab/test_class2_data.txt");

delete T;

////////////////////////////////////
}
