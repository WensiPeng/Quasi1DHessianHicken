#ifndef exactgmres_h
#define exactgmres_h
#include<Eigen/Core>
#include<Eigen/Sparse>
using namespace Eigen;
MatrixXd exactGMRES(SparseMatrix <double> A, MatrixXd B);
#endif
