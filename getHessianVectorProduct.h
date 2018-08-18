#ifndef getHessianVectorProduct_h
#define getHessianVectorProduct_h

#include<Eigen/Core>

Eigen::VectorXd getHessianVectorProduct(
    std::vector <double> x,
    std::vector <double> dx,
    std::vector <double> W,
    std::vector <double> S,
    std::vector <double> designVar,
    Eigen::VectorXd vecW);
#endif


