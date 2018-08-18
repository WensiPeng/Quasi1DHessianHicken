#include "globals.h"
#include "quasiOneD.h"
#include "fitness.h"
#include "grid.h"
#include "adjoint.h"
#include "directDifferentiation.h"
#include <iostream>
#include <Eigen/Core>

VectorXd finiteD(
    std::vector <double> x,
    std::vector <double> dx,
    std::vector <double> S,
    std::vector <double> designVar,
    int gType,
    double h,
    double currentI,
    VectorXd &grad,
    VectorXd &consGrad);

void getGradient(int gType,
    double currentI,
    std::vector <double> x,
    std::vector <double> dx,
    std::vector <double> S,
    std::vector <double> W,
    std::vector <double> designVar,
    VectorXd &psi,
    VectorXd &consPsi,
    VectorXd &grad,
    VectorXd &consGrad)
{
    if(gType < 0)
    {
        double h = 1e-8;
        grad = finiteD(x, dx, S, designVar, gType, h, currentI, grad, consGrad);
    }
    else if(gType == 1)
    {
        adjoint(x, dx, S, W, designVar, psi, consPsi, grad, consGrad);
    }
    else if(gType == 2)
    {
        grad = directDifferentiation(x, dx, S, W, designVar);
    }
    return 0;
}


void finiteD(
    std::vector <double> x,
    std::vector <double> dx,
    std::vector <double> S,
    std::vector <double> designVar,
    int gType,
    double h,
    double currentI,
    VectorXd &grad,
    VectorXd &consGrad)
{
    VectorXd grad(nDesVar);
    std::vector <double> W(3 * nx, 0);
    std::vector <double> tempS(nx + 1);

    double I0, I1, I2, dh;

    std::vector <double> tempD(nDesVar);

    if(currentI < 0 && gType != -3)
    {
        quasiOneD(x, dx, S, W);
        I0 = evalFitness(dx, W);
    }
    else
    {
        I0 = currentI;
    }
    for(int i = 0; i < nDesVar; i++)
    {

        dh = designVar[i] * h;
        tempD = designVar;

        if(gType == -1) // FFD
        {
            tempD[i] += dh;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, dx, tempS, W);
            I1 = evalFitness(dx, W);
            grad[i] = (I1 - I0) / dh;
        }
        else if(gType == -2) // BFD
        {
            tempD[i] -= dh;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, dx, tempS, W);
            I2 = evalFitness(dx, W);
            grad[i] = (I0 - I2) / dh;
        }
        else if(gType == -3) // CFD
        {
            tempD[i] += dh;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, dx, tempS, W);
            I1 = evalFitness(dx, W);
            PL1 = TotalPressureLoss(W);
            tempD = designVar;

            tempD[i] -= dh;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, dx, tempS, W);
            I2 = evalFitness(dx, W);
            PL2 = TotalPressureLoss(W);
            grad[i] = (I1 - I2) / (2 * dh);
            consGrad[i] = (PL1 - PL2) / (2 * dh);
        }
    }
    std::cout<<"Gradient from FD: "<<std::endl;
    for(int i = 0; i < nDesVar; i++)
        std::cout<<grad[i]<<std::endl;
    
    std::cout<<"constraint Gradient from FD: "<<std::endl;
    for(int i = 0; i < nDesVar; i++)
        std::cout<<consGrad[i]<<std::endl;

    return 0;
}
