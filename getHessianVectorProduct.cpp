#include<iostream>
#include<iomanip>
#include<math.h>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Sparse>
#include"globals.h"
#include"convert.h"
#include"parametrization.h"
#include"residuald1.h"
#include"residuald2.h"
#include"objectiveDerivatives.h"
#include"BCDerivatives.h"
#include"directDifferentiation.h"
#include"flux.h"
#include"quasiOneD.h"
#include"grid.h"
#include"petscGMRES.h"

using namespace Eigen;

VectorXd getHessianVectorProduct(
    std::vector <double> x,
    std::vector <double> dx,
    std::vector <double> W,
    std::vector <double> S,
    std::vector <double> designVar,
    VectorXd vecW)
{
    // *************************************
    // Evaluate Area to Design Derivatives
    // *************************************
    // Evaluate dSdDes
    MatrixXd dSdDes(nx + 1, nDesVar);
    dSdDes = evaldSdDes(x, dx, designVar);//from parametrization.cpp line10
    // Evaluate ddSdDesdDes
    std::vector <MatrixXd> ddSdDesdDes(nx + 1); // by (nDes, nDes)
    ddSdDesdDes = evalddSdDesdDes(x, dx, designVar);//from parametrization.cpp line145
    
    // *************************************
    // Evaluate Objective Derivatives
    // *************************************
    // Evaluate dIcdW
    VectorXd dIcdW(3 * nx);
    dIcdW = evaldIcdW(W, dx);
    // Evaluate ddIcdWdW
    SparseMatrix <double> ddIcdWdW;
    ddIcdWdW = evaldIcdWdW(W, dx);
    // Evaluate ddIcdWdDes
    MatrixXd ddIcdWdDes(3 * nx, nDesVar);
    ddIcdWdDes = evalddIcdWdS() * dSdDes;//(3nx * nx+1)*(nx+1 * nDesVar)
    // Evaluate dIcdS
    VectorXd dIcdS(nx + 1);
    dIcdS = evaldIcdS();
    // Evaluate dIcdDes
    VectorXd dIcdDes(nDesVar);
    dIcdDes = dIcdS.transpose() * dSdDes;
    // Evaluate ddIcdDesdDes
    MatrixXd ddIcdDesdDes(nDesVar, nDesVar);
    ddIcdDesdDes = dSdDes.transpose() * evalddIcdSdS() * dSdDes;
    // *************************************
    // Evaluate Residual
    // *************************************
    //// Get Fluxes
    std::vector <double> Flux(3 * (nx + 1), 0);
    getFlux(Flux, W);
    // Evaluate dRdS
    MatrixXd dRdS(3 * nx, nx + 1);
    dRdS = evaldRdS(Flux, S, W); //Derivatives residual1.cpp line597
    // Evaluate dRdDes
    MatrixXd dRdDes(3 * nx, nDesVar);
    dRdDes = dRdS * dSdDes;
    // Evaluate dRdW
    std::vector <double> dt(nx, 1);
    SparseMatrix <double> dRdW;
    dRdW = evaldRdW(W, dx, dt, S);//residual1.cpp line38
    // Evaluate ddRdWdS
    std::vector <SparseMatrix <double> > ddRdWdS(3 * nx);// by (3 * nx, nx + 1)
    ddRdWdS = evalddRdWdS(W, S);//residual2 line680
    // Evaluate ddRdWdDes
    std::vector <MatrixXd> ddRdWdDes(3 * nx);// by (3 * nx, nDesVar)
    for(int Ri = 0; Ri < 3 * nx; Ri++)
    {
        ddRdWdDes[Ri] = ddRdWdS[Ri] * dSdDes;
    }
    // Evaluate ddRdWdW
    std::vector < SparseMatrix <double> > ddRdWdW(3 * nx);// by (3 * nx, 3 * nx)
    ddRdWdW = evalddRdWdW(W, S);
    
    // *************************************
    // Sparse LU of Jacobian Transpose dRdW
    // *************************************
    SparseLU <SparseMatrix <double>, COLAMDOrdering< int > > slusolver1;
    // SparseLU <matrixtype, ordering type>
    slusolver1.compute(-dRdW.transpose());
    if(slusolver1.info() != 0)
        std::cout<<"Factorization failed. Error: "<<slusolver1.info()<<std::endl;
    
    SparseLU <SparseMatrix <double>, COLAMDOrdering< int > > slusolver2;
    slusolver2.compute(-dRdW);
    if(slusolver2.info() != 0)
        std::cout<<"Factorization failed. Error: "<<slusolver2.info()<<std::endl;
    
    
    // *************************************
    // Solve for Adjoint 1 psi(1 Flow Eval)
    // *************************************
    VectorXd psi(3 * nx);
    psi = slusolver1.solve(dIcdW);
    
    // *************************************
    // Solve for Adjoint 2 z (1 Flow Eval)
    // *************************************
    VectorXd z(3 * nx);
    VectorXd dRdDestimesvecW(3 * nx);
    dRdDestimesvecW.setZero();
    dRdDestimesvecW = dRdDes * vecW;
    z = slusolver2.solve(dRdDestimesvecW);
    
    // *************************************
    // Solve for Adjoint 3 lambda (1 Flow Eval)
    // *************************************
    VectorXd lambda(3 * nx);
    VectorXd RHS(3 * nx);
    VectorXd dgTvecWdW(3 * nx);
    MatrixXd dsTdW(3 * nx, 3 * nx);
    dgTvecWdW = ddIcdWdDes * vecW;
    for(int Ri = 0; Ri < 3 * nx; Ri++)
    {
        dgTvecWdW += psi(Ri) * ddRdWdDes[Ri] * vecW;
    }
    dsTdW = ddIcdWdW;
    for(int Ri = 0; Ri < 3 * nx; Ri++)
    {
        dsTdW += psi(Ri) * ddRdWdW[Ri];
    }
    RHS = dgTvecWdW + dsTdW * z;
    lambda = slusolver1.solve(RHS);

    // *************************************
    // Evaluate Hw
    // *************************************
    VectorXd Hw(nDesVar);
    MatrixXd ddIcdWdS(3 * nx, nx + 1);
    ddIcdWdS = evalddIcdWdS();// 3nx * nx+1
    Hw.setZero();
    Hw = dSdDes.transpose() * ddIcdWdS.transpose() * z + dRdDes.transpose() * lambda
         + ddIcdDesdDes *vecW;
    for(int Si = 0; Si < nx + 1; Si++)
    {
        Hw += dIcdS(Si) * ddSdDesdDes[Si] * vecW;
        Hw += psi.dot(dRdS.col(Si)) * ddSdDesdDes[Si] * vecW;
    }
    for(int Ri = 0; Ri < 3 * nx; Ri++)
    {
        Hw += psi(Ri) * (ddRdWdDes[Ri].transpose() * z);
    }
    //std::cout<<"Hw = "<<std::endl;
    //std::cout<<Hw<<std::endl;
    //for some reasons ddRdDesdDes is zero???
//    std::cout<<"psi:"<<psi<<std::endl;
//    std::cout<<"z:"<<z<<std::endl;
//    std::cout<<"lambda:"<<lambda<<std::endl;
    return Hw;
}



