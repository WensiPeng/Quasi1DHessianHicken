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
#include"exactGMRES.h"

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
    dSdDes = evaldSdDes(x, dx, designVar);
    // Evaluate ddSdDesdDes
    std::vector <MatrixXd> ddSdDesdDes(nx + 1); // by (nDes, nDes)
    ddSdDesdDes = evalddSdDesdDes(x, dx, designVar);
    
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
    ddIcdWdDes = evalddIcdWdS() * dSdDes;
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
    // Evaluate Residual Derivatives
    // *************************************
    //// Get Fluxes
    std::vector <double> Flux(3 * (nx + 1), 0);
    getFlux(Flux, W);
    // Evaluate dRdS
    MatrixXd dRdS(3 * nx, nx + 1);
    dRdS = evaldRdS(Flux, S, W);
    // Evaluate dRdDes
    MatrixXd dRdDes(3 * nx, nDesVar);
    dRdDes = dRdS * dSdDes;
    // Evaluate dRdW
    std::vector <double> dt(nx, 1);
    SparseMatrix <double> dRdW;
    dRdW = evaldRdW(W, dx, dt, S);
    // Evaluate ddRdWdS
    std::vector <SparseMatrix <double> > ddRdWdS(3 * nx);// by (3 * nx, nx + 1)
    ddRdWdS = evalddRdWdS(W, S);
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
    // Solve for Adjoint (1 Flow Eval)
    // *************************************
    VectorXd psi(3 * nx);
    //SparseLU <SparseMatrix <double>, COLAMDOrdering< int > > slusolver1;
    //slusolver1.compute(-dRdW.transpose());
    //if(slusolver1.info() != 0)
    //   std::cout<<"Factorization failed. Error: "<<slusolver1.info()<<std::endl;
    //psi = slusolver1.solve(dIcdW);
    psi = exactGMRES(-dRdW.transpose(),dIcdW);
    //std::cout<<"psi: "<< psi.norm()<<std::endl;
    //std::cout<<"psi Error: "<<(psi - slusolver1.solve(dIcdW)).norm()<<std::endl;
    
    
    // *************************************
    // Evaluate dWdDes (nDesVar Flow Eval)
    // *************************************
    MatrixXd dWdDes(3 * nx, nDesVar);
    VectorXd dWdDestimesVecW(3 * nx);
    VectorXd RHS(3 * nx);
    RHS = dRdDes * vecW;
    //SparseLU <SparseMatrix <double>, COLAMDOrdering< int > > factdrdw;
    //factdrdw.compute(-dRdW);
    //if(factdrdw.info() != 0)
    //    std::cout<<"Factorization failed. Error: "<<factdrdw.info()<<std::endl;
    //if(exactHessian == 1)  dWdDes = factdrdw.solve(dRdDes);
    //else if(exactHessian < 0)
    //{
    // Iterative Solution of dWdDes (approximate to 1e-1)
    //dWdDes = solveGMRES(-dRdW,dRdDes);
    dWdDestimesVecW = exactGMRES(-dRdW,RHS);

    // Iterative Solution of dWdDes (accurate to 1e-15)
    //    dWdDes = exactGMRES(-dRdW,dRdDes);
    //std::cout<<"dWdDes ||r||/||b|| residual:"<<std::endl;
    //std::cout<<(-dRdW*dWdDes - dRdDes).norm()/dRdDes.norm()<<std::endl;
    /*
     // Direct Solution of dWdDes
     MatrixXd realdWdDes(3*nx,nDesVar);
     realdWdDes = factdrdw.solve(dRdDes);
     
     std::cout<<"Relative error of approximate dWdDes vs exact dWdDes:"<<std::endl;
     std::cout<<(realdWdDes - dWdDes).norm()/realdWdDes.norm()<<std::endl;
     for(int icol = 0; icol < nDesVar; icol++)
     std::cout << icol << "\t" << (realdWdDes.col(icol) - dWdDes.col(icol)).norm()/realdWdDes.col(icol).norm()<<std::endl;
     */
    //}
    
    
    // *************************************
    // Evaluate total derivative DDIcDDesDDes
    // *************************************
    VectorXd Hw(nDesVar);
    Hw.setZero();
    Hw = ddIcdDesdDes * vecW;
    Hw += dWdDestimesVecW.transpose() * ddIcdWdDes;
    Hw += (dWdDestimesVecW.transpose() * ddIcdWdDes).transpose();
    std::cout<<"HVP1 DDIcDDESDDES : \n"<<Hw<<std::endl;
    std::cout<<"Hw size : \n"<<(ddIcdWdW * dWdDestimesVecW).rows()<<"*"<<(ddIcdWdW * dWdDestimesVecW).cols()<<std::endl;
    Hw += ddIcdWdW * dWdDestimesVecW;  
    std::cout<<"HVP2 DDIcDDESDDES : \n"<<Hw<<std::endl;
    for(int Si = 0; Si < nx + 1; Si++)
    {
        Hw += dIcdS(Si) * ddSdDesdDes[Si] * vecW;
        Hw += psi.dot(dRdS.col(Si)) * ddSdDesdDes[Si] * vecW;
    }
    std::cout<<"HVP3 DDIcDDESDDES : \n"<<Hw<<std::endl;
    for(int Ri = 0; Ri < 3 * nx; Ri++)
    {
        //        DDIcDDesDDes += psi(Ri) * ddRdDesdDes; //ddRdDesdDes is 0
        Hw += psi(Ri) * (dWdDestimesVecW.transpose() * ddRdWdDes[Ri]);
        Hw += (psi(Ri) * (dWdDestimesVecW.transpose() * ddRdWdDes[Ri])).transpose();
        Hw += psi(Ri) * (dWdDestimesVecW.transpose() * ddRdWdW[Ri]);
    }

    // *************************************
    // adjoint-adjoint
    // *************************************
    /*
     
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
    /*SparseLU <SparseMatrix <double>, COLAMDOrdering< int > > slusolver1;
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
    //psi = slusolver1.solve(dIcdW);
    psi = exactGMRES(-dRdW.transpose(),dIcdW);
    //std::cout<<"psi"<<std::endl;
    // *************************************
    // Solve for Adjoint 2 z (1 Flow Eval)
    // *************************************
    VectorXd z(3 * nx);
    VectorXd dRdDestimesvecW(3 * nx);
    dRdDestimesvecW.setZero();
    dRdDestimesvecW = dRdDes * vecW;
    //z = slusolver2.solve(dRdDestimesvecW);
    z = exactGMRES(-dRdW,dRdDestimesvecW);
    //std::cout<<"z"<<std::endl;
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
    
    //lambda = slusolver1.solve(RHS);
    lambda = exactGMRES(-dRdW,RHS);
    //std::cout<<"lambda"<<std::endl;
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
    
    //for some reasons ddRdDesdDes is zero???
    //    std::cout<<"psi:"<<psi<<std::endl;
    //    std::cout<<"z:"<<z<<std::endl;
    //    std::cout<<"lambda:"<<lambda<<std::endl;
    */
    return Hw;
}



