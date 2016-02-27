// Calculates the discrete costate fluxes
#include<iostream>
#include<math.h>
#include<vector>
#include<Eigen/Eigen>
#include <stdio.h>
#include <iomanip>
#include "adjoint.h"
#include"quasiOneD.h"
#include "globals.h"
#include "convert.h"
#include "flux.h"

using namespace Eigen;

VectorXd adjoint(
    std::vector <double> x, 
    std::vector <double> dx, 
    std::vector <double> S,
    std::vector <double> W,
    std::vector <double> &psi,
    std::vector <double> designVar)
{
    //Get Primitive Variables
    std::vector <double> rho(nx), u(nx), e(nx);
    std::vector <double> T(nx), p(nx), c(nx), Mach(nx);
    WtoP(W, rho, u, e, p, c, T); 
    
    // Evalutate dt and d(dt)dW
    std::vector <double> dt(nx, 1);

    // Evaluate dQdW
    std::vector <double> dQdW(3 * nx, 0);
    evaldQdW(dQdW, W, S);

    // Get Jacobians and Fluxes
    std::vector <double> Ap_list(nx * 3 * 3, 0), An_list(nx * 3 * 3, 0);
    std::vector <double> Flux(3 * (nx + 1), 0);
    if(FluxScheme == 0) StegerJac(W, Ap_list, An_list, Flux);
    if(FluxScheme == 1) ScalarJac(W, Ap_list, An_list);
        
    // Transposed Boundary Flux Jacobians
    std::vector <double> dBidWi(3 * 3, 0);
    std::vector <double> dBidWd(3 * 3, 0);
    std::vector <double> dBodWd(3 * 3, 0);
    std::vector <double> dBodWo(3 * 3, 0);

    // Build A matrix
    SparseMatrix <double> dRdWt, dRdW;
    SparseMatrix <double> matAFD, matAFD2;
    dRdW = evaldRdW(Ap_list, An_list, W, dQdW, dx, dt, S, u[0]/c[0]);
    matAFD2 = evaldRdW_FD(W, S, u[0]/c[0]);
    dRdWt = dRdW.transpose();
//  dRdWt = matAFD2.transpose();
    std::cout.precision(17);
//  std::cout<<dRdW<<std::endl;
//  std::cout<<matAFD2<<std::endl;
//  std::cout<<"(matAFD2 - dRdW).norm() / dRdW.norm():"<<std::endl;
//  std::cout<<(matAFD2 - dRdW).norm() / dRdW.norm()<<std::endl;

//  dRdWt.coeffRef(dRdWt.rows() - 3, dRdWt.cols() - 3) += 0.00001;
//  dRdWt.coeffRef(dRdWt.rows() - 2, dRdWt.cols() - 2) += 0.00001;
//  dRdWt.coeffRef(dRdWt.rows() - 1, dRdWt.cols() - 1) += 0.00001;

    // Build B matrix
    // Evaluate dIcdW
    VectorXd bvec(3 * nx);
    bvec = -evaldIcdW(W, dx);
//  std::cout<<"Vector B:"<<std::endl;
//  std::cout<<bvec<<std::endl;

    VectorXd psiV(3 * nx);
    psiV.setZero();
    // Solver type eig_solv
    // 0 = Sparse LU
    // 1 = Dense LU Full Piv
    // 2 = Sparse Iterative BiCGSTAB
    int eig_solv = 0;
    int directSolve = 1;
    if(directSolve == 1)
    {
        psiV = solveSparseAXB(dRdWt, bvec, eig_solv);
    }
    else
    {
        psiV = itSolve(dRdWt, bvec);
    }

    // If supersonic copy psi2 onto psi1 garbage
    if(u[0] > c[0])
    {
        psiV(0) = psiV(3);
        psiV(1) = psiV(4);
        psiV(2) = psiV(5);
    }

//  std::cout<<"Adjoint Result:"<<std::endl;
//  std::cout<<psiV<<std::endl;

    // Save Adjoint 
    FILE *Results;
    Results = fopen("Adjoint.dat", "w");
    fprintf(Results, "%d\n", nx);
    for(int k = 0; k < 3; k++)
    for(int i = 0; i < nx; i++)
        fprintf(Results, "%.15f\n", psiV(i * 3 + k));

    fclose(Results);

    // Evaluate dIcdS
    VectorXd dIcdS(nx + 1);
    dIcdS.setZero();

    // Get Fluxes
    getFlux(Flux, W);

    // Evaluate psiV * dRdS
    VectorXd psidRdS(nx + 1);
    psidRdS = evalpsidRdS(psiV, Flux, p);

    // Finite Difference dRdS
    MatrixXd dRdS(3 * nx, nx + 1);
    MatrixXd dRdSFD(3 * nx, nx + 1);
    dRdS = evaldRdS(Flux, S, W);
    dRdSFD = evaldRdS_FD(Flux, S, W);
    std::cout<<"(dRdSFD - dRdS).norm() / dRdS.norm()"<<std::endl;
    std::cout<<(dRdSFD - dRdS).norm() / dRdS.norm()<<std::endl;

    VectorXd psidRdSFD(nx + 1);
    psidRdSFD.setZero();
    psidRdSFD = psiV.transpose() * dRdS;
    std::cout<<"(psidRdSFD - psidRdS).norm() / psidRdS.norm()"<<std::endl;
    std::cout<<(psidRdSFD - psidRdS).norm() / psidRdS.norm()<<std::endl;
    // Evaluate dSdDesign
    MatrixXd dSdDesign(nx + 1, designVar.size());
    dSdDesign = evaldSdDesign(x, dx, designVar);

    VectorXd grad(designVar.size());
    grad = psidRdSFD.transpose() * dSdDesign;

    std::cout<<"Gradient from Adjoint:"<<std::endl;
    std::cout<<std::setprecision(15)<<grad<<std::endl;
//  std::cout<<"Analytical then FD 1st Block"<<std::endl;
//  std::cout<<MatrixXd(dRdW).block(0, 0, 3, 3)<<std::endl;
//  std::cout<<std::endl;
//  std::cout<<MatrixXd(matAFD2).block(0, 0, 3, 3)<<std::endl;
//  std::cout<<std::endl;
//  std::cout<<std::endl;
//  std::cout<<"Analytical then FD 2nd Block"<<std::endl;
//  std::cout<<MatrixXd(dRdW).block(0, 3, 3, 3)<<std::endl;
//  std::cout<<std::endl;
//  std::cout<<MatrixXd(matAFD2).block(0, 3, 3, 3)<<std::endl;
    std::cout<<std::endl;
    return grad;
}


// Calculates Jacobian
// Steger-Warming Flux Splitting
void StegerJac(std::vector <double> W,
               std::vector <double> &Ap_list,
               std::vector <double> &An_list,
               std::vector <double> &Flux)
{
    double eps = 0.1;
    double gam = 1.4;
    double M[3][3] = {{0}},
           Minv[3][3] = {{0}},
           N[3][3] = {{0}},
           Ninv[3][3] = {{0}},
           lambdaP[3][3],
           lambdaN[3][3];
    double lambdaa[3];
    
    
    double Ap[3][3], An[3][3], tempP[3][3], tempN[3][3], prefix[3][3], suffix[3][3];
    
    std::vector <double> rho(nx), u(nx), p(nx), c(nx);
    std::vector <double> Ap_list1(nx * 3 * 3, 0), An_list1(nx * 3 * 3, 0);

    double beta = gam - 1;

    for(int i = 0; i < nx; i++)
    {
        rho[i] = W[i * 3 + 0];
        u[i] = W[i * 3 + 1] / rho[i];
        p[i] = (gam-1) * (W[i * 3 + 2] - rho[i] * pow(u[i], 2) / 2);
        c[i] = sqrt( gam * p[i] / rho[i] );
    }


    for(int i = 0; i < nx; i++)
    {
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        {
            Ap[row][col] = 0;
            An[row][col] = 0;
            tempP[row][col] = 0;
            tempN[row][col] = 0;
            prefix[row][col] = 0;
            suffix[row][col] = 0;
            lambdaP[row][col] = 0;
            lambdaN[row][col] = 0;
        }
    
        M[0][0] = 1.0;
        M[1][0] = -u[i] / rho[i];
        M[2][0] = 0.5 * u[i] * u[i] * beta;
        M[1][1] = 1.0 / rho[i];
        M[2][1] = -u[i] * beta;
        M[2][2] = beta;
        Minv[0][0] = 1.0;
        Minv[1][0] = u[i];
        Minv[2][0] = 0.5 * u[i] * u[i];
        Minv[1][1] = rho[i];
        Minv[2][1] = u[i] * rho[i];
        Minv[2][2] = 1.0 / beta;
        N[0][0] = 1.0;
        N[1][1] = rho[i] * c[i];
        N[2][1] = -rho[i] * c[i];
        N[0][2] = -1.0 / (c[i] * c[i]);
        N[1][2] = 1.0;
        N[2][2] = 1.0;
        Ninv[0][0] = 1.0;
        Ninv[0][1] = 1.0 / (2.0 * c[i] * c[i]);
        Ninv[0][2] = 1.0 / (2.0 * c[i] * c[i]);
        Ninv[1][1] = 1.0 / (2.0 * rho[i] * c[i]);
        Ninv[1][2] = -1.0 / (2.0 * rho[i] * c[i]);
        Ninv[2][1] = 0.5;
        Ninv[2][2] = 0.5;
        lambdaa[0] = u[i];
        lambdaa[1] = u[i] + c[i];
        lambdaa[2] = u[i] - c[i];
        
        for(int k = 0; k < 3; k++)
            if(lambdaa[k] > 0)
                lambdaP[k][k] = (lambdaa[k] + sqrt(pow(lambdaa[k], 2) + pow(eps, 2))) / 2.0;
            else
                lambdaN[k][k] = (lambdaa[k] - sqrt(pow(lambdaa[k], 2) + pow(eps, 2))) / 2.0;

        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
            for(int k = 0; k < 3; k++)
            {
                prefix[row][col]+= Minv[row][k] * Ninv[k][col];
                suffix[row][col]+= N[row][k] * M[k][col];
            }
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
            for(int k = 0; k < 3; k++)
            {
                tempP[row][col] += prefix[row][k] * lambdaP[k][col];
                tempN[row][col] += prefix[row][k] * lambdaN[k][col];
            }
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
            for(int k = 0; k < 3; k++)
            {
                Ap[row][col]+= tempP[row][k] * suffix[k][col];
                An[row][col]+= tempN[row][k] * suffix[k][col];
            }
        // could remove above loop and just use aplist and anlist
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        {
            int vec_pos = (i * 3 * 3) + (row * 3) + col;
            Ap_list1[vec_pos] = Ap[row][col];
            An_list1[vec_pos] = An[row][col];
        }

    }

    for(int i = 1; i < nx; i++)
    {
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        {
            int Ap_pos = ((i - 1) * 3 * 3) + (row * 3) + col;
            int An_pos = (i * 3 * 3) + (row * 3) + col;
            Flux[i * 3 + row] += Ap_list1[Ap_pos] * W[(i - 1) * 3 + col]
                                 + An_list1[An_pos] * W[i * 3 + col];
        }
    }

}

void JacobianCenter(std::vector <double> &J,
                    double u, double c)
{
    J[0] = 0.0;
    J[1] = 1.0;
    J[2] = 0.0;
    J[3] = u * u * (gam - 3.0) / 2.0;
    J[4] = u * (3.0 - gam);
    J[5] = gam - 1.0;
    J[6] = ( pow(u, 3) * (gam - 1.0) * (gam - 2.0) - 2.0 * u * c * c ) / (2.0 * (gam - 1.0));
    J[7] = ( 2.0 * c * c + u * u * ( -2.0 * gam * gam + 5.0 * gam - 3.0 ) ) 
           / (2.0 * (gam - 1.0));
    J[8] = u * gam;
}

void ScalarJac(std::vector <double> W,
               std::vector <double> &Ap_list,
               std::vector <double> &An_list)
{
    std::vector <double> rho(nx), u(nx), e(nx);
    std::vector <double> T(nx), p(nx), c(nx), Mach(nx);
    WtoP(W, rho, u, e, p, c, T); 

    int vec_pos, k;
    double lamb;

    std::vector <double> J(9, 0);
    std::vector <double> dlambdadWp(3, 0);
    std::vector <double> dlambdaPdW(3, 0);
    std::vector <double> dlambdaNdW(3, 0);
    double dlambdadc, dcdr, dcdp;
    double dlambdadr, dlambdadu, dlambdadp;
    std::vector <double> dwpdw(9, 0);
    // A+
    for(int i = 0; i < nx - 1; i++)
    {
        // dF/dW
        JacobianCenter(J, u[i], c[i]);

        // lambda
        lamb = (u[i] + u[i + 1] + c[i] + c[i + 1]) / 2.0;
        // dlambdaP/dW
        dlambdadc = 0.5;
        dcdr = - p[i] * gam / (2.0 * c[i] * rho[i] * rho[i]);
        dcdp = gam / (2.0 * c[i] * rho[i]);

        dlambdadr = dlambdadc * dcdr;
        dlambdadu = 0.5;
        dlambdadp = dlambdadc * dcdp;

        dlambdadWp[0] = dlambdadr;
        dlambdadWp[1] = dlambdadu;
        dlambdadWp[2] = dlambdadp;
        dWpdW(dwpdw, W, i);
        dlambdaPdW[0] = 0;
        dlambdaPdW[1] = 0;
        dlambdaPdW[2] = 0;
        for(int row = 0; row < 1; row++)
        for(int col = 0; col < 3; col++)
        for(int k = 0; k < 3; k++)
            dlambdaPdW[row * 3 + col] += dlambdadWp[row * 3 + k] * dwpdw[k * 3 + col];

        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        {
            vec_pos = (i * 9) + (row * 3) + col; // NOT Transposed
            k = row * 3 + col;
            Ap_list[vec_pos] = J[k] / 2.0 - dlambdaPdW[col] * Scalareps
                               * (W[(i + 1) * 3 + row] - W[i * 3 + row]) / 2.0;
            if(row == col)
            {
                Ap_list[vec_pos] += Scalareps * lamb / 2.0;
            }
        }
    }
    
    // A-
    for(int i = 1; i < nx; i++)
    {
        // dF/dW
        JacobianCenter(J, u[i], c[i]);

        // lambda
        lamb = (u[i] + u[i - 1] + c[i] + c[i - 1]) / 2.0;
        // dlambdaP/dW
        dlambdadc = 0.5;
        dcdr = - p[i] * gam / (2.0 * c[i] * rho[i] * rho[i]);
        dcdp = gam / (2.0 * c[i] * rho[i]);

        dlambdadr = dlambdadc * dcdr;
        dlambdadu = 0.5;
        dlambdadp = dlambdadc * dcdp;

        dlambdadWp[0] = dlambdadr;
        dlambdadWp[1] = dlambdadu;
        dlambdadWp[2] = dlambdadp;
        dWpdW(dwpdw, W, i);
        dlambdaNdW[0] = 0;
        dlambdaNdW[1] = 0;
        dlambdaNdW[2] = 0;
        for(int row = 0; row < 1; row++)
        for(int col = 0; col < 3; col++)
        for(int k = 0; k < 3; k++)
            dlambdaNdW[row * 3 + col] += dlambdadWp[row * 3 + k] * dwpdw[k * 3 + col];

        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        {
            vec_pos = (i * 9) + (row * 3) + col; // NOT Transposed
            k = row * 3 + col;
            An_list[vec_pos] = J[k] / 2.0 - dlambdaNdW[col] * Scalareps
                               * (W[i * 3 + row] - W[(i - 1) * 3 + row]) / 2.0;
            if(row == col)
            {
                An_list[vec_pos] -= Scalareps * lamb / 2.0;
            }
        }
    }
}


MatrixXd evaldSdDesign(
    std::vector <double> x, 
    std::vector <double> dx, 
    std::vector <double> designVar)
{
    MatrixXd dSdDesign(nx + 1, designVar.size());
    double d1 = designVar[0];
    double d2 = designVar[1];
    double d3 = designVar[2];
    double xh;
    for(int i = 0; i < nx + 1; i++)
    {
        if(i == 0 || i == nx)
        {
            dSdDesign(i, 0) = 0;
            dSdDesign(i, 1) = 0;
            dSdDesign(i, 2) = 0;
        }
        else
        {
            xh = fabs(x[i] - dx[i] / 2.0);
            dSdDesign(i, 0) = - pow(sin(PI * pow(xh, d2)), d3);
            dSdDesign(i, 1) = - d1 * d3 * PI * pow(xh, d2)
                              * cos(PI * pow(xh, d2)) * log(xh)
                              * pow(sin(PI * pow(xh, d2)), d3 - 1);
            dSdDesign(i, 2) = - d1 * log(sin(PI * pow(xh, d2)))
                              * pow(sin(PI * pow(xh, d2)), d3);
        }
    }
    return dSdDesign;
}

VectorXd evaldIcdW(std::vector <double> W, std::vector <double> dx)
{
    VectorXd dIcdW(3 * nx);

    std::vector <double> ptarget(nx, 0);
    double dpdw[3], rho, u, p;
    ioTargetPressure(-1, ptarget);
    for(int i = 0; i < nx; i++)
    {
        rho = W[i * 3 + 0];
        u = W[i * 3 + 1] / rho;
        p = (gam - 1) * ( W[i * 3 + 2] - rho * u * u / 2.0 );

        dpdw[0] = (gam - 1) / 2.0 * u * u;
        dpdw[1] = - (gam - 1) * u;
        dpdw[2] = (gam - 1);

        dIcdW[i * 3 + 0] = (p / ptin - ptarget[i]) * dpdw[0] * dx[i] / ptin;
        dIcdW[i * 3 + 1] = (p / ptin - ptarget[i]) * dpdw[1] * dx[i] / ptin;
        dIcdW[i * 3 + 2] = (p / ptin - ptarget[i]) * dpdw[2] * dx[i] / ptin;
    }
    return dIcdW;
}

void evaldQdW(std::vector <double> &dQdW,
                   std::vector <double> W,
                   std::vector <double> S)
{
    double dpdw[3], rho, u, dS;
    for(int i = 0; i < nx; i++)
    {
        rho = W[i * 3 + 0];
        u = W[i * 3 + 1] / rho;

        dpdw[0] = (gam - 1) / 2.0 * u * u;
        dpdw[1] = - (gam - 1) * u;
        dpdw[2] = (gam - 1);

        dS = S[i + 1] - S[i];

        dQdW[i * 3 + 0] = dpdw[0] * dS;
        dQdW[i * 3 + 1] = dpdw[1] * dS;
        dQdW[i * 3 + 2] = dpdw[2] * dS;
    }
}

void BCJac(std::vector <double> W,
           std::vector <double> dt,
           std::vector <double> dx,
           std::vector <double> &dBidWi,
           std::vector <double> &dBidWd,
           std::vector <double> &dBodWd,
           std::vector <double> &dBodWo)
{
    std::vector <double> rho(nx), u(nx), e(nx), p(nx), c(nx), T(nx);
    std::vector <double> dbdwp(9, 0), dwpdw(9);

    for(int i = 0; i < 9; i++)
    {
        dBidWi[i] = 0;
        dBidWd[i] = 0;
        dBodWd[i] = 0;
        dBodWo[i] = 0;
    }

    WtoP(W, rho, u, e, p, c, T);

    // ************************
    // OUTLET JACOBIANS
    // ************************

    double i1, i2;
    double r1, r2, p1, p2, u1, u2, c1, c2;
    i1 = nx - 1;
    i2 = nx - 2;
    r1 = rho[i1];
    r2 = rho[i2];
    p1 = p[i1];
    p2 = p[i2];
    u1 = u[i1];
    u2 = u[i2];
    c1 = c[i1];
    c2 = c[i2];

    // Shorthand
    double gamr, fu, drho, dp, du, cr, uu;
    drho = r1 - r2;
    dp = p1 - p2;
    du = u1 - u2;
    cr = r1 * c1;
    uu = u1 * u1;

    // Speed of Sound
    double dc1dr1, dc2dr2, dc1dp1, dc2dp2;
    dc1dr1 = - p1 * gam / (2.0 * cr * r1);
    dc2dr2 = - p2 * gam / (2.0 * c2 * r2 * r2);
    dc1dp1 = gam / (2.0 * cr);
    dc2dp2 = gam / (2.0 * c2 * r2);

    double eig1, eig2, eig3;
    double deig1du1, deig1du2;
    double deig2dr1, deig2du1, deig2dp1, deig2dr2, deig2du2, deig2dp2;
    double deig3dr1, deig3du1, deig3dp1, deig3dr2, deig3du2, deig3dp2;
    // Eigenvalue
    eig1 = (u1 + u2) / 2.0;
    eig2 = eig1 + (c1 + c2) / 2.0;
    eig3 = eig1 - (c1 + c2) / 2.0;

    deig1du1 = 0.5;
    deig1du2 = 0.5;

    deig2dr1 = dc1dr1 / 2.0;
    deig2du1 = deig1du1;
    deig2dp1 = dc1dp1 / 2.0;
    deig2dr2 = dc2dr2 / 2.0;
    deig2du2 = deig1du2;
    deig2dp2 = dc2dp2 / 2.0;

    deig3dr1 = - dc1dr1 / 2.0;
    deig3du1 = deig1du1;
    deig3dp1 = - dc1dp1 / 2.0;
    deig3dr2 = - dc2dr2 / 2.0;
    deig3du2 = deig1du2;
    deig3dp2 = - dc2dp2 / 2.0;

    // Riemann invariants
    double R1, R2, R3;
    double dR1dr1, dR1du1, dR1dp1, dR1dr2, dR1du2, dR1dp2;
    double dR2dr1, dR2du1, dR2dp1, dR2dr2, dR2du2, dR2dp2;
    double dR3dr1, dR3du1, dR3dp1, dR3dr2, dR3du2, dR3dp2;
    R1 = - eig1 * (drho - dp / (c1 * c1));
    R2 = - eig2 * (dp + cr * du);
    R3 = - eig3 * (dp - cr * du);

    dR1dr1 = - eig1 * (1.0 + 2.0 * dp * dc1dr1 / pow(c1, 3) );
    dR1du1 = deig1du1 * (dp - c1 * c1 * drho) / (c1 * c1);
    dR1dp1 = eig1 * (c1 - 2 * dp * dc1dp1) / pow(c1, 3);
    dR1dr2 = eig1;
    dR1du2 = deig1du2 * (dp - c1 * c1 * drho) / (c1 * c1);
    dR1dp2 = - eig1 / (c1 * c1);

    dR2dr1 = - du * eig2 * (c1 + r1 * dc1dr1) - (dp + cr * du) * deig2dr1;
    dR2du1 = - cr * eig2 - (dp + cr * du) * deig2du1;
    dR2dp1 = - eig2 * (1.0 + du * r1 * dc1dp1) - (dp + cr * du) * deig2dp1;
    dR2dr2 = - (dp + cr * du) * deig2dr2;
    dR2du2 = cr * eig2 - (dp + cr * du) * deig2du2;
    dR2dp2 = eig2 - (dp + cr * du) * deig2dp2;

    dR3dr1 = eig3 * du * (c1 + r1 * dc1dr1) - (dp - cr * du) * deig3dr1; 
    dR3du1 = cr * eig3 - (dp - cr * du) * deig3du1;
    dR3dp1 = - eig3 - dp * deig3dp1 + du * r1 * eig3 * dc1dp1;
    dR3dr2 = - (dp - cr * du) * deig3dr2;
    dR3du2 = - cr * eig3 - (dp - cr * du) * deig3du2;
    dR3dp2 = eig3 - (dp - cr * du) * deig3dp2;

    // dp1/dt
    double dp1dt;
    double dp1dtdr1, dp1dtdu1, dp1dtdp1;
    double dp1dtdr2, dp1dtdu2, dp1dtdp2;
    if(u1 < c1)
    {
        dp1dt = 0;
        dp1dtdr1 = 0;
        dp1dtdu1 = 0;
        dp1dtdp1 = 0;
        dp1dtdr2 = 0;
        dp1dtdu2 = 0;
        dp1dtdp2 = 0;
    }
    else
    {
        dp1dt = (R2 + R3) / 2.0;
        dp1dtdr1 = (dR2dr1 + dR3dr1) / 2.0;
        dp1dtdu1 = (dR2du1 + dR3du1) / 2.0;
        dp1dtdp1 = (dR2dp1 + dR3dp1) / 2.0;
        dp1dtdr2 = (dR2dr2 + dR3dr2) / 2.0;
        dp1dtdu2 = (dR2du2 + dR3du2) / 2.0;
        dp1dtdp2 = (dR2dp2 + dR3dp2) / 2.0;
    }

    // drho1/dt
    double dr1dt;
    double dr1dtdr1, dr1dtdu1, dr1dtdp1;
    double dr1dtdr2, dr1dtdu2, dr1dtdp2;
    dr1dt = R1 + dp1dt / (c1 * c1);

    dr1dtdr1 = dR1dr1 + dp1dtdr1 / (c1 * c1) - 2.0 * dp1dt * dc1dr1 / pow(c1, 3);
    dr1dtdu1 = dR1du1 + dp1dtdu1 / (c1 * c1);
    dr1dtdp1 = dR1dp1 + dp1dtdp1 / (c1 * c1) - 2.0 * dp1dt * dc1dp1 / pow(c1, 3);
    dr1dtdr2 = dR1dr2 + dp1dtdr2 / (c1 * c1);
    dr1dtdu2 = dR1du2 + dp1dtdu2 / (c1 * c1);
    dr1dtdp2 = dR1dp2 + dp1dtdp2 / (c1 * c1);

    // du1/dt
    double du1dt;
    double du1dtdr1, du1dtdu1, du1dtdp1;
    double du1dtdr2, du1dtdu2, du1dtdp2;
    du1dt = (R2 - dp1dt) / (cr);

    du1dtdr1 = ( (dp1dt - R2) * r1 * dc1dr1
               + c1 * (dp1dt - R2 - r1 * dp1dtdr1 + r1 * dR2dr1) )
               / (cr * cr);
    du1dtdu1 = (dR2du1 - dp1dtdu1) / cr;
    du1dtdp1 = ( (dp1dt - R2) * dc1dp1 + c1 * (dR2dp1 - dp1dtdp1) ) / (cr * c1);
    du1dtdr2 = (dR2dr2 - dp1dtdr2) / cr;
    du1dtdu2 = (dR2du2 - dp1dtdu2) / cr;
    du1dtdp2 = (dR2dp2 - dp1dtdp2) / cr;

    // d(ru)1/dt
//  double dru1dt;
//  dru1dt = r1 * du1dt + u1 * dr1dt;
    double dru1dtdr1, dru1dtdu1, dru1dtdp1;
    double dru1dtdr2, dru1dtdu2, dru1dtdp2;
    dru1dtdr1 = du1dt + u1 * dr1dtdr1 + r1 * du1dtdr1;
    dru1dtdu1 = dr1dt + u1 * dr1dtdu1 + r1 * du1dtdu1;
    dru1dtdp1 = u1 * dr1dtdp1 + r1 * du1dtdp1;
    dru1dtdr2 = u1 * dr1dtdr2 + r1 * du1dtdr2;
    dru1dtdu2 = u1 * dr1dtdu2 + r1 * du1dtdu2;
    dru1dtdp2 = u1 * dr1dtdp2 + r1 * du1dtdp2;

    // de1/dt
//  double de1dt;
//  de1dt = dp1dt * Cv / R + u1 * r1 * du1dt + uu * dr1dt / 2.0;
    double de1dtdr1, de1dtdu1, de1dtdp1;
    double de1dtdr2, de1dtdu2, de1dtdp2;

    de1dtdr1 = dp1dtdr1 * Cv / R + uu * dr1dtdr1 / 2.0 + r1 * u1 * du1dtdr1 
               + du1dt * u1;
    de1dtdu1 = dp1dtdu1 * Cv / R + uu * dr1dtdu1 / 2.0 + r1 * u1 * du1dtdu1 
               + du1dt * r1 + dr1dt * u1;
    de1dtdp1 = dp1dtdp1 / (gam - 1) + uu * dr1dtdp1 / 2.0 + r1 * u1 * du1dtdp1;
    de1dtdr2 = dp1dtdr2 / (gam - 1) + uu * dr1dtdr2 / 2.0 + r1 * u1 * du1dtdr2;
    de1dtdu2 = dp1dtdu2 / (gam - 1) + uu * dr1dtdu2 / 2.0 + r1 * u1 * du1dtdu2;
    de1dtdp2 = dp1dtdp2 / (gam - 1) + uu * dr1dtdp2 / 2.0 + r1 * u1 * du1dtdp2;

    dbdwp[0] = dr1dtdr1;
    dbdwp[1] = dr1dtdu1;
    dbdwp[2] = dr1dtdp1;
    dbdwp[3] = dru1dtdr1;
    dbdwp[4] = dru1dtdu1;
    dbdwp[5] = dru1dtdp1;
    dbdwp[6] = de1dtdr1;
    dbdwp[7] = de1dtdu1;
    dbdwp[8] = de1dtdp1;

    std::cout.precision(17);
    // Get Transformation Matrix
    dWpdW(dwpdw, W, nx - 1);

    for(int row = 0; row < 3; row++)
    for(int col = 0; col < 3; col++)
    for(int k = 0; k < 3; k++)
        dBodWo[row * 3 + col] += dbdwp[row * 3 + k] * dwpdw[k * 3 + col];
    
    dbdwp[0] = dr1dtdr2;
    dbdwp[1] = dr1dtdu2;
    dbdwp[2] = dr1dtdp2;
    dbdwp[3] = dru1dtdr2;
    dbdwp[4] = dru1dtdu2;
    dbdwp[5] = dru1dtdp2;
    dbdwp[6] = de1dtdr2;
    dbdwp[7] = de1dtdu2;
    dbdwp[8] = de1dtdp2;

    // Get Transformation Matrix
    dWpdW(dwpdw, W, nx - 2);

    for(int row = 0; row < 3; row++)
    for(int col = 0; col < 3; col++)
    for(int k = 0; k < 3; k++)
        dBodWd[row * 3 + col] += dbdwp[row * 3 + k] * dwpdw[k * 3 + col];


    // *********************
    // INLET JACOBIANS
    // *********************
    // Subsonic Inlet
    if(u[0] < c[0])
    {
        // Values at time-step N
        i1 = 0;
        i2 = 1;
        r1 = rho[i1];
        r2 = rho[i2];
        p1 = p[i1];
        p2 = p[i2];
        u1 = u[i1];
        u2 = u[i2];
        c1 = c[i1];
        c2 = c[i2];

        // Shorthand
        drho = r2 - r1;
        dp = p2 - p1;
        du = u2 - u1;
        cr = r1 * c1;
        uu = u1 * u1;
        gamr = (gam - 1.0) / (gam + 1.0);
        fu = 1.0 - gamr * u1 * u1 / a2;

        // Speed of Sound
        dc1dr1 = - p1 * gam / (2.0 * cr * r1);
        dc2dr2 = - p2 * gam / (2.0 * c2 * r2 * r2);
        dc1dp1 = gam / (2.0 * cr);
        dc2dp2 = gam / (2.0 * c2 * r2);

        // Eigenvalue
        eig1 = (u1 + u2) / 2.0;
        eig3 = eig1 - (c1 + c2) / 2.0;
    
        deig1du1 = 0.5;
        deig1du2 = 0.5;
    
        deig3dr1 = - dc1dr1 / 2.0;
        deig3du1 = deig1du1;
        deig3dp1 = - dc1dp1 / 2.0;
        deig3dr2 = - dc2dr2 / 2.0;
        deig3du2 = deig1du2;
        deig3dp2 = - dc2dp2 / 2.0;

        // Riemann Invariants
        R3 = - eig3 * (dp - cr * du);

        dR3dr1 = -eig3 * (-c1 * du - du * r1 * dc1dr1) - (dp - cr * du) * deig3dr1;
        dR3du1 = -cr * eig3 - (dp - cr * du) * deig3du1;
        dR3dp1 = eig3 * (1 + du * r1 * dc1dp1) - (dp - cr * du) * deig3dp1;
        dR3dr2 = -(dp - cr * du) * deig3dr2;
        dR3du2 = cr * eig3 - (dp - cr * du) * deig3du2;
        dR3dp2 = -eig3 - (dp - cr * du) * deig3dp2;
        // dp1
        double dp1du1_n, dp1du1du1;
        // Same Values
        dp1du1_n = -2.0 * gamr * ptin * u1 * pow(fu, 1.0 / (gam - 1.0)) * gam
                 / (a2 * (gam - 1.0));

        dp1du1du1 = 2.0 * gamr * ptin * pow(fu, gam/(gam - 1.0)) * gam
                    * (a2 - a2 * gam + gamr * uu * (gam + 1))
                    / pow((a2 - gamr * uu) * (gam - 1.0), 2);

        // du1
        du1dt = R3 / (dp1du1_n - cr);
        du1dtdr1 = dR3dr1 / (dp1du1_n - cr)
                   - R3 * (-c1 - r1 * dc1dr1) / pow((dp1du1_n - cr), 2);
        du1dtdu1 = dR3du1 / (dp1du1_n - cr)
                   - R3 * dp1du1du1 / pow((dp1du1_n - cr), 2);
        du1dtdp1 = dR3dp1 / (dp1du1_n - cr)
                   + (R3 * r1 * dc1dp1) / pow((dp1du1_n - cr), 2);
        du1dtdr2 = dR3dr2 / (dp1du1_n - cr);
        du1dtdu2 = dR3du2 / (dp1du1_n - cr);
        du1dtdp2 = dR3dp2 / (dp1du1_n - cr);

        // Primitive values at time-step n+1
        double unp1, pnp1, rnp1, tnp1, funp1;
        unp1 = u1 + du1dt;
        pnp1 = ptin * pow(1 - gamr * pow(unp1, 2) / a2, gam / (gam - 1.0));
        tnp1 = Ttin * ( 1 - gamr * unp1 * unp1 / a2 );
        rnp1 = pnp1 / (R * tnp1);
        funp1 = 1.0 - gamr * unp1 * unp1 / a2;
        double dpnp1dunp1;
        dpnp1dunp1 = -2.0 * gamr * ptin * unp1 * pow(funp1, 1.0 / (gam - 1.0)) * gam
                     / (a2 * (gam - 1.0));
        
        // dp1
        dp1dt  = pnp1 - p1;
        dp1dtdr1 = dpnp1dunp1 * du1dtdr1;
        dp1dtdu1 = dpnp1dunp1 * (du1dtdu1 + 1);
        dp1dtdp1 = dpnp1dunp1 * du1dtdp1 - 1;
        dp1dtdr2 = dpnp1dunp1 * du1dtdr2;
        dp1dtdu2 = dpnp1dunp1 * du1dtdu2;
        dp1dtdp2 = dpnp1dunp1 * du1dtdp2;

        // dr1
        // Total derivative from rho_n+1 to p_n+1 and u_n+1
        double drnp1dpnp1, drnp1dtnp1, dtnp1dpnp1;
        drnp1dpnp1 = 1 / (R * tnp1);
        drnp1dtnp1 = -pnp1 / (R * tnp1 * tnp1);
        dtnp1dpnp1 = Ttin / ptin * (gam - 1.0) / gam * pow(pnp1 / ptin, - 1.0 / gam);
        double Drnp1Dpnp1 = drnp1dpnp1 + drnp1dtnp1 * dtnp1dpnp1;
        double drnp1dunp1 = Drnp1Dpnp1 * dpnp1dunp1;

        dr1dt = rnp1 - r1;

        dr1dtdr1 = drnp1dunp1 * du1dtdr1 - 1;
        dr1dtdu1 = drnp1dunp1 * (du1dtdu1 + 1);
        dr1dtdp1 = drnp1dunp1 * du1dtdp1;

        dr1dtdr2 = drnp1dunp1 * du1dtdr2;
        dr1dtdu2 = drnp1dunp1 * du1dtdu2;
        dr1dtdp2 = drnp1dunp1 * du1dtdp2;

        // dru1/dt
//      dru1dt = r1 * du1dt + u1 * dr1dt;

        dru1dtdr1 = du1dt + u1 * dr1dtdr1 + r1 * du1dtdr1;
        dru1dtdu1 = dr1dt + u1 * dr1dtdu1 + r1 * du1dtdu1;
        dru1dtdp1 = u1 * dr1dtdp1 + r1 * du1dtdp1;
        dru1dtdr2 = u1 * dr1dtdr2 + r1 * du1dtdr2;
        dru1dtdu2 = u1 * dr1dtdu2 + r1 * du1dtdu2;
        dru1dtdp2 = u1 * dr1dtdp2 + r1 * du1dtdp2;

        // de1/dt
//      de1dt = dp1dt / (gam - 1.0) + r1 * u1 * du1dt + uu * dr1dt / 2.0;

        de1dtdr1 = dp1dtdr1 * Cv / R + uu * dr1dtdr1 / 2.0 + r1 * u1 * du1dtdr1 
                   + du1dt * u1;
        de1dtdu1 = dp1dtdu1 * Cv / R + uu * dr1dtdu1 / 2.0 + r1 * u1 * du1dtdu1 
                   + du1dt * r1 + dr1dt * u1;
        de1dtdp1 = dp1dtdp1 / (gam - 1.0) + uu * dr1dtdp1 / 2.0 + r1 * u1 * du1dtdp1;
        de1dtdr2 = dp1dtdr2 / (gam - 1.0) + uu * dr1dtdr2 / 2.0 + r1 * u1 * du1dtdr2;
        de1dtdu2 = dp1dtdu2 / (gam - 1.0) + uu * dr1dtdu2 / 2.0 + r1 * u1 * du1dtdu2;
        de1dtdp2 = dp1dtdp2 / (gam - 1.0) + uu * dr1dtdp2 / 2.0 + r1 * u1 * du1dtdp2;

        // Assign dR1/dWp1
        dbdwp[0] = dr1dtdr1;
        dbdwp[1] = dr1dtdu1;
        dbdwp[2] = dr1dtdp1;
        dbdwp[3] = dru1dtdr1;
        dbdwp[4] = dru1dtdu1;
        dbdwp[5] = dru1dtdp1;
        dbdwp[6] = de1dtdr1;
        dbdwp[7] = de1dtdu1;
        dbdwp[8] = de1dtdp1;

        // Get Transformation Matrix
        dWpdW(dwpdw, W, 0);
        
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        for(int k = 0; k < 3; k++)
            dBidWi[row * 3 + col] += dbdwp[row * 3 + k] * dwpdw[k * 3 + col];

        // Assign dR1/dWp2
        dbdwp[0] = dr1dtdr2;
        dbdwp[1] = dr1dtdu2;
        dbdwp[2] = dr1dtdp2;
        dbdwp[3] = dru1dtdr2;
        dbdwp[4] = dru1dtdu2;
        dbdwp[5] = dru1dtdp2;
        dbdwp[6] = de1dtdr2;
        dbdwp[7] = de1dtdu2;
        dbdwp[8] = de1dtdp2;

        // Get Transformation Matrix
        dWpdW(dwpdw, W, 1);

        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        for(int k = 0; k < 3; k++)
            dBidWd[row * 3 + col] += dbdwp[row * 3 + k] * dwpdw[k * 3 + col];
    }

    // Supersonic Inlet
    else
    {
        for(int i = 0; i < 9; i++)
        {
            dBidWi[i] = 0;
            dBidWd[i] = 0;
            if(i % 4 == 0)
                dBidWi[i] = 1;
        }
    }
}

SparseMatrix<double> evaldRdW(std::vector <double> Ap,
                                  std::vector <double> An,
                                  std::vector <double> W,
                                  std::vector <double> dQdW,
                                  std::vector <double> dx,
                                  std::vector <double> dt,
                                  std::vector <double> S,
                                  double Min)
{
    SparseMatrix<double> dRdW(3 * nx, 3 * nx);
    int Ri, Wi;
    int k, rowi, coli;
    double val;
    std::vector <double> dBidWi(9), dBidWd(9), dBodWd(9), dBodWo(9);
    BCJac(W, dt, dx, dBidWi, dBidWd, dBodWd, dBodWo);
    // Input 4 lines where BC Jacobians occur
    // psi(1), psi(2), psi(n-1), psi(n)
    for(int row = 0; row < 3; row++)
    {
        for(int col = 0; col < 3; col++)
        {
            k = row * 3 + col;
            // d(inlet)/d(inlet)
            // R0, W0
            Ri = 0;
            Wi = 0;
            rowi = Ri * 3 + row;
            coli = Wi * 3 + col;

            val = - dBidWi[k];
            dRdW.insert(rowi, coli) = val;

            // d(inlet)/d(domain)
            // R0, W1
            Ri = 0;
            Wi = 1;
            rowi = Ri * 3 + row;
            coli = Wi * 3 + col;

            val = - dBidWd[k];
            dRdW.insert(rowi, coli) = val;

            // d(outlet)/d(outlet)
            // R = nx - 1, W = nx - 1
            Ri = nx - 1;
            Wi = nx - 1;
            rowi = Ri * 3 + row;
            coli = Wi * 3 + col;

            val = - dBodWo[k];
            dRdW.insert(rowi, coli) = val;

            // d(outlet)/d(domain)
            // R = nx - 1, W = nx - 2
            Ri = nx - 1;
            Wi = nx - 2;
            rowi = Ri * 3 + row;
            coli = Wi * 3 + col;

            val = - dBodWd[k];
            dRdW.insert(rowi, coli) = val;
        }
    }
    for(int Ri = 1; Ri < nx - 1; Ri++)
    {
        Wi = Ri - 1;
        if(Wi >= 0)
        {
            for(int row = 0; row < 3; row++)
            for(int col = 0; col < 3; col++)
            {
                k = row * 3 + col;
                rowi = Ri * 3 + row;
                coli = Wi * 3 + col;

                val = - Ap[Wi * 9 + k] * S[Ri];
                dRdW.insert(rowi, coli) = val;
            }
        }

        Wi = Ri;
        if(Wi >= 0 && Wi <= nx - 1)
        {
            for(int row = 0; row < 3; row++)
            for(int col = 0; col < 3; col++)
            {
                k = row * 3 + col;
                rowi = Ri * 3 + row;
                coli = Wi * 3 + col;

                val = Ap[Wi * 9 + k] * S[Ri + 1];
                val -= An[Wi * 9 + k] * S[Ri];
                if(row == 1) 
                {
                    val -= dQdW[Wi * 3 + col];
                }

                dRdW.insert(rowi, coli) = val;
            }
        }

        Wi = Ri + 1;
        if(Wi <= nx - 1)
        {
            for(int row = 0; row < 3; row++)
            for(int col = 0; col < 3; col++)
            {
                k = row * 3 + col;
                rowi = Ri * 3 + row;
                coli = Wi * 3 + col;

                val = An[Wi * 9 + k] * S[Ri + 1];
                dRdW.insert(rowi, coli) = val;
            }
        }
    }
    if(Min > 1.0)
    {
        // Supersonic Inlet, don't solve for psi(0)
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        {
            // R1, W0
            Ri = 1;
            Wi = 0;
            rowi = Ri * 3 + row;
            coli = Wi * 3 + col;
    
            dRdW.coeffRef(rowi, coli) = 0;
        }
    }
    return dRdW;
}

VectorXd buildbMatrix(std::vector <double> dIcdW)
{
    VectorXd matb(3 * nx);

    for(int i = 0; i < nx; i++)
    for(int k = 0; k < 3; k++)
        matb(i * 3 + k) = -dIcdW[i * 3 + k];
    
    return matb;
}

VectorXd evalpsidRdS(
    VectorXd psiV,
    std::vector <double> Flux,
    std::vector <double> p)
{
    VectorXd psidRdS(nx + 1);
    psidRdS.setZero();
    for(int i = 2; i < nx - 1; i++)
    for(int k = 0; k < 3; k++)
    {
        psidRdS(i) += psiV((i - 1) * 3 + k) * Flux[i * 3 + k];
        psidRdS(i) -= psiV(i * 3 + k) * Flux[i * 3 + k];
        if(k == 1)
        {
            psidRdS(i) -= psiV((i - 1) * 3 + k) * p[i - 1];
            psidRdS(i) += psiV(i * 3 + k) * p[i];
        }
    }

    // Evaluate psiV * dRdS neat the Boundaries
    for(int k = 0; k < 3; k++)
    {
        // Cell 0 Inlet is not a function of the shape

        // Cell 1
        psidRdS(1) -= psiV(1 * 3 + k) * Flux[1 * 3 + k];

        // Cell nx - 1
        psidRdS(nx - 1) += psiV((nx - 2) * 3 + k) * Flux[(nx - 1) * 3 + k];

        // Cell nx Outlet is not a function of the shape

        if(k == 1)
        {
            psidRdS(1) += psiV(1 * 3 + k) * p[1];
            psidRdS(nx - 1) -= psiV((nx - 2) * 3 + k) * p[nx - 1];
        }
    }
    return psidRdS;
}

MatrixXd evaldRdS(std::vector <double> Flux, std::vector <double> S,
                  std::vector <double> W)
{
    MatrixXd dRdS(3 * nx, nx + 1);
    std::vector <double> Q(3 * nx, 0), p(nx);
    WtoQ(W, Q, S);
    getp(W, p);
    int Si, kR, kS;
    dRdS.setZero();
    for(int Ri = 1; Ri < nx - 1; Ri++)
    {
        for(int k = 0; k < 3; k++)
        {
            kR = Ri * 3 + k;

            Si = Ri;
            kS = Si * 3 + k;
            dRdS(kR, Si) = -Flux[kS];
            if(k == 1) dRdS(kR, Si) += p[Ri];

            Si = Ri + 1;
            kS = Si * 3 + k;
            dRdS(kR, Si) = Flux[kS];
            if(k == 1) dRdS(kR, Si) += -p[Ri];
        }
    }
    return dRdS;
}

MatrixXd evaldRdS_FD(std::vector <double> Flux, std::vector <double> S,
                  std::vector <double> W)
{
    MatrixXd dRdS(3 * nx, nx + 1);
    std::vector <double> Resi0(3 * nx, 0), Resi1(3 * nx, 0), Resi2(3 * nx, 0);
    std::vector <double> Sd(nx + 1, 0);
    std::vector <double> Q(3 * nx, 0);
    double h = 0.000000001;
    double pert;
    int ki, kip;
    dRdS.setZero();
    for(int Ri = 1; Ri < nx - 1; Ri++)
    {
        for(int Si = 0; Si < nx + 1; Si++)
        {
            for(int m = 0; m < nx + 1; m++)
                Sd[m] = S[m];

            pert = S[Si] * h;
            Sd[Si] = S[Si] + pert;
            
            WtoQ(W, Q, Sd);
            
            for(int k = 0; k < 3; k++)
            {
                ki = Ri * 3 + k;
                kip = (Ri + 1) * 3 + k;
                Resi1[ki] = Flux[kip] * Sd[Ri + 1] - Flux[ki] * Sd[Ri] - Q[ki];
            }
            
            for(int m = 0; m < nx + 1; m++)
                Sd[m] = S[m];

            Sd[Si] = S[Si] - pert;

            WtoQ(W, Q, Sd);

            for(int k = 0; k < 3; k++)
            {
                ki = Ri * 3 + k;
                kip = (Ri + 1) * 3 + k;
                Resi2[ki] = Flux[kip] * Sd[Ri + 1] - Flux[ki] * Sd[Ri] - Q[ki];
                dRdS(ki, Si) = (Resi1[ki] - Resi2[ki]) / (2 * pert);
            }
        }
    }
    
    return dRdS;
}

SparseMatrix<double> evaldRdW_FD(std::vector <double> W,
                                  std::vector <double> S,
                                  double Min)
{
    SparseMatrix<double> dRdW(3 * nx, 3 * nx);
    int Ri, Wi;
    int rowi, coli;
    std::vector <double> Wd(3 * nx, 0), F(3 * nx, 0), Q(3 * nx, 0); 
    std::vector <double> Flux(3 * (nx + 1), 0);
    std::vector <double> Resi1(3 * nx, 0), Resi2(3 * nx, 0);
    std::vector <double> dRdW_block(9, 0), dRdWp(9, 0), dwdwp(9, 0);
    WtoF(W, F);
    WtoQ(W, Q, S);
    getFlux(Flux, W);
    int ki, kip;
    double pert;

    // DR/DW
    for(int Ri = 0; Ri < nx; Ri++) // LOOP OVER R
    {
        for(int Wi = 0; Wi < nx; Wi++) // LOOP OVER W
        {
            double h = 0.00000001;
            for(int statei = 0; statei < 3; statei++) // LOOP OVER STATEI
            {
                for(int i = 0; i < 3 * nx; i++)
                    Wd[i] = W[i];
                
                pert = W[Wi * 3 + statei] * h;
                Wd[Wi * 3 + statei] = W[Wi * 3 + statei] + pert;

                // RESI 1
                // Inlet
                if(Ri == 0) inletBC(Wd, Resi1, 1.0, 1.0);
                // Outlet
                else if (Ri == nx - 1) outletBC(Wd, Resi1, 1.0, 1.0);
                // Domain
                else
                {
                    WtoF(Wd, F);
                    WtoQ(Wd, Q, S);
                    getFlux(Flux, Wd);
                    
                    for(int resii = 0; resii < 3; resii++)
                    {
                        ki = Ri * 3 + resii;
                        kip = (Ri + 1) * 3 + resii;
                        Resi1[ki] = Flux[kip] * S[Ri + 1] - Flux[ki] * S[Ri] - Q[ki];
                    }
                }

                for(int i = 0; i < 3 * nx; i++)
                    Wd[i] = W[i];
                
                Wd[Wi * 3 + statei] = W[Wi * 3 + statei] - pert;
                // RESI 2
                // Inlet
                if(Ri == 0) inletBC(Wd, Resi2, 1.0, 1.0);
                // Outlet
                else if (Ri == nx - 1) outletBC(Wd, Resi2, 1.0, 1.0);
                // Domain
                else
                {
                    WtoF(Wd, F);
                    WtoQ(Wd, Q, S);
                    getFlux(Flux, Wd);
                    
                    for(int resii = 0; resii < 3; resii++)
                    {
                        ki = Ri * 3 + resii;
                        kip = (Ri + 1) * 3 + resii;
                        Resi2[ki] = Flux[kip] * S[Ri + 1] - Flux[ki] * S[Ri] - Q[ki];
                    }
                }
                
                for(int resii = 0; resii < 3; resii++)
                {
                    ki = Ri * 3 + resii;
                    dRdW_block[resii * 3 + statei] = (Resi1[ki] - Resi2[ki]) / (2 * pert);
                }
                
            } // END STATEI LOOP

            dWdWp(dwdwp, W, Wi);

            std::fill(dRdWp.begin(), dRdWp.end(), 0.0);
            for(int row = 0; row < 3; row++)
            for(int col = 0; col < 3; col++)
            for(int k = 0; k < 3; k++)
                dRdWp[row * 3 + col] += dRdW_block[row * 3 + k] * dwdwp[k * 3 + col];

            for(int row = 0; row < 3; row++)
            for(int col = 0; col < 3; col++)
            {
                rowi = Ri * 3 + row;
                coli = Wi * 3 + col;
                dRdW.insert(rowi, coli) = dRdW_block[row * 3 + col];
//                dRdW.coeffRef(rowi, coli) = dRdWp[row * 3 + col];
            }
        }  // END LOOP OVER W
    } // END LOOP OVER R
    if(Min > 1.0)
    {
        // Supersonic Inlet, don't solve for psi(0)
        for(int row = 0; row < 3; row++)
        for(int col = 0; col < 3; col++)
        {
            // R0, W0
            Ri = 0;
            Wi = 0;
            rowi = Ri * 3 + row;
            coli = Wi * 3 + col;
            dRdW.coeffRef(rowi, coli) = 0;
            if(row == col)
                dRdW.coeffRef(rowi, coli) = 1;

            // R1, W0
            Ri = 1;
            Wi = 0;
            rowi = Ri * 3 + row;
            coli = Wi * 3 + col;
    
            dRdW.coeffRef(rowi, coli) = 0;
        }
    }
    return dRdW;
}

MatrixXd solveSparseAXB(SparseMatrix <double> A, MatrixXd B, int eig_solv)
{
    MatrixXd X(A.rows(), B.cols());
    X.setZero();
    MatrixXd matAdense(3 * nx, 3 * nx);
    MatrixXd eye(3 * nx, 3 * nx);
    eye.setIdentity();
    matAdense = A * eye;

    double offset = 0;//0.00001;
    matAdense = matAdense + eye * offset;

    JacobiSVD<MatrixXd> svd(matAdense);
    double svdmax = svd.singularValues()(0);
    double svdmin = svd.singularValues()(svd.singularValues().size()-1);
    double cond = svdmax / svdmin;
//  std::cout<<"Condition Number SVD"<<std::endl;
//  std::cout<<cond<<std::endl;
//  std::cout<<"Max/Min Singular Values"<<std::endl;
//  std::cout<<svdmax<< " / "<<svdmin<<std::endl;

    // Sparse LU
    if(eig_solv == 0)
    {
        SparseLU <SparseMatrix <double>, COLAMDOrdering< int > > slusolver;
        slusolver.analyzePattern(A);
        slusolver.factorize(A);
        
        if(slusolver.info() != 0)
            std::cout<<"Factorization failed. Error: "<<slusolver.info()<<std::endl;
    
        // Solve for X
        X = slusolver.solve(B);
    }
    // Dense LU full pivoting
    if(eig_solv == 1)
    {
        // Full Pivoting LU Factorization
        X = matAdense.fullPivLu().solve(B);
    }
    // Iterative LU
    if(eig_solv == 2)
    {
        BiCGSTAB<SparseMatrix <double> > itsolver;
        itsolver.compute(A);
        if(itsolver.info() == 0)
            std::cout<<"Iterative Factorization success"<<std::endl;
        else
            std::cout<<"Factorization failed. Error: "<<itsolver.info()<<std::endl;
        std::cout << "#iterations:     " << itsolver.iterations() << std::endl;
        std::cout << "estimated error: " << itsolver.error()      << std::endl;
        X = itsolver.solve(B);
    }
//  std::cout<<"||Ax - B||"<<std::endl;
//  std::cout<<(matAdense * X - B).norm()<<std::endl;

    return X;
}

VectorXd itSolve(SparseMatrix <double> A, VectorXd b)
{
    double resi1 = 1, resi2 = 1, resi3 = 1;
    // Directly Solve the Linear System Iteratively
    // Using Sub-Matrices
    //  --------------
    // |  A1   |  A2  |   | b1 |
    // |--------------| = |    |
    // |  A3   |  A4  |   | b2 |
    //  --------------
    MatrixXd A1(3 * (nx - 1), 3 * (nx - 1));
    MatrixXd A2(3 * (nx - 1), 3 * (nx - 1));
    MatrixXd A3(3 * (nx - 1), 3 * (nx - 1));
    MatrixXd A4(3 * (nx - 1), 3 * (nx - 1));
    A1 = MatrixXd(A.block(0, 0, 3 * (nx - 1), 3 * (nx - 1)));
    A2 = MatrixXd(A.block(3 * (nx - 2), 3 * (nx - 1), 3, 3));
    A3 = MatrixXd(A.block(3 * (nx - 1), 3 * (nx - 2), 3, 3));
    A4 = MatrixXd(A.block(3 * (nx - 1), 3 * (nx - 1), 3, 3));
    A4 = A4 + MatrixXd(3, 3).setIdentity() * 0.000001;
//  std::cout<<A<<std::endl;
//  std::cout<<std::endl;
//  std::cout<<std::endl;
//  std::cout<<std::endl;


    VectorXd b1(3 * (nx - 1)), b2(3);
    b1 = b.head(3 * (nx - 1));
    b2 = b.tail(3);

    VectorXd b1mod(3 * (nx - 1)), b2mod(3);
    b1mod = b1;
    b2mod = b2;
    
    VectorXd fullX(3 * nx);
    fullX.setZero();
    fullX.setOnes();
    fullX = MatrixXd(A).fullPivLu().solve(b);

    VectorXd x1(3 * (nx - 1));
    VectorXd x2(3);
    x1 = fullX.head(3 * (nx - 1));
    x2 = fullX.tail(3);

//  b1mod.tail(3) = b1.tail(3) - A2 * x2.tail(3);
//  b2mod.tail(3) = b2.tail(3) - A3 * x1.tail(3);

    double tol1 = 5e-13;
    double tol2 = tol1;
    double tol3 = 1;//tol1;
    int it = 0;
    while(resi1 > tol1 || resi2 > tol2 || resi3 > tol3)
    {
        it++;
        x1 = A1.fullPivLu().solve(b1mod);
        x2 = A4.fullPivLu().solve(b2mod);
        b1mod.tail(3) = b1.tail(3) - A2 * x2.tail(3);
        b2mod.tail(3) = b2.tail(3) - A3 * x1.tail(3);

        resi1 = (A1 * x1 - b1mod).norm();
        resi2 = (A4 * x2 - b2mod).norm();
        fullX.head(3 * (nx - 1)) = x1;
        fullX.tail(3) = x2;
        resi3 = (A * fullX - b).norm();
        
        std::cout<<"Iteration: "<<it
                 <<" resi1: "<<resi1
                 <<" resi2: "<<resi2
                 <<" resi3: "<<resi3
                 <<std::endl;
    }
    JacobiSVD<MatrixXd> svd(A1);
    double svdmax = svd.singularValues()(0);
    double svdmin = svd.singularValues()(svd.singularValues().size()-1);
    double cond = svdmax / svdmin;
    std::cout<<"Condition Number A1"<<std::endl;
    std::cout<<cond<<std::endl;
    std::cout<<"Max/Min Singular Values"<<std::endl;
    std::cout<<svdmax<< " / "<<svdmin<<std::endl;
    JacobiSVD<MatrixXd> svd2(A4);
    svdmax = svd2.singularValues()(0);
    svdmin = svd2.singularValues()(svd2.singularValues().size()-1);
    cond = svdmax / svdmin;
    std::cout<<"Condition Number A4"<<std::endl;
    std::cout<<cond<<std::endl;
    std::cout<<"Max/Min Singular Values"<<std::endl;
    std::cout<<svdmax<< " / "<<svdmin<<std::endl;

    std::cout<<"||Ax - b||"<<std::endl;
    std::cout<<(A * fullX - b).norm()<<std::endl;

    return fullX;
}
