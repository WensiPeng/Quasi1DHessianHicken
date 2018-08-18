#include<vector>
#include<Eigen/Core>
#include<Eigen/Sparse>
#include"globals.h"
#include"fitness.h"
#include<iostream>

using namespace Eigen;

// dIc / dW
VectorXd evaldIcdW(
    std::vector <double> W,
    std::vector <double> dx)
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

VectorXd evaldIcdW_FD(std::vector <double> W,
                      std::vector <double> dx)
{
    VectorXd dIcdW(3 * nx);
    std::vector <double> Wd(3 * nx, 0);
    std::vector <double> ptarget(nx, 0);
    double pert, I, Ipert;
    
    // DIc/DW

        for(int Wi = 0; Wi < nx; Wi++) // LOOP OVER W
        {
            double h = 1e-8;
            for(int statei = 0; statei < 3; statei++) // LOOP OVER STATEI
            {
                for(int i = 0; i < 3 * nx; i++)
                    Wd[i] = W[i];
                
                pert = W[Wi * 3 + statei] * h;
                Wd[Wi * 3 + statei] = W[Wi * 3 + statei] + pert;
                
                I = evalFitness(dx, W);
                Ipert = evalFitness(dx, Wd);
                
                for(int i = 0; i < 3 * nx; i++)
                    Wd[i] = W[i];
                Wd[Wi * 3 + statei] = W[Wi * 3 + statei] - pert;
                I = evalFitness(dx, Wd);
                
                
                dIcdW[3 * Wi + statei] = (Ipert - I)/(pert * 2);
            }
        }  // END LOOP OVER W
    return dIcdW;
}

// ddIc / dWdW
SparseMatrix <double> evaldIcdWdW(
    std::vector <double> W,
    std::vector <double> dx)
{
    SparseMatrix <double> ddIcdWdW(3 * nx, 3 * nx);
    Matrix3d ddIcdWdW_temp = Matrix3d::Zero();
    Matrix3d ddpdWdW = Matrix3d::Zero();
    Vector3d dpdW = Vector3d::Zero();

    std::vector <double> ptarget(nx, 0);
    double rho, u, p;
    double dxptin2;
    ioTargetPressure(-1, ptarget);
    for(int Wi = 0; Wi < nx; Wi++)
    {
        rho = W[Wi * 3 + 0];
        u = W[Wi * 3 + 1] / rho;
        p = (gam - 1) * ( W[Wi * 3 + 2] - rho * u * u / 2.0 );

        dpdW(0) = (gam - 1) / 2.0 * u * u;
        dpdW(1) = - (gam - 1) * u;
        dpdW(2) = (gam - 1);
        
        ddpdWdW(0, 0) = -u * u * (gam - 1.0) / rho;
        ddpdWdW(1, 1) = (1.0 - gam) / rho;
        ddpdWdW(1, 0) = u * (gam - 1.0) / rho;
        ddpdWdW(0, 1) = ddpdWdW(1, 0);

        dxptin2 = dx[Wi] / pow(ptin, 2);
        ddIcdWdW_temp = dxptin2 * dpdW * dpdW.transpose()
                        + (dxptin2 * p - ptarget[Wi] * dx[Wi] / ptin) * ddpdWdW;

        for(int ki = 0; ki < 3; ki++)
        {
            for(int kj = 0; kj < 3; kj++)
            {
                ddIcdWdW.insert(Wi*3+ki, Wi*3+kj) = ddIcdWdW_temp(ki, kj);
            }
        }
    }
    return ddIcdWdW;
}

VectorXd evaldPLdW(
    std::vector <double> W,
    std::vector <double> dx)
{
    VectorXd dPLdW(3 * nx).setZero;
    VectorXd dPLdWp(3);
    MatrixXd dwpdw(3,3);
    
    dwpdw = (W, nx - 1);

    double rhoout = W[(nx - 1) * 3 + 0];
    double uout = W[(nx - 1) * 3 + 1] / rhoout;
    double pout = (gam - 1) * (W[(nx - 1) * 3 + 2] - rhoout * pow(uout, 2) / 2);
    double ToverTt = 1 - pow(uout, 2) / a2 * (gam - 1) / (gam + 1);
    double poverpt = pow(ToverTt, (gam / (gam - 1)));
    
    double dpdr = (gam - 1) / 2.0 * uout * uout;
    double dpdu = - (gam - 1) * uout;
    double dpde = (gam - 1);
    double dpoverptdu = gam/(gam - 1) * pow(1 - uout * uout / a2 * (gam - 1)/(gam + 1), gam/(gam - 1) - 1) * (-2 * uout/a2 * (gam - 1)/(gam + 1));
    
    dPLdWp[0] = (dpdr * poverpt)/pow(poverpt, 2);
    dPLdWp[1] = (dpdu * poverpt - pout * dpoverptdu)/pow(poverpt, 2);
    dPLdWp[2] = (dpde * poverpt)/pow(poverpt, 2);
    
    dPLdW.bottomRows(3) = dPLdWp * dwpdw;

    return -dPLdW;
}

SparseMatrix <double> evaldPLdWdW(
    std::vector <double> W,
    std::vector <double> dx)
{
    SparseMatrix <double> ddPLdWdW(3 * nx, 3 * nx);
    MatrixXd ddPLdWpodWpo(3, 3);
    MatrixXd ddPLdWodWo(3, 3);
    MatrixXd dwpdw(3,3);
    dwpdw = (W, nx - 1);
    
    double rhoout = W[(nx - 1) * 3 + 0];
    double uout = W[(nx - 1) * 3 + 1] / rhoout;
    double pout = (gam - 1) * (W[(nx - 1) * 3 + 2] - rhoout * pow(uout, 2) / 2);
    double ToverTt = 1 - pow(uout, 2) / a2 * (gam - 1) / (gam + 1);
    double poverpt = pow(ToverTt, (gam / (gam - 1)));
    
    double dpdr = (gam - 1) / 2.0 * uout * uout;
    double dpdu = - (gam - 1) * uout;
    double dpde = (gam - 1);
    double dpoverptdu = gam/(gam - 1) * pow(1 - uout * uout / a2 * (gam - 1)/(gam + 1), gam/(gam - 1) - 1) * (-2 * uout/a2 * (gam - 1)/(gam + 1));
    double ddpoverdudu = gam/(gam - 1) * pow((-2 * uout/a2 * (gam - 1)/(gam + 1)), 2) * (gam/(gam - 1) - 1) * pow(1 - uout * uout / a2 * (gam - 1)/(gam + 1), gam/(gam - 1) - 2) + gam/(gam - 1) * pow(1 - uout * uout / a2 * (gam - 1)/(gam + 1), gam/(gam - 1) - 1) * (-2/a2 * (gam - 1)/(gam + 1));
    
    double dPLdr = (dpdr * poverpt)/pow(poverpt, 2);
    double dPLdu = (dpdu * poverpt - pout * dpoverptdu)/pow(poverpt, 2);
    double dPLde = (dpde * poverpt)/pow(poverpt, 2);
    
    ddPLdrdr = 0;
    ddPLdudu = (pow(poverpt, 2) * (- (gam - 1) * poverpt + dpdu * dpoverptdu - dpdu * dpoverptdu - pout * ddpoverdudu) - (dpdu * poverpt - pout * dpoverptdu) * 2 * poverpt * dpoverptdu)/pow(poverpt, 4);
    ddPLdede = 0;
    ddPLdrdu = (pow(poverpt, 2) * dpdr * dpoverptdu - (dpdr * poverpt) * 2 * poverpt * dpoverptdu)/pow(poverpt, 4);
    ddPLdrde = 0;
    ddPLdedu = (pow(poverpt, 2) * (dpde * dpoverptdu) - (dpde * poverpt) * 2 * poverpt * dpoverptdu)/pow(poverpt, 4);
    
    ddPLdWpodWpo(0, 0) = ddPLdrdr;
    ddPLdWpodWpo(1, 1) = ddPLdudu;
    ddPLdWpodWpo(2, 2) = ddPLdede;
    ddPLdWpodWpo(0, 1) = ddPLdrdu;
    ddPLdWpodWpo(1, 0) = ddPLdrdu;
    ddPLdWpodWpo(0, 2) = ddPLdrde;
    ddPLdWpodWpo(2, 0) = ddPLdrde;
    ddPLdWpodWpo(1, 2) = ddPLdedu;
    ddPLdWpodWpo(2, 1) = ddPLdedu;
    
    ddPLdWodWo = ddPLdWpodWpo * dwpdw;
    
    for(int ki = 0; ki < 3; ki++)
    {
        for(int kj = 0; kj < 3; kj++)
        {
            ddPLdWdW.insert((nx - 1)*3+ki, (nx - 1)*3+kj) = ddPLdWodWo(ki, kj);
        }
    }
    return ddPLdWdW;
}




VectorXd evaldIcdS()
{
    VectorXd dIcdS(nx + 1);
    dIcdS.setZero();
    return dIcdS;
}

MatrixXd evalddIcdSdS()
{
    MatrixXd ddIcdSdS(nx + 1, nx + 1);
    ddIcdSdS.setZero();
    return ddIcdSdS;
}

MatrixXd evalddIcdWdS()
{
    MatrixXd ddIcdWdS(3 * nx, nx + 1);
    ddIcdWdS.setZero();
    return ddIcdWdS;
}
