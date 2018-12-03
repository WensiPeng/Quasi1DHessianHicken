#include<iostream>
#include<fstream>
#include<math.h>
#include<vector>
#include<iomanip>
#include<Eigen/Dense>
#include<Eigen/Cholesky>
#include<Eigen/SparseCholesky>
#include<stdlib.h>//exit
#include"quasiOneD.h"
#include"fitness.h"
#include"grid.h"
#include"adjoint.h"
#include"directDifferentiation.h"
#include"globals.h"
#include"gradient.h"
#include"analyticHessian.h"
#include"getHessianVectorProduct.h"
#include"output.h"
#include<time.h>
#include<stdlib.h>     /* srand, rand */

using namespace Eigen;

MatrixXd finiteD2g(
    std::vector<double> x,
    std::vector<double> dx,
    std::vector<double> area,
    std::vector<double> designVar,
    double h);
MatrixXd finiteD2(std::vector<double> x,
    std::vector<double> dx,
    std::vector<double> area,
    std::vector<double> designVar,
    double h,
    double currentI);

double stepBacktrackUncons(
    double alpha,
    std::vector<double> &designVar,
    VectorXd &searchD,
    VectorXd pk,
    VectorXd gradient,
    double currentI,
    std::vector<double> x,
    std::vector<double> dx,
    std::vector<double> &W);

MatrixXd BFGS(
    MatrixXd oldH,
    VectorXd oldg,
    VectorXd currentg,
    VectorXd searchD);

MatrixXd GMRES(
    std::vector <double> x,
    std::vector <double> dx,
    std::vector <double> W,
    std::vector <double> S,
    std::vector <double> designVar,
    VectorXd RHS,
    double tol);

MatrixXd CG(
            std::vector <double> x,
            std::vector <double> dx,
            std::vector <double> W,
            std::vector <double> S,
            std::vector <double> designVar,
            VectorXd RHS,
            double tol);

double checkCond(MatrixXd H);
MatrixXd invertHessian(MatrixXd H);
LLT<MatrixXd> checkPosDef(MatrixXd H);

VectorXd implicitSmoothing(VectorXd gradient, double epsilon);

void test_grad(int gradientType1, int gradientType2, double currentI, 
	std::vector<double> x, std::vector<double> dx, 
	std::vector<double> area, std::vector<double> W, 
	std::vector<double> designVar, VectorXd psi)
{
	printf("Comparing Adjoint, Direct-Differentiation, and Central FD\n");
	int n_des = designVar.size();
    VectorXd adjoint_gradient(n_des);
    VectorXd direct_gradient(n_des);
    VectorXd finite_gradient(n_des);
    adjoint_gradient = getGradient(1, currentI, x, dx, area, W, designVar, psi);
    direct_gradient = getGradient(2, currentI, x, dx, area, W, designVar, psi);
    finite_gradient = getGradient(-3, currentI, x, dx, area, W, designVar, psi);

	printf(" Adjoint                  Direct-Diff             Central FD              AD-DA                   AD-FD                   DA-FD\n");
	for (int i = 0; i < n_des; i++) {
		double g1 = adjoint_gradient[i];
		double g2 = direct_gradient[i];
		double g3 = finite_gradient[i];
		double r1 = (g1-g2)/(g1+1e-15);
		double r2 = (g1-g3)/(g1+1e-15);
		double r3 = (g2-g3)/(g2+1e-15);
		printf("%23.15e %23.15e %23.15e %23.15e %23.15e %23.15e\n", g1, g2, g3, r1, r2, r3);
	}
	return;
}

void test_hessian(double currentI,
	std::vector<double> x, std::vector<double> dx, 
	std::vector<double> area, std::vector<double> W, 
	std::vector<double> designVar)
{
    exactHessian = 1; // Calculate exact Hessian to compare with BFGS
    MatrixXd directAdjoint = getAnalyticHessian(x, dx, W, area, designVar, 3);
    double err;
    std::cout<<"DA: "<<directAdjoint<<std::endl;

    MatrixXd H;
    for (int i = 3; i < 8; i++) {
        double h = pow(10,-i);
        std::cout<<"h = "<<h<<std::endl;
    std::cout.setstate(std::ios_base::failbit);
        MatrixXd H = finiteD2g(x, dx, area, designVar, h);
    std::cout.clear();
        err = (directAdjoint - H).norm()/directAdjoint.norm();
        std::cout<<"DA - FDg: "<<err<<std::endl;
        //std::cout<<std::setprecision(15)<<H<<std::endl;

    std::cout.setstate(std::ios_base::failbit);
        H = finiteD2(x, dx, area, designVar, h, currentI);
    std::cout.clear();
        err = (directAdjoint - H).norm()/directAdjoint.norm();
        std::cout<<"DA - FD: "<<err<<std::endl;
        //std::cout<<std::setprecision(15)<<H<<std::endl;
    }

    H= getAnalyticHessian(x, dx, W, area, designVar, 1);
    err = (directAdjoint - H).norm()/directAdjoint.norm();
    std::cout<<"DA - AD: "<<err<<std::endl;
    //std::cout<<std::setprecision(15)<<H<<std::endl;

    H= getAnalyticHessian(x, dx, W, area, designVar, 2);
    err = (directAdjoint - H).norm()/directAdjoint.norm();
    std::cout<<"DA - AA: "<<err<<std::endl;
    //std::cout<<std::setprecision(15)<<H<<std::endl;


    H= getAnalyticHessian(x, dx, W, area, designVar, 0);
    err = (directAdjoint - H).norm()/directAdjoint.norm();
    std::cout<<"DA - DD: "<<err<<std::endl;
    //std::cout<<std::setprecision(15)<<H<<std::endl;
    return;
}
void design(
    std::vector<double> x, std::vector<double> dx,
    std::vector<double> area, std::vector<double> designVar)
{
    std::vector<double> W(3 * nx, 0);

    std::vector<double> normGradList;
    std::vector<double> timeVec;
    std::vector<double> Herror;
    std::vector<double> svdvalues;
    std::vector<double> svdvaluesreal;
    std::vector<double> Hcond;
    MatrixXd H(nDesVar, nDesVar), H_BFGS(nDesVar, nDesVar), realH(nDesVar, nDesVar);
    double normGrad;
    double currentI;
    double alpha;

    int printConv = 1;

    VectorXd pk(nDesVar), searchD(nDesVar);

    clock_t tic = clock();
    clock_t toc;
    double elapsed = 0;
    
    timeVec.push_back(elapsed);
    
    std::ofstream myfile;
    myfile.open("convergence.dat");
    myfile << " Iteration \t Cost Function \t Gradient Norm \t Average Error \n";

    quasiOneD(x, area, W);
    currentI = evalFitness(dx, W);

    VectorXd psi(3 * nx);

    VectorXd gradient(nDesVar);
    VectorXd oldGrad(nDesVar); //BFGS
    gradient = getGradient(gradientType, currentI, x, dx, area, W, designVar, psi);

	bool testingGradient = false;
	//#testingGradient = false;
	if (testingGradient) {
		test_grad(gradientType, -3, currentI, x, dx, area, W, designVar, psi);
        test_hessian(currentI, x, dx, area, W, designVar);
		exit(0);
	}

    // Initialize B
    H.setIdentity();
    H = H * 1.0;
    if (exactHessian == 1 || exactHessian == -1)
    {
        //std::cout<<"exact Hessian use "<<hessianType<<std::endl;
        H = getAnalyticHessian(x, dx, W, area, designVar, hessianType);
//      H = finiteD2(x, dx, area, designVar, h, currentI);
        checkCond(H);
        H = invertHessian(H);
    }

    normGrad = 0;
    for (int i = 0; i < nDesVar; i++)
        normGrad += pow(gradient[i], 2);
    normGrad = sqrt(normGrad);
    normGradList.push_back(normGrad);
    int iDesign = 0;

    // Design Loop
    while(normGrad > gradConv && iDesign < maxDesign)
    {
        iDesign++ ;

        if (printConv == 1)
        {
            std::cout<<"Iteration :"<<iDesign<<
                "    GradientNorm: "<<normGrad<<std::endl;
            std::cout<<"Current Design:\n";
            for (int i = 0; i < nDesVar; i++)
                std::cout<<designVar[i]<<std::endl;

//          std::cout<<"Current Shape:\n";
//          for (int i = 0; i < nx + 1; i++)
//              std::cout<<area[i]<<std::endl;
        }
        std::cout<<"Current Fitness: "<<currentI<<std::endl;

//      1  =  Steepest Descent
//      2  =  Quasi-Newton (BFGS)
//      3  =  Newton
//      4  =  Truncated Newton with Adjoint-Direct Matrix-Vector Product
        if (descentType == 1)
        {
            //int expo = rand() % 5 + 1 - 3;
            //double averageerr = 0;
            //for (int i = 0; i<nDesVar; i++) {
            //    srand (time(NULL));
            //    double fMin = -2.0;
            //    double fMax = 2.0;
            //    double expo = (double)rand() / RAND_MAX;
            //    expo =  fMin + expo * (fMax - fMin);
            //    expo =  0.0;
            //    pk(i) =  -10*gradient(i)*pow(10,expo);
            //    averageerr += fabs(expo)/nDesVar;
            //    //pk(i) =  -gradient(i);
            //}
            //myfile << iDesign << "\t" << currentI <<"\t"<< normGrad << "\t" << averageerr << "\n";
            //myfile.flush();
            pk =  -500*gradient;
        }
        else if (descentType == 2)
        {
            if (iDesign > 1)
            {
                H_BFGS = BFGS(H, oldGrad, gradient, searchD);
                H = H_BFGS;
            }

//          realH = getAnalyticHessian(x, dx, W, area, designVar, 2);
//          JacobiSVD<MatrixXd> svd1(H.inverse(), ComputeFullU | ComputeFullV);
//          JacobiSVD<MatrixXd> svd2(realH, ComputeFullU | ComputeFullV);

//          std::cout<<"svd1"<<std::endl;
//          std::cout<<svd1.singularValues()<<std::endl;
//          std::cout<<"svd2"<<std::endl;
//          std::cout<<svd2.singularValues()<<std::endl;
//          for (int i = 0; i < nDesVar; i++)
//          {
//              svdvalues.push_back(svd1.singularValues()(i));
//              svdvaluesreal.push_back(svd2.singularValues()(i));
//          }
//
//          std::cout<<"svd singular values error"<<std::endl;
//          std::cout<<
//              (svd1.singularValues()-svd2.singularValues()).norm()
//              /svd2.singularValues().norm()<<std::endl;

//          std::cout<<"svd singular vectors error"<<std::endl;
//          std::cout<<
//              (svd1.matrixV()-svd2.matrixV()).norm()
//              /svd2.matrixV().norm()<<std::endl;
            // Eigenvalues are not returned in ascending order.
//          std::cout<<"eig1"<<std::endl;
//          std::cout<<H.inverse().eigenvalues()<<std::endl;
//          std::cout<<"eig2"<<std::endl;
//          std::cout<<realH.eigenvalues()<<std::endl;
//          std::cout<<"Eig error"<<std::endl;
//          std::cout<<
//              (H.inverse().eigenvalues()-realH.eigenvalues()).norm()
//              /realH.eigenvalues().norm()<<std::endl;

//          realH = realH.inverse();
//          Hcond.push_back(checkCond(realH.inverse()));
//          double err = (realH - H).norm()/realH.norm();
//          std::cout<<"Hessian error: "<<err<<std::endl;
//          Herror.push_back(err);

            pk = -H * gradient;
        }
        else if (descentType == 3)
        {
            H = getAnalyticHessian(x, dx, W, area, designVar, hessianType);
            realH = getAnalyticHessian(x, dx, W, area, designVar, 2);
            double err = (realH - H).norm()/realH.norm();
            std::cout<<"Hessian error: "<<err<<std::endl;
            Herror.push_back(err);

            checkCond(H);
            H = invertHessian(H);
            Hcond.push_back(checkCond(realH.inverse()));

            pk = -H * gradient;
        }
        
        else if (descentType == 4)
        {
            VectorXd vecW(nDesVar);
            VectorXd HVP(nDesVar);
            VectorXd realHVP(nDesVar);
            MatrixXd realH(nDesVar,nDesVar);
            HVP.setZero();
            //VectorXd AH(3 * nx);
            //for(int i = 0; i < nDesVar; i++)
              //  vecW[i] = 1;
            //tol = 0.1 * 0.1/nDesVar;
            exactHessian = 1;
            realH = getAnalyticHessian(x, dx, W, area, designVar, 3);
            std::cout<<"eigenvalues = "<<std::endl;
            EigenSolver<MatrixXd> es(realH);
            std::cout<<es.eigenvalues()<<std::endl;
            //realHVP = realH * vecW;
            //HVP = getHessianVectorProduct(x,dx,W,area,designVar,vecW);
            pk = GMRES(x,dx,W,area,designVar,-gradient,newtonTol);
        }
      
        else if (descentType == 5)
        {
            //MatrixXd realH(nDesVar,nDesVar);
            //exactHessian = 1;
            //realH = getAnalyticHessian(x, dx, W, area, designVar, 3);
            //std::cout<<"eigenvalues = "<<std::endl;
            //EigenSolver<MatrixXd> es(realH);
            //std::cout<<es.eigenvalues()<<std::endl;
            pk = CG(x,dx,W,area,designVar,-gradient,newtonTol);
         }
      
//        std::cout<<realH<<std::endl;
        
//      std::cout<<"pk before smoothing:\n"<<std::endl;
//      std::cout<<pk<<std::endl;

//      pk = implicitSmoothing(pk, 0.5);
        alpha = 1.0;

        std::cout<<"gradient:\n"<<std::endl;
        std::cout<<-gradient<<std::endl;
        std::cout<<"pk:\n"<<std::endl;
        std::cout<<pk<<std::endl;

        currentI = stepBacktrackUncons(alpha, designVar, searchD, pk, gradient, currentI, x, dx, W);

        area = evalS(designVar, x, dx, desParam);
        oldGrad = gradient;
        gradient = getGradient(gradientType, currentI, x, dx, area, W, designVar, psi);

        normGrad = 0;
        for (int i = 0; i < nDesVar; i++)
            normGrad += pow(gradient[i], 2);
        normGrad = sqrt(normGrad);
        normGradList.push_back(normGrad);

        toc = clock();
        elapsed = (double)(toc-tic) / CLOCKS_PER_SEC;
        timeVec.push_back(elapsed);
        std::cout<<"Time: "<<elapsed<<std::endl;

        std::cout<<"End of Design Iteration: "<<iDesign<<std::endl<<std::endl<<std::endl;
    }

    std::cout<<"Final Gradient:"<<std::endl;
    std::cout<<gradient<<std::endl;

    std::cout<<std::endl<<"Final Design:"<<std::endl;
    for (int i = 0; i < nDesVar; i++)
        std::cout<<designVar[i]<<std::endl;

    std::cout<<"Fitness: "<<evalFitness(dx, W)<<std::endl;
    
    if (descentType == 5 || descentType ==4) {
        if (newtonTol == 0.5e-1){
            outVec("OptConv.dat", "w", normGradList);
            outVec("OptTime.dat", "w", timeVec);
        }
        else if (newtonTol == 1e-2){
            outVec("OptConv2.dat", "w", normGradList);
            outVec("OptTime2.dat", "w", timeVec);
        }
        else if (newtonTol == 1e-3){
            outVec("OptConv3.dat", "w", normGradList);
            outVec("OptTime3.dat", "w", timeVec);
        }
        else if (newtonTol == 0.8e-1){
            outVec("OptConv4.dat", "w", normGradList);
            outVec("OptTime4.dat", "w", timeVec);
        }
        else if (newtonTol < 1e-8){
            outVec("OptConv5.dat", "w", normGradList);
            outVec("OptTime5.dat", "w", timeVec);
        }
    }
    
    if (exactHessian == 1) {
        if (hessianType == 2) {
            outVec("OptConvEH.dat", "w", normGradList);
            outVec("OptTimeEH.dat", "w", timeVec);
        }
        if (exactHessian == -1) {
            outVec("OptConvAH.dat", "w", normGradList);
            outVec("OptTimeAH.dat", "w", timeVec);
        }
        
    }

    
    outVec("HessianErr.dat", "w", Herror);
    outVec("HessianCond.dat", "w", Hcond);
    outVec("svd.dat", "w", svdvalues);
    outVec("svdreal.dat", "w", svdvaluesreal);

    myfile.close();

    return;
}

double stepBacktrackUncons(
    double alpha,
    std::vector<double> &designVar,
    VectorXd &searchD,
    VectorXd pk,
    VectorXd gradient,
    double currentI,
    std::vector<double> x,
    std::vector<double> dx,
    std::vector<double> &W)
{
//  std::vector<double> W(3 * nx, 0);

    double c1 = 1e-4;
    std::vector<double> tempS(nx + 1);
    double newVal;

    double c_pk_grad = 0;
    double minN;
    double minD = 0;
   // if (pk.minCoeff() < -1) {
   //     pk = pk/abs(pk.minCoeff());
    //    std::cout<<"normalize pk"<<std::endl;
   // }

    c_pk_grad = c1 * gradient.dot(pk);

    std::vector<double> tempD(nDesVar);
    //for (int i = 0; i < nDesVar; i++)
    //{
        //tempD[i] = designVar[i] + alpha * pk[i];
        //if (tempD[i]<minD) {
        //    minD = tempD[i];
        //    minN = i;
       // }
    //}
    
    //if (minD < 0) {
    //    //pk = pk * (abs(designVar[minN]/pk[minN]));
    //    pk = pk/pk.norm();
    //    std::cout<<"normalize pk :"<<std::endl;
     //   std::cout<<pk<<std::endl;
    //}
    
    for (int i = 0; i < nDesVar; i++)
    {
        tempD[i] = designVar[i] + alpha * pk[i];
    }
    tempS = evalS(tempD, x, dx, desParam);
    double scalePk = quasiOneD(x, tempS, W);
    while (scalePk == 0) {
        std::cout<<"scale Pk"<<std::endl;
        pk = pk/2;
        for (int i = 0; i < nDesVar; i++)
        {
            tempD[i] = designVar[i] + alpha * pk[i];
        }
        tempS = evalS(tempD, x, dx, desParam);
        scalePk = quasiOneD(x, tempS, W);
    }
    
    newVal = evalFitness(dx, W);

    while(newVal > (currentI + alpha * c_pk_grad) && alpha > 1e-16)
    {
        alpha = alpha * 0.5;
        std::cout<<"Alpha Reduction: "<<alpha<<std::endl;

        for (int i = 0; i < nDesVar; i++)
            tempD[i] = designVar[i] + alpha * pk[i];
        tempS = evalS(tempD, x, dx, desParam);
        quasiOneD(x, tempS, W);
        newVal = evalFitness(dx, W);
        std::cout<<"newVal: "<<newVal<<std::endl;
        std::cout<<"currentI + alpha/2.0 * c_pk_grad: "<<
        currentI + alpha/ 2.0 * c_pk_grad<<std::endl;
    }
    if (alpha < 1e-16) std::cout<<"Error. Can't find step size"<<std::endl;

    designVar = tempD;
    for (int i = 0; i < nDesVar; i++)
    {
        searchD[i] = alpha * pk[i];
    }

    return newVal;
}

MatrixXd finiteD2g(
    std::vector<double> x,
    std::vector<double> dx,
    std::vector<double> area,
    std::vector<double> designVar,
    double h)
{
    MatrixXd Hessian(nDesVar, nDesVar);
    std::vector<double> W(3 * nx, 0);
    std::vector<double> tempS(nx + 1);
    std::vector<double> tempD(nDesVar);
    VectorXd gradp(nDesVar);
    VectorXd gradn(nDesVar);
    VectorXd psi(3 * nx);

    double currentI = -1;
    Hessian.setZero();
    for (int i = 0; i < nDesVar; i++)
    {
        tempD = designVar;
        tempD[i] += h;
        tempS = evalS(tempD, x, dx, desParam);
        quasiOneD(x, tempS, W);
        gradp = getGradient(1, currentI, x, dx, tempS, W, tempD, psi);

        tempD = designVar;
        tempD[i] -= h;
        tempS = evalS(tempD, x, dx, desParam);
        quasiOneD(x, tempS, W);
        gradn = getGradient(1, currentI, x, dx, tempS, W, tempD, psi);
        for (int j = 0; j < nDesVar; j++)
        {
            Hessian(i, j) += (gradp(j) - gradn(j)) / (2*h);
            Hessian(j, i) += (gradp(j) - gradn(j)) / (2*h);
        }
    }
    Hessian = Hessian / 2.0;
    return Hessian;
}


MatrixXd finiteD2(
    std::vector<double> x,
    std::vector<double> dx,
    std::vector<double> area,
    std::vector<double> designVar,
    double h,
    double currentI)
{
    std::vector<double> W(3 * nx, 0);
    MatrixXd Hessian(nDesVar, nDesVar);
    std::vector<double> tempS(nx + 1);

    double I, I1, I2, I3, I4, dhi, dhj;

    std::vector<double> tempD(nDesVar);

    if (currentI < 0 && gradientType != 3)
    {
        quasiOneD(x, area, W);
        I = evalFitness(dx, W);
    }
    else
    {
        I = currentI;
    }
    for (int i = 0; i < nDesVar; i++)
    for (int j = i; j < nDesVar; j++)
    {
        dhi = designVar[i] * h;
        dhj = designVar[j] * h;
        if (i == j) {
            tempD = designVar;
            tempD[i] += dhi;
            tempD[j] += dhj;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I1 = evalFitness(dx, W);

            tempD = designVar;
            tempD[i] += dhi;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I2 = evalFitness(dx, W);

            tempD = designVar;
            tempD[i] -= dhi;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I3 = evalFitness(dx, W);

            tempD = designVar;
            tempD[i] -= dhi;
            tempD[j] -= dhj;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I4 = evalFitness(dx, W);
            Hessian(i, j) = (-I1 + 16*I2 - 30*I + 16*I3 - I4) / (12 * dhi * dhj);
        } else {
            tempD = designVar;
            tempD[i] += dhi;
            tempD[j] += dhj;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I1 = evalFitness(dx, W);

            tempD = designVar;
            tempD[i] += dhi;
            tempD[j] -= dhj;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I2 = evalFitness(dx, W);

            tempD = designVar;
            tempD[i] -= dhi;
            tempD[j] += dhj;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I3 = evalFitness(dx, W);

            tempD = designVar;
            tempD[i] -= dhi;
            tempD[j] -= dhj;
            tempS = evalS(tempD, x, dx, desParam);
            quasiOneD(x, tempS, W);
            I4 = evalFitness(dx, W);

            Hessian(i, j) = (I1 - I2 - I3 + I4) / (4 * dhi * dhj);
            Hessian(j, i) = Hessian(i, j);
        }
    }
    //Hessian = Hessian + Hessian.transposeInPlace();
    //Hessian = Hessian / 2.0;
    return Hessian;
}

double checkCond(MatrixXd H) {
    JacobiSVD<MatrixXd> svd(H);
    double svdmax = svd.singularValues()(0);
    double svdmin = svd.singularValues()(svd.singularValues().size()-1);
    double cond = svdmax / svdmin;
    std::cout<<"Condition Number of H:"<<std::endl;
    std::cout<<cond<<std::endl;

    return cond;
}

MatrixXd invertHessian(MatrixXd H) {
    LLT<MatrixXd> llt = checkPosDef(H);
    return llt.solve(MatrixXd::Identity(H.rows(), H.rows()));
}

LLT<MatrixXd> checkPosDef(MatrixXd H) {
    LLT<MatrixXd> llt;
    VectorXcd eigval = H.eigenvalues();
    double shift = 1e-5;
    if (eigval.real().minCoeff() < 0) {
        MatrixXd eye(H.rows(),H.rows());
        eye.setIdentity();
        std::cout<<"Matrix is not Positive Semi-Definite"<<std::endl;
        std::cout<<"Eigenvalues:"<<std::endl;
        std::cout<<eigval<<std::endl;
        llt.compute(H + (shift - eigval.real().minCoeff()) * eye);
        checkCond(H + (shift - eigval.real().minCoeff()) * eye);
    }
    else {
        llt.compute(H);
    }
    return llt;
}

MatrixXd BFGS(
    MatrixXd oldH,
    VectorXd oldg,
    VectorXd currentg,
    VectorXd searchD)
{
    MatrixXd newH(nDesVar, nDesVar);
    VectorXd dg(nDesVar), dx(nDesVar);
    MatrixXd dH(nDesVar, nDesVar), a(nDesVar, nDesVar), b(nDesVar, nDesVar);

    dg = currentg - oldg;
    dx = searchD;

    a = ((dx.transpose() * dg + dg.transpose() * oldH * dg)(0) * (dx * dx.transpose()))
         / ((dx.transpose() * dg)(0) * (dx.transpose() * dg)(0));
    b = (oldH * dg * dx.transpose() + dx * dg.transpose() * oldH) / (dx.transpose() * dg)(0);

    dH = a - b;

    newH = oldH + dH;

    return newH;
}


MatrixXd CG(
            std::vector <double> x,
            std::vector <double> dx,
            std::vector <double> W,
            std::vector <double> S,
            std::vector <double> designVar,
            VectorXd RHS,
            double tol){
    std::vector<double> normResList;
    VectorXd pk(nDesVar), pkold(nDesVar);
    pk.setZero();
    VectorXd r = RHS;
    VectorXd rold(nDesVar);
    VectorXd d = r;
    VectorXd q(nDesVar);
    int k;
    double alpha, beta, delta;
    delta = r.dot(r);
    
    for (int i = 0; r.norm() > RHS.norm() * tol; i++) {
        pkold = pk;
        q = getHessianVectorProduct(x, dx, W, S, designVar, d);
        if (d.dot(q) < 0) {
            if (i == 0) {
                return d;
            } else {
                std::cout<<"at iteration = "<<i<<std::endl;
                std::cout<<"indefite Hessian, return pk"<<std::endl;
                std::cout<<"residual = "<<r.norm()/RHS.norm()<<std::endl;
                return pkold;
            }
        }
        alpha = (r.dot(r)) / (d.dot(q));
        pk += alpha * d;
        rold = r;
        r -= alpha * q;
        beta = (r.dot(r))/(rold.dot(rold));
        d = r + beta * d;
        delta = r.dot(r) + beta * beta * delta;
        normResList.push_back(r.norm()/RHS.norm());
        outVec("CGconv.dat", "w", normResList);
        k = i;
        //std::cout<<"at iteration = "<<i<<std::endl;
        //std::cout<<"residual = "<<r.norm()<<std::endl;
    }
    std::cout<<"at iteration = "<<k<<std::endl;
    std::cout<<"residual = "<<r.norm()/RHS.norm()<<std::endl;
    return pk;
}

MatrixXd GMRES(
               std::vector <double> x,
               std::vector <double> dx,
               std::vector <double> W,
               std::vector <double> S,
               std::vector <double> designVar,
               VectorXd RHS,
               double tol)
{   std::vector<double> normResList;
    MatrixXd v(nDesVar, nDesVar+1);
    //MatrixXd v(nDesVar + 2, nDesVar+3);
    VectorXd vecW(nDesVar);
    MatrixXd b(nDesVar+2, nDesVar+1);
    //MatrixXd b(nDesVar+4, nDesVar+3);
    VectorXd r(nDesVar);
    VectorXd xk(nDesVar);
    VectorXd vtemp(nDesVar);
    int k;
    r[0] = 1;
    v.setZero();
    b.setZero();
    v.col(0) = - RHS/RHS.norm();
    for(k=0 ; k < nDesVar && r.norm() > RHS.norm() * tol ; k++)
    {
        //       std::cout<<"k = "<<k<<std::endl;
        MatrixXd B(k+2, k+1);
        VectorXd HVP(nDesVar);
        B.setZero();
        B.topLeftCorner(k+1, k) = b.topLeftCorner(k+1, k);
        MatrixXd V(nDesVar, k+2);
        V.setZero();
        V.leftCols(k+1) = v.leftCols(k+1);
        vecW = V.col(k);
        
        HVP = getHessianVectorProduct(x, dx, W, S, designVar, vecW);
        if (HVP.dot(vecW) < 0) {
            std::cout<<"at iteration NO : "<<k<<std::endl;
            std::cout<<"indefinite Hessian, return current vk"<<std::endl;
            return vecW;
        }
        B.topRightCorner(k+1,1) = V.leftCols(k+1).transpose() * HVP;

        vtemp = getHessianVectorProduct(x, dx, W, S, designVar, vecW) - V.leftCols(k+1) * B.topRightCorner(k+1,1);
        V.col(k+1) = vtemp / vtemp.norm();
        
        
        B(k+1,k) = vtemp.norm();
        b.topLeftCorner(k+2, k+1) = B;
        
        v.leftCols(k+2) = V;
        if (k > 0)
        {
            VectorXd y(k+1);
            VectorXd e1(k+2);
            e1.setZero();
            e1.row(0) << 1;
            y = B.bdcSvd(ComputeThinU | ComputeThinV).solve(-RHS.norm() * e1);
            xk = V.leftCols(k+1) * y;
            //r = RHS - V.leftCols(k+1) * B.topLeftCorner(k+1, k) * y;
            r = getHessianVectorProduct(x, dx, W, S, designVar, xk) - RHS;
            normResList.push_back(r.norm()/RHS.norm());
            outVec("GMRESconv.dat", "w", normResList);
        }
        //
    }
    std::cout<<k<<"iteration, residual = "<<r.norm()<<std::endl;
    if (getHessianVectorProduct(x, dx, W, S, designVar, vecW).dot(vecW) < 0) {
        std::cout<<"at iteration NO : "<<k<<std::endl;
        std::cout<<"indefinite Hessian, return current vk"<<std::endl;
        return vecW;
    }

    return xk;
    
}

VectorXd implicitSmoothing(VectorXd gradient, double epsilon)
{
    int n = gradient.size();
    MatrixXd A(n, n);
    A.setZero();

    for (int i = 0; i < n-1; i++) {
        A(i  , i) = 1.0 + 2.0 * epsilon;
        A(i+1, i) = -epsilon;
        A(i, i+1) = -epsilon;
    }
    A(n-1,n-1) = 1.0 + 2.0 * epsilon;

    LLT<MatrixXd> llt;
    llt.compute(A);
    if (llt.info() != 0)
        std::cout<<"Factorization failed. Error: "<<llt.info()<<std::endl;
    VectorXd smoothGrad = llt.solve(gradient);

    return smoothGrad;
}
