#include<iostream>
#include<math.h>
#include<vector>
#include<Eigen/Core>
#include<Eigen/Sparse>
#include"petscksp.h"

using namespace Eigen;

MatrixXd solveGMRES(SparseMatrix <double> Ain, MatrixXd Bin)
{
    PetscErrorCode ierr;
    Vec         x, b;
    Mat         A;
    KSP         ksp;
    PC          pc;
    PetscInt    m = Ain.rows(), n = Ain.cols();
    PetscInt    r, c;
    PetscScalar v;
    PetscInt    *indices;
    PetscScalar *values;


    MatCreate(PETSC_COMM_WORLD,&A);
    MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);
    MatSetFromOptions(A);
    MatSetUp(A);

    for(int k = 0; k < Ain.outerSize(); ++k)
    {
        // Iterate over inside
        for(SparseMatrix<double>::InnerIterator it(Ain,k); it; ++it)
        {
            r = it.row();
            c = it.col();
            v = it.value();
            MatSetValues(A, 1, &r, 1, &c, &v, INSERT_VALUES);
        }
    }
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);


    KSPCreate(PETSC_COMM_WORLD,&ksp);

    KSPSetOperators(ksp,A,A);
    KSPGetPC(ksp,&pc);
    PCSetType(pc,PCILU);

    KSPSetType(ksp, KSPGMRES);
    KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
    KSPSetFromOptions(ksp);

    MatrixXd X(Bin.rows(), Bin.cols());
    ierr = PetscMalloc(Bin.cols() * sizeof(PetscScalar), &values);
    ierr = PetscMalloc(Bin.cols() * sizeof(PetscInt), &indices);
    for(int bcol = 0; bcol < Bin.cols(); bcol++)
    {
        VecCreate(PETSC_COMM_WORLD, &b);
        VecSetSizes(b,PETSC_DECIDE, Bin.rows());
        VecSetFromOptions(b);
        VecDuplicate(b, &x);

        for(int brow = 0; brow < Bin.rows(); brow++)
        {
            v = Bin(brow, bcol);
            VecSetValue(b, brow, v, INSERT_VALUES);
        }

        KSPSolve(ksp,b,x);


        for(PetscInt brow = 0; brow < Bin.rows(); brow++)
        {
            values[brow] = -1.0;
            indices[brow] = brow;
        }
        ierr = VecGetValues(x, Bin.rows(), indices, values);
        VecDestroy(&x);
        VecDestroy(&b);

        for(int brow = 0; brow < Bin.rows(); brow++)
        {
            X(brow, bcol) = values[brow];
        }

    }
    MatDestroy(&A);


    return X;
}
