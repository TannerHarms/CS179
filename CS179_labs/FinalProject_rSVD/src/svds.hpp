/** 
 * Header file defining svds performed using the cpu.    
 * @author Tanner Harms
 * @date May 17, 2021
 */

#ifdef SVD_FUNCTIONS
#define SVD_FUNCTIONS

#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "helper_cuda.h"

#include <vector>
#include <cublas_v2.h>
#include <curand.h>

using std::cout;
using std::endl;
using std::min;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixBase;


/***********************************************************************************/
/* 
    Class for standard SVD computed using a CPU.
 */
class SVD_cpu {
public:
    SVD_cpu(const MatrixXd &X, ...) 
        : U_(), V_(), S_() {
    ComputeSVD_cpu(X,...)
    }

    VectorXd singularValues() { return S_; }
    MatrixXd matrixU() { return U_; }
    MatrixXd matrixV() { return V_; }

private:
    MatrixXd U_, V_;
    VectorXd S_;

    ComputeSVD_cpu(const MatrixXd &X, ...){

    }
}

/***********************************************************************************/
/* 
    Class for standard SVD computed using a GPU.
 */
class SVD_gpu {
public:
    SVD_gpu(const MatrixXd &X, ...) 
        : U_(), V_(), S_() {
    ComputeSVD_gpu(X,...)
    }

    VectorXd singularValues() { return S_; }
    MatrixXd matrixU() { return U_; }
    MatrixXd matrixV() { return V_; }

private:
    MatrixXd U_, V_;
    VectorXd S_;

    ComputeSVD_gpu(const MatrixXd &X, ...){

    }
}

/***********************************************************************************/
/* 
    Class for randomized SVD computed using a CPU.
 */
class SVD_gpu {
public:
    SVD_gpu(const MatrixXd &X, int rank, int oversamples, int powiter) 
        : U_(), V_(), S_() {
    ComputeSVD_gpu(X,...)
    }

    VectorXd singularValues() { return S_; }
    MatrixXd matrixU() { return U_; }
    MatrixXd matrixV() { return V_; }

private:
    MatrixXd U_, V_;
    VectorXd S_;

    ComputeSVD_gpu(const MatrixXd &X, ...){

    }
}

/***********************************************************************************/
/* 
    Class for randomized SVD computed using a GPU.
 */
class SVD_gpu {
public:
    SVD_gpu(const MatrixXd &X, ...) 
        : U_(), V_(), S_() {
    ComputeSVD_gpu(X,...)
    }

    VectorXd singularValues() { return S_; }
    MatrixXd matrixU() { return U_; }
    MatrixXd matrixV() { return V_; }

private:
    MatrixXd U_, V_;
    VectorXd S_;

    ComputeSVD_gpu(const MatrixXd &X, ...){

    }
}


/*
    %% This comes straight from kazuotani14 %%

    Computes spectral norm of error in reconstruction, from SVD matrices.
    Spectral norm = square root of maximum eigenvalue of matrix. 
    Intuitively: the maximum 'scale', by which a matrix can 'stretch' a vector.
    
    Note: The definition of an eigenvalue is for square matrices. For non-square matrices, 
    we can define singular values: 
        Definition: The singular values of a m×n matrix A are the 
        positive square roots of the nonzero eigenvalues of the corresponding matrix A'A. 
        The corresponding eigenvectors are called the singular vectors.
*/
double diff_spectral_norm(MatrixXd A, MatrixXd U, VectorXd S, MatrixXd V, int n_iter=20) {
    int nr = A.rows();

    VectorXd y = VectorXd::Random(nr);
    y.normalize();

    MatrixXd B = (A - U*S.asDiagonal()*V.transpose());

    // TODO make this more efficient (don't explicitly calculate B)
    if(B.rows() != B.cols())
        B = B*B.transpose();

    // Run n iterations of the power method
    // TODO implement and compare fbpca's method
    for(int i=0; i<n_iter; ++i) {
        y = B*y;
        y.normalize();
    }
    double eigval = abs((B*y).dot(y) / y.dot(y));
    if(eigval==0) return 0;

    return sqrt(eigval);
}

double diff_frobenius_norm(MatrixXd A, MatrixXd U, VectorXd S, MatrixXd V) {

}

#endif
