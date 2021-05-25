/** 
 * Header file defining svds performed using the cpu.    
 * @author Tanner Harms
 * @date May 17, 2021
 */

#ifndef SVD_FUNCTIONS
#define SVD_FUNCTIONS

#pragma once

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <sstream>
#include <random>
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
    SVD Superclass
 */

class SVD 
{
protected: 
    MatrixXd* X_;           // Pointer to input matrix
    MatrixXd U_, V_;        // Left and Right Singular Vectors
    VectorXd S_;            // Singular Values
    MatrixXd R_;            // Reconstructed Matrix for verification purposes
    double Spec_Norm;       // Spectral norm
    double Frob_Norm;       // Frobenius norm
    float SVD_Compute_Time; // Time to compute the SVD.  

public:
    // Constructor
    SVD(MatrixXd* Xptr);

    // Destructor
    virtual ~SVD();

    // Accessors
    MatrixXd* inputMatrixPtr(); 
    MatrixXd matrixU(); 
    MatrixXd matrixV(); 
    VectorXd singularValues(); 
    MatrixXd reconstruction(); 
    double spectralNorm();
    double frobeniusNorm();
    float computeTime();

    // Compute the SVD.  This is a pure virtual function
    virtual void ComputeSVD() = 0;

    // Evaluation Functions
    MatrixXd Reconstruct(int rank);         // Get reconstructed data using SVD
    double SpectralNorm(MatrixXd mat);      // Get the spectral norm
    double FrobeniusNorm(MatrixXd mat);     // Get the frobenius norm
    void Evaluate();   
};


/***********************************************************************************/
/* 
    Class for standard SVD computed using a CPU.
 */
class SVD_cpu : public SVD
{
public:
    // Constructor
    SVD_cpu(MatrixXd* InputDataPtr);
    
    // Destructor
    ~SVD_cpu();

    // Compute SVD function
    void ComputeSVD() override;
};

/***********************************************************************************/
/* 
    Class for randomized SVD computed using a CPU.
 */
class rSVD_cpu : public SVD
{
protected:
    svd_params_s rsvd_params;
public:
    // Constructor
    rSVD_cpu(MatrixXd* InputDataPtr, svd_params_s params);
    
    // Destructor
    ~rSVD_cpu();

    // Accessor
    svd_params_s getParams();

    // Mutator
    void setParams(svd_params_s new_params);

    // Compute SVD function
    MatrixXd RangeFinder(int size, int powiter);
    void ComputeSVD() override;
};

#if 0

/***********************************************************************************/
/* 
    Class for standard SVD computed using a GPU.
 */
class SVD_gpu 
{
public:
    // Constructor
    SVD_gpu(MatrixXd InputDataPtr);
    
    // Destructor
    ~SVD_gpu();

    // Compute SVD function
    void ComputeSVD() override;
};


/***********************************************************************************/
/* 
    Class for randomized SVD computed using a GPU.
 */
class rSVD_gpu 
{
protected:
    svd_params_s rsvd_params;

    // Randomized Range Finder
    MatrixXd RangeFinder();
public:
    // Constructor
    rSVD_gpu(MatrixXd InputDataPtr);
    
    // Destructor
    ~rSVD_gpu();

    // Accessor
    svd_params_s getParams();

    // Mutator
    void setParams(svd_params_s new_params);

    // Compute SVD function
    void ComputeSVD(svd_params_s new_params) override;
};


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

#endif
