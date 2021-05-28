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
#include <cmath>
#include <vector>

#include <curand.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "Eigen/Dense"

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
    void Reconstruct(int rank);             // Get reconstructed data using SVD
    double SpectralNorm(MatrixXd mat, int n_iter);      // Get the spectral norm
    double FrobeniusNorm(MatrixXd mat);     // Get the frobenius norm
    void Evaluate(int rank);   
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


/***********************************************************************************/
/* 
    Class for standard SVD computed using a GPU.
 */
class SVD_gpu : public SVD
{
public:
    // Constructor
    SVD_gpu(MatrixXd* InputDataPtr);
    
    // Destructor
    ~SVD_gpu();

    // Compute SVD function
    void ComputeSVD() override;
};


/***********************************************************************************/
/* 
    Class for randomized SVD computed using a GPU.
 */
class rSVD_gpu : public SVD
{
protected:
    svd_params_s rsvd_params;

public:
    // Constructor
    rSVD_gpu(MatrixXd* InputDataPtr, svd_params_s params);
    
    // Destructor
    ~rSVD_gpu();

    // Accessor
    svd_params_s getParams();

    // Mutator
    void setParams(svd_params_s new_params);

    // Compute SVD function
    // void TestMatrix(double* A, int nr_rows_A, int nr_cols_A);
    void ComputeSVD() override;
};

#endif
