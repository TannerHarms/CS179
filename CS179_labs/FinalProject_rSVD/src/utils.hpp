#ifndef UTILS
#define UTILS

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cassert>
#include "Eigen/Dense"

#include <cuda_runtime.h>

#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixBase;

/********************************************************************************/
// Error Utilities
/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

/********************************************************************************/
// Argument utilities
struct svd_params_s {
    float rank;
    float powiter;
    float oversamples;
};

svd_params_s parse_args(int argc, char **argv);

/********************************************************************************/
// Data utilities

// Data struct
class data {
public:
    // Constructor
    data(int M, int N, double* data);

    // Destructor
    ~data();

    // Accessors
    int getM();
    int getN();
    double* getData_ptr();
    MatrixXd getData_mXd();

    // Mutators
    void setM(int numRows);
    void setN(int numCols);
    void setData_Ptr(double *data);
    void setData_mXd(MatrixXd data);

    // function to print the array
    void printData(void);

private:
    int m, n;
    double* arr;
    MatrixXd matXd;
};

// Import Data from Text File
data import_from_file(char *file_name);

/********************************************************************************/
// Matrix printing utility for testing GPU SVD results
void printMatrix(int m, int n, const double*A, int lda, const char* name);

#endif