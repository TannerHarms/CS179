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
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

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

svd_params_s parse_args(int argc, char **argv) {
    // make sure it is the right number of inputs.
    if (argc != 5) {
        std::cerr << "Incorrect number of arguments.\n";
        std::cerr << "Usage: ./rSVD <input file> <rank> " <<
            "<power iterations> <oversaples> \n";
        exit(EXIT_FAILURE);
    }
    
    svd_params_s svd_params;

    // Parameters for the RSVD. 
    svd_params.rank = atof(argv[2]);
    svd_params.powiter = atof(argv[3]);
    svd_params.oversamples = atof(argv[4]);

    return svd_params;
}

/********************************************************************************/
// Data utilities

// Data struct
class data {
public:
    // Constructor
    data(int M, int N, double* data) : m(), n(), arr() {
        m = M;
        n = N;
        arr = data;
        matXd = Eigen::Map<Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >(&data[0], m, n);
    }

    // Destructor
    ~data() { delete[] arr; };

    // Accessors
    int getM() { return m; }
    int getN() { return n; }
    double* getData_ptr() { return arr; }
    MatrixXd getData_mXd() { return matXd; }

    // Mutators
    void setM(int numRows) { 
        m = numRows; 
        return;
    }
    void setN(int numCols) { 
        n = numCols; 
        return;
    }
    void setData_Ptr(double *data) { 
        arr = data;
        return;
    }
    void setData_mXd(MatrixXd data) { 
        matXd = data;
        return;
    }

    // function to print the array
    void printData(void) 
    {
        cout << "data is size " << m << "-by-" << n << "\n" << endl;
        
        for (int i = 0; i < m; i++) 
        {
            for (int j = 0; j < n; j++) 
            {
                std::cout << "\t" << arr[j + i * n]; 
            }
            std::cout << std::endl;
        }
    }

private:
    int m, n;
    double* arr;
    MatrixXd matXd;
};

// Import Data from Text File
data import_from_file(char *file_name) {
    
    // Open the input file for reading
    fstream inputFile(file_name);

    // Initialize Variables
    int i = 0, j = 0, n = 0;    // array sizes and indices
    std::string line;                       // line variable
    std::vector<std::vector<double>> vec_arr;           // the array

    // loop through all lines of text file.
    while (std::getline(inputFile, line))   // get lines while they come
    {
        // Initiate values
        double value;
        
        // Get istringstream instance
        std::istringstream fline(line);

        // add a row to the array
        vec_arr.push_back(vector<double>());
        
        // iterate through the line
        j = 0;
        while (fline >> value) // get values while they come
        {
            // add a new element to the row
            vec_arr[i].push_back(value); 
            j++;
        }
        i++;

        // set the value of n
        if (n == 0) n = j; // number of columns
        else if (n != j) {       // error if row size does not match
            cerr << "Error line " << i << " - " << j << " values instead of " << n << endl;
        }
    }
    int m = i;

    // Convert vector array to double array
    double *arr;
    arr = new double[m*n];
    for (int i = 0; (i < m); i++)
    {
        // arr[i] = new double[n];
        for (int j = 0; (j < n); j++)
        {
            // std::cout << vec_arr[i][j] << endl;
            arr[j + i*n] = vec_arr[i][j];
        }
    }

    // Initialize output structure
    data inputData(m, n, arr);

    return inputData;
}

/********************************************************************************/
// Import Data from Python.


/********************************************************************************/
// Timing utilities
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
      gpuErrchk(cudaEventCreate(&start));       \
      gpuErrchk(cudaEventCreate(&stop));        \
      gpuErrchk(cudaEventRecord(start));        \
    }

#define STOP_RECORD_TIMER(name) {                           \
      gpuErrchk(cudaEventRecord(stop));                     \
      gpuErrchk(cudaEventSynchronize(stop));                \
      gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
      gpuErrchk(cudaEventDestroy(start));                   \
      gpuErrchk(cudaEventDestroy(stop));                    \
    }

#endif