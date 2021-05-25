/**
 * Performs randomized SVD on an m x n matrix provided by the user.
 * @author Tanner Harms
 * @date May 17th, 2021
 */

#include "helper_cuda.h"
#include "utils.hpp"
#include "svds.hpp"

#include "Eigen/Dense"
#include "helper_cuda.h"

#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixBase;


void PrintResults(SVD& svd_obj)
{
    cout << "U is size " << svd_obj.matrixU().rows() << "-by-" << svd_obj.matrixU().cols() << endl;
    cout << "V is size " << svd_obj.matrixV().rows() << "-by-" << svd_obj.matrixV().cols() << endl;
    cout << "S has " << svd_obj.singularValues().size() << " elements" << endl;
    cout << "Spectral Norm of the error: " << svd_obj.spectralNorm() << endl;
    cout << "Frobenius Norm of the error: " << svd_obj.frobeniusNorm() << endl;
    cout << "Randomized SVD Compute Time on CPU = " << svd_obj.computeTime() << " seconds. \n" << endl;
    return;
}

/********************************************************************************/
/* Main function:
 * Inputs:
 *      input_directory: file containing the data.
 *      rank: desired rank for approximation
 *      q: power iterations
 *      p: oversampling parameter
 */
int main(int argc, char **argv)
{
    // Parse inputs into structures:  if invalid, then usage
    svd_params_s svd_params = parse_args(argc, argv);
    char *data_path = argv[1];

    // testing inputs
    std::cout << "input file is: " << data_path << std::endl;
    std::cout << "target rank: " << svd_params.rank << std::endl;
    std::cout << "number of power iterations: " << svd_params.powiter << std::endl;
    std::cout << "number of oversamples: " << svd_params.oversamples << "\n" << std::endl;

    // Read file to import matrix.  
    data inputData = import_from_file(data_path);
    
    // Get the matrix and print it to the terminal
    MatrixXd mat; 
    mat = inputData.getData_mXd();
    cout << "Input matrix size: " << inputData.getM() << "-by-" << inputData.getN() << "\n" << endl;

    // Perform standard SVD
    SVD_cpu svd_cpu(&mat);
    svd_cpu.Evaluate(svd_params.rank);
    cout << "Standard SVD using the CPU." << endl;
    cout << "Reconstruction rank = " << svd_params.rank << "." << endl;
    PrintResults(svd_cpu);
    
    // Perform rSVD (time it)
    rSVD_cpu rsvd_cpu(&mat, svd_params);
    rsvd_cpu.Evaluate(svd_params.rank);
    cout << "Randomized SVD using the CPU." << endl;
    cout << "Reconstruction rank = " << svd_params.rank << "." << endl;
    PrintResults(rsvd_cpu);
    
    // if (rsvd_cpu.reconstruction()) {
        cout << "R_ size is " <<rsvd_cpu.reconstruction().size() << endl;
    // } else {
    //     cout << "R_ is something else" << endl;
    // }
#if 0

    // Perform standard SVD on GPU (time it)

    // Perform rSVD in GPU (time it)
#endif

    return 0;

} // end main