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
    std::cout << mat << std::endl;
    
    // set up floats to store the time
    //float time_SVD_cpu;//, time_SVD_gpu;
    // float time_rSVD_cpu, time_rSVD_gpu;

    // Perform standard SVD
    SVD_cpu svd_cpu(&mat);
    cout << "Standard SVD Compute Time on CPU = " << svd_cpu.computeTime() << " seconds." << endl;

    // Perform rSVD (time it)
    rSVD_cpu rsvd_cpu(&mat, svd_params);
    cout << "Randomized SVD Compute Time on CPU = " << rsvd_cpu.computeTime() << " seconds." << endl;

#if 0

    // Perform standard SVD on GPU (time it)

    // Perform rSVD in GPU (time it)
#endif

    return 0;

} // end main