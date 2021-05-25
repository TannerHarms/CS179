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
    float time_SVD_cpu;//, time_SVD_gpu;
    // float time_rSVD_cpu, time_rSVD_gpu;

    // Perform standard SVD (time it)
    std::cout << "test 1" << std::endl;
    START_TIMER();
    std::cout << "test 2" << std::endl;
    SVD_cpu svd_cpu(&mat);
    std::cout << "test 3" << std::endl;
    STOP_RECORD_TIMER(time_SVD_cpu);
    std::cout << "test 4" << std::endl;

#if 0

    // Perform rSVD (time it)
    START_TIMER();
        // Get reconstruction error for rank r.
        // Print error.  Print time.
    STOP_RECORD_TIMER(time_rSVD_cpu);

    // Perform standard SVD on GPU (time it)
    START_TIMER();
        // Get reconstruction error for rank r.
        // Print error.  Print time.
    STOP_RECORD_TIMER(time_SVD_gpu);

    // Perform rSVD in GPU (time it)
    START_TIMER();
        // Get reconstruction error for rank r.
        // Print error.  Print time.
    STOP_RECORD_TIMER(time_rSVD_gpu);
#endif

    return 0;

} // end main