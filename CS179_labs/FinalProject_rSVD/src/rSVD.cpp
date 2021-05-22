/**
 * Performs randomized SVD on an m x n matrix provided by the user.
 * @author Tanner Harms
 * @date May 17th, 2021
 */

#include "helper_cuda.h"
//#include "svds.hpp"
#include "utils.hpp"


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
    inputData.printData();

#if 0
    // set up floats to store the time
    float time_SVD_cpu, time_SVD_gpu;
    float time_rSVD_cpu, time_rSVD_gpu;

    // Perform standard SVD (time it)
    START_TIMER();
        // Get reconstruction error for rank r.
        // Print error.  Print time.
    STOP_RECORD_TIMER(time_SVD_cpu);

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