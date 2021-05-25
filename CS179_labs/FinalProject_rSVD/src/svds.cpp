

#include "utils.hpp"
#include "svds.hpp"


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


/***********************************************************************************/
/*
    SVD Superclass
*/

// Constructor
SVD::SVD(MatrixXd* Xptr) 
    : X_(Xptr), U_(), V_(), S_(), R_(), SVD_Compute_Time() {}

// Destructor
SVD::~SVD()
{
    std::cout << "SVD Destroyed." << std::endl;
    return;
}

// Accessors
MatrixXd* SVD::inputMatrixPtr() { return X_; }
MatrixXd SVD::matrixU() { return U_; }
MatrixXd SVD::matrixV() { return V_; }
VectorXd SVD::singularValues() { return S_; }
MatrixXd SVD::reconstruction() { return R_; }
double SVD::spectralNorm() { return Spec_Norm; }
double SVD::frobeniusNorm() { return Frob_Norm; }
float SVD::computeTime() { return SVD_Compute_Time; }

// Evaluation Functions
// Reconstruct a rank-r matrix using the SVD results.
MatrixXd SVD::Reconstruct(int rank)
{
    MatrixXd temp = MatrixXd(0,0);
    return temp;
}

// Get the Spectral Norm of the approximation error
double SVD::SpectralNorm(MatrixXd mat)
{
    return 0;
}

// Get the Frobenius Norm of the approximation error
double SVD::FrobeniusNorm(MatrixXd mat)
{
    return 0;
}

// Set the evaluation properties in the SVD Class
void SVD::Evaluate()
{
    return;
}



/***********************************************************************************/
/* 
    Class for standard SVD computed using a CPU.
*/

// Constructor
SVD_cpu::SVD_cpu(MatrixXd* InputDataPtr) : SVD(InputDataPtr) {
    START_TIMER();
    ComputeSVD();
    STOP_RECORD_TIMER(SVD_Compute_Time);
};

// Destructor
SVD_cpu::~SVD_cpu()
{
    std::cout << "SVD_cpu destroyed." << std::endl;
    return;
}

// Computing the SVD
void SVD_cpu::ComputeSVD()
{
    //MatrixXd X = *X_;
    Eigen::JacobiSVD<MatrixXd> svd(*X_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U_ = svd.matrixU();
    V_ = svd.matrixV();
    S_ = svd.singularValues();
    return;
}



/***********************************************************************************/
/* 
    Class for randomized SVD computed using a CPU.
*/

// Constructor
rSVD_cpu::rSVD_cpu(MatrixXd* InputDataPtr, svd_params_s params) 
    : SVD(InputDataPtr), rsvd_params(params) 
{
    START_TIMER();
    ComputeSVD();
    STOP_RECORD_TIMER(SVD_Compute_Time);
};

// Destructor
rSVD_cpu::~rSVD_cpu()
{
    std::cout << "rSVD_cpu destroyed." << std::endl;
    return;
}

// Accessors
svd_params_s rSVD_cpu::getParams() { return rsvd_params; }

// Mutators
void rSVD_cpu::setParams(svd_params_s new_params) 
{
/* 
  Set new parameter then recompute the RSVD.  Also times the RSVD.
*/
    rsvd_params = new_params;
    START_TIMER();
    ComputeSVD();
    STOP_RECORD_TIMER(SVD_Compute_Time);
    return;    
}

// Range Finder ( Following the work of Kazuotani14 on Github)
// https://github.com/kazuotani14/RandomizedSvd/blob/master/randomized_svd.h
MatrixXd rSVD_cpu::RangeFinder(int size, int powiter)
{
    // Get rows and columns:
    int nr = (*X_).rows(), nc = (*X_).cols();

    // Make the Test Matrix
    MatrixXd L(nr, size);
    Eigen::FullPivLU<MatrixXd> lu1(nr, size); // rank revealing LU decomposition.  Tall and skinny
                                              // stable and well tested with large arrays
    MatrixXd Q = MatrixXd::Random(nc, size);  // Random test matrix.  Short and fat
    Eigen::FullPivLU<MatrixXd> lu2(nc, nr);

    // Do normalized power iterations.
    for (int i = 0; i < powiter; i++)
    {
        lu1.compute((*X_) * Q);
        L.setIdentity();
        L.block(0, 0, nr, size).triangularView<Eigen::StrictlyLower>() = lu1.matrixLU();

        lu2.compute((*X_).transpose() * L);
        Q.setIdentity();
        Q.block(0, 0, nc, size).triangularView<Eigen::StrictlyLower>() = lu2.matrixLU();
    }

    // Get a skinny Q matrix output.  
    Eigen::ColPivHouseholderQR<MatrixXd> qr((*X_) * Q);
    return qr.householderQ() * MatrixXd::Identity(nr, size);
}

// Computing the SVD ( Following the work of Kazuotani14 on Github)
// https://github.com/kazuotani14/RandomizedSvd/blob/master/randomized_svd.h
void rSVD_cpu::ComputeSVD() 
{
    using namespace std::chrono;

    // shorthand
    int rank = rsvd_params.rank;
    int powiter = rsvd_params.powiter;
    int oversamples = rsvd_params.oversamples;
    //MatrixXd X = (*X_);

    // Check if matrix is too small for desired rank/oversamples
    if((rank + oversamples) > min((*X_).rows(), (*X_).cols())) {
      rank = min((*X_).rows(), (*X_).cols());
      oversamples = 0;
    }

    MatrixXd Q = RangeFinder(rank + oversamples, powiter);
    MatrixXd B = Q.transpose() * (*X_);

    // Compute the SVD on the thin matrix (much cheaper than SVD on original)
    Eigen::JacobiSVD<MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Get only the leading rank-r Singular vectors and Singular Values.
    U_ = (Q * svd.matrixU()).block(0, 0, (*X_).rows(), rank);
    V_ = svd.matrixV().block(0, 0, (*X_).cols(), rank);
    S_ = svd.singularValues().head(rank);

}



/***********************************************************************************/
/* 
    Class for standard SVD computed using a GPU.
*/

// Constructor

// Destructor

// Computing the SVD



/***********************************************************************************/
/* 
    Class for randomized SVD computed using a GPU.
*/

// Constructor

// Destructor

// Accessors

// Mutators

// Computing the SVD