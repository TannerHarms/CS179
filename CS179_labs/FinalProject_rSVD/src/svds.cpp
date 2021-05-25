

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
    : X_(Xptr), U_(), V_(), S_(), R_(), 
        Spec_Norm(), Frob_Norm(), SVD_Compute_Time() {}

// Destructor
SVD::~SVD()
{
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
void SVD::Reconstruct(int rank)
{
    // Ensure that the size of the data is larger than the desired rank
    if (rank > S_.size()) {
        std::cerr << "SVD.Reconstruct: Rank is too large. " << endl;
        exit(EXIT_FAILURE);
    }
    
    // Truncated arrays
    MatrixXd U_r = U_.leftCols(rank);
    MatrixXd V_r = V_.leftCols(rank);
    VectorXd S_r = S_.head(rank);

    // Create the reconstruction: Approximation = U_r * S_R * V_r^T
    R_ = U_r * S_r.asDiagonal() * V_r.transpose();
    return;
}

// Get the Spectral Norm of the approximation error
double SVD::SpectralNorm(MatrixXd mat, int n_iter)
/* 
    The spectral norm is the maximum eigenvalue of the matrix X^H * X, where H
    indicates the hermitian transpose.  In fact, the non-zero eigenvalues of X^H * X 
    and X * X^H are the same, so we can make a statement to efficiently compute the
    spectral norm here.  
*/
{
    // Efficiently compute the matrix transpose
    if ( mat.rows() <= mat.cols() ) { // short and fat matrix
        mat = mat * mat.transpose();
    } else if ( mat.cols() < mat.rows() ) { // tall and skinny matrix
        mat = mat.transpose() * mat;
    }

    // Create a vector for identifying the largest eigenvalue of the input array
    VectorXd y = VectorXd::Random(mat.rows());

    // We use a power iteration method to compute the maximum eigenvalue.  Since we 
    // are using a large matrix, we dont want to compute all of the eigenvalues, only
    // the largest one.  
    for ( int i = 0; i < n_iter; i++ )
    {
        y = mat * y;
        y.normalize();
    }
    VectorXd Ay = mat * y;

    // get the max eigenvalue according to max{||Ay||_2/||y||_2}
    double maxeig = abs( Ay.dot(y) ) / ( y.dot(y) );

    if(maxeig==0) return 0;
    return sqrt(maxeig);
}

// Get the Frobenius Norm of an input matrix.
double SVD::FrobeniusNorm(MatrixXd mat)
/* 
    The Frobenius norm is the square root of the sum of the absolute value
    of every element in an array.  It can also be computed as the trace of 
    X X^T.  When the array is real, it can be called the Hilbert-Schmidt norm
    as well.  
*/
{
    double F_norm = mat.norm();
    return F_norm;
}

// Set the evaluation properties in the SVD Class
void SVD::Evaluate(int rank)
{
    // If the reconstruction has not yet been done, do it. 
    if (R_.size() == 0) {
        Reconstruct(rank);
    }

    // Get the error between the orignal data and the reconstruction.
    MatrixXd X_err = (*X_) - R_;

    // Compute Spectral Norm and store to class
    static int n_iter = 30;
    Spec_Norm = SpectralNorm(X_err, n_iter);

    // Compute Frobenius Norm and store to class
    Frob_Norm = FrobeniusNorm(X_err);
    
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