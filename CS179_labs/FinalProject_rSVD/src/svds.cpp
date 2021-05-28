#include <sys/time.h>
#include "utils.hpp"
#include "svds.hpp"

#include "helper_cuda.h"



/********************************************************************************/
// Timing utilities
struct timeval t1, t2;

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
    gettimeofday(&t1, 0);
    ComputeSVD();
    gettimeofday(&t2, 0);
    SVD_Compute_Time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
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
    gettimeofday(&t1, 0);
    ComputeSVD();
    gettimeofday(&t2, 0);
    SVD_Compute_Time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
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
    gettimeofday(&t1, 0);
    ComputeSVD();
    gettimeofday(&t2, 0);
    SVD_Compute_Time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
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
SVD_gpu::SVD_gpu(MatrixXd* InputDataPtr) 
    : SVD(InputDataPtr) {
    gettimeofday(&t1, 0);
    ComputeSVD();
    gettimeofday(&t2, 0);
    SVD_Compute_Time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
};

// Destructor
SVD_gpu::~SVD_gpu()
{
    std::cout << "SVD_gpu destroyed." << std::endl;
    return;
}

// Computing the SVD
void SVD_gpu::ComputeSVD()
/* 
  Note that I am following an example found on stack exchange for this implementation
  https://docs.nvidia.com/cuda/cusolver/index.html#gesvdj-example1
*/

{
    //-----------------------------------------------------------------------------//
    // Step 0:  Initialize variables to work with Cuda.

    // cusolver types.
    cusolverDnHandle_t cusolverH;   // cusolver handle
    cudaStream_t stream;            // stream handle
    gesvdjInfo_t gesvdj_params;     // Jacobi SVD parameters

    // Input data
    const int m = (*X_).rows(), n = (*X_).cols(), minmn = min(m,n);
    const int ldx = m;  // leading dimension X
    double* X = (*X_).data();    // m x n

    // Allocate space for U, S, and V
    const int ldu = m;  // leading dimension U
    double U[ldu*minmn];    // m x m
    const int ldv = n;  // leading dimension V
    double V[ldv*minmn];    // n x n   
    double S[minmn]; 

    // create device variables 
    double *d_X, *d_S, *d_U, *d_V;  // device SVD variables
    int* d_info;                    // error info
    int lwork, info;                // size of workspace and host copy of error info
    double* d_work;                 // device workspace for gesvdj

    // Configure gesvdj
    const double tol = 1.e-14;
    const int max_sweeps = 100;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eig vecs
    int econ = 1;    // economy size on

    //-----------------------------------------------------------------------------//
    // Step 1:  Create a cusolver handle and bind a stream.
    CUSOLVER_CALL( cusolverDnCreate(&cusolverH) );
    CUDA_CALL( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUSOLVER_CALL( cusolverDnSetStream(cusolverH, stream) );

    //-----------------------------------------------------------------------------//
    // Step 2:  Configuration of gesvdj
    CUSOLVER_CALL( cusolverDnCreateGesvdjInfo(&gesvdj_params) );
    CUSOLVER_CALL( cusolverDnXgesvdjSetMaxSweeps( gesvdj_params, max_sweeps) );
    CUSOLVER_CALL( cusolverDnXgesvdjSetTolerance( gesvdj_params, tol) );

    //-----------------------------------------------------------------------------//
    // Step 3:  Allocate device memory and copy data to it

    // Allocate the memory on the device
    CUDA_CALL( cudaMalloc ((void**) &d_X, sizeof(double) * ldx * n) );
    CUDA_CALL( cudaMalloc ((void**) &d_U, sizeof(double) * ldu * minmn) );
    CUDA_CALL( cudaMalloc ((void**) &d_V, sizeof(double) * ldv * minmn) );
    CUDA_CALL( cudaMalloc ((void**) &d_S, sizeof(double) * minmn) );
    CUDA_CALL( cudaMalloc ((void**) &d_info, sizeof(int)) );

    // Copy the initial array to the device.
    CUDA_CALL( cudaMemcpy(d_X, X, sizeof(double) * ldx * n, cudaMemcpyHostToDevice) );

    //-----------------------------------------------------------------------------//
    // Step 4:  Query the workspace of SVD
    
    // Get the buffer size
    CUSOLVER_CALL( cusolverDnDgesvdj_bufferSize(
        cusolverH,
        jobz,   // CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
        econ,   // 1 for economy size
        m,      // num rows
        n,      // num cols
        d_X,    // data on device m-by-n
        ldx,    // leading dimension of data
        d_S,    // min(m,n) sing vals on device
        d_U,    // m-by-min(m,n) for econ = 1
        ldu,    // leading dimension of U
        d_V,    // n-by-min(m,n) for 
        ldv,    // leading dimension of V
        &lwork,
        gesvdj_params) );

    // Allocate the workspace
    CUDA_CALL( cudaMalloc((void**) &d_work, sizeof(double) * lwork) );

    //-----------------------------------------------------------------------------//
    // Step 5:  Compute SVD

    // Do the SVD
    CUSOLVER_CALL( cusolverDnDgesvdj(
        cusolverH,
        jobz,  // CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors 
        econ,  // econ = 1 for economy size 
        m,     // nubmer of rows of A, 0 <= m 
        n,     // number of columns of A, 0 <= n  
        d_X,   // m-by-n 
        ldx,   // leading dimension of A 
        d_S,   // min(m,n). The singular values in descending order 
        d_U,   // m-by-min(m,n) if econ = 1 
        ldu,   // leading dimension of U, ldu >= max(1,m) 
        d_V,   // n-by-min(m,n) if econ = 1  
        ldv,   // leading dimension of V, ldv >= max(1,n) 
        d_work,
        lwork,
        d_info,
        gesvdj_params) );

    // Memcopy back to the host.
    CUDA_CALL( cudaMemcpy(U, d_U, sizeof(double) * ldu * minmn, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(V, d_V, sizeof(double) * ldv * minmn, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(S, d_S, sizeof(double) * minmn, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) );

    //-----------------------------------------------------------------------------//
    // Step 6:  Convert back to MatrixXd format for additional work in the class.
    U_ = Eigen::Map
        <Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic> >
        (&U[0], m, minmn);
    V_ = Eigen::Map
        <Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic> >
        (&V[0], n, minmn);
    S_ = Eigen::Map<Eigen::VectorXd>(&S[0], minmn);
    
    //-----------------------------------------------------------------------------//
    // Step 7:  Free variables
    if (d_X   ) CUDA_CALL( cudaFree(d_X) );
    if (d_U   ) CUDA_CALL( cudaFree(d_U) );
    if (d_V   ) CUDA_CALL( cudaFree(d_V) );
    if (d_S   ) CUDA_CALL( cudaFree(d_S) );
    if (d_info) CUDA_CALL( cudaFree(d_info) );
    if (d_work) CUDA_CALL( cudaFree(d_work) );

    if (cusolverH    ) CUSOLVER_CALL( cusolverDnDestroy(cusolverH) );
    if (stream       ) CUDA_CALL( cudaStreamDestroy(stream) );
    if (gesvdj_params) CUSOLVER_CALL( cusolverDnDestroyGesvdjInfo(gesvdj_params) );

    cudaDeviceReset();
    return;

}



/***********************************************************************************/
/* 
    Class for randomized SVD computed using a GPU.
*/

// Constructor
rSVD_gpu::rSVD_gpu(MatrixXd* InputDataPtr, svd_params_s params) 
    : SVD(InputDataPtr), rsvd_params(params) 
{
    gettimeofday(&t1, 0);
    ComputeSVD();
    gettimeofday(&t2, 0);
    SVD_Compute_Time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
};

// Destructor
rSVD_gpu::~rSVD_gpu()
{
    std::cout << "rSVD_gpu destroyed." << std::endl;
    return;
}

// Accessors
svd_params_s rSVD_gpu::getParams() { return rsvd_params; }

// Mutators
void rSVD_gpu::setParams(svd_params_s new_params) 
{
/* 
  Set new parameter then recompute the RSVD.  Also times the RSVD.
*/
    rsvd_params = new_params;
    gettimeofday(&t1, 0);
    ComputeSVD();
    gettimeofday(&t2, 0);
    SVD_Compute_Time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    return;    
}

#if 0
// Fill a matrix with random variables using curand
void rSVD_gpu::TestMatrix(double* A, int nr_rows_A, int nr_cols_A)
{
    size_t numelA = (size_t) nr_rows_A * nr_cols_A;
    double mn = 0.0, std = 1.0;
    
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    CURAND_CALL( curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT) );

    // Set the seed for the random number generator using the system clock
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock()) );

    // Fill the array with random numbers on the device
    CURAND_CALL( curandGenerateNormalDouble(prng, A, numelA, mn, std) );
    return;
}
#endif

// Computing the SVD
void rSVD_gpu::ComputeSVD()
/* 
  Implement the same algorithm as above, only using GPU accelerated commands from cuBLAS and
  cuSOLVER.
*/
{
    //-----------------------------------------------------------------------------//
    // Step 0:  Initialize variables to work with Cuda.

    // Handles:
    cusolverDnHandle_t cusolverH;           // cusolver handle
    cublasHandle_t cublasH;                 // cublas handle
    cudaStream_t stream;                    // stream handle
    
    // Initializing handles.
    CUSOLVER_CALL( cusolverDnCreate(&cusolverH) );
    CUBLAS_CALL( cublasCreate(&cublasH) );
    CUDA_CALL( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUSOLVER_CALL( cusolverDnSetStream(cusolverH, stream) );
    CUBLAS_CALL( cublasSetStream(cublasH, stream) );

    // cuBLAS identifiers
    cublasOperation_t transOn  = CUBLAS_OP_T;   // transpose
    cublasOperation_t transOff = CUBLAS_OP_N;   // no transpose

    // Input data
    const int m = (*X_).rows(), n = (*X_).cols(), minmn = min(m,n);
    const int ldx = m;  // leading dimension X
    double* X = (*X_).data();    // m x n

    // parsing the rsvd parameters
    int rank = rsvd_params.rank;
    int powiter = rsvd_params.powiter;
    int oversamples = rsvd_params.oversamples;

    // Check if matrix is too small for desired rank/oversamples
    if((rank + oversamples) > min((*X_).rows(), (*X_).cols())) {
      rank = min((*X_).rows(), (*X_).cols());
      oversamples = 0;
    }

    // Copy data to device
    double *d_X;
    CUDA_CALL( cudaMalloc ((void**) &d_X, sizeof(double) * ldx * n) );
    CUBLAS_CALL( cublasSetMatrix(ldx, n, sizeof(double), X, ldx, d_X, ldx) );


    //-----------------------------------------------------------------------------//
    // Step 1:  Set then randomized test matrix.

    // allocate the test matrix
    const int ldp = n;
    // double P[ldp * (rank + oversamples)];
    
    // Allocate P on the device.  
    double *d_P;
    CUDA_CALL( cudaMalloc ((void**) &d_P, sizeof(double) * ldp * (rank + oversamples)) );

    // Since curand isnt working, I am just doing a quick random generator here.
    size_t numelP = (size_t) ldp * (rank + oversamples);
    double mn = 0.0, std = 1.0;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mn, std);
    double P[numelP];
    for(int i = 0; i < (int) numelP; i++) 
    {
        P[i] = distribution(generator);
    }
    CUBLAS_CALL( cublasSetMatrix(ldp, rank + oversamples, sizeof(double), P, ldp, d_P, ldp) );
        
    // // Use another function to set the values to normally distributed random.
    // TestMatrix(d_P, ldp, rank + oversamples);


    //-----------------------------------------------------------------------------//
    // Step 2:  Implement Power Iterations.

    double one = 1.0, zero = 0.0;

    // Z = X*P
    const int ldz = m;
    // double Z[ldz * (rank + oversamples)];
    double *d_Z;
    CUDA_CALL( cudaMalloc ((void**) &d_Z, sizeof(double) * ldz * (rank + oversamples)) );
    CUDA_CALL( cublasDgemm(cublasH, transOff, transOff, 
        m, rank + oversamples, n, &one, 
        d_X, ldx, 
        d_P, ldp, 
        &zero, d_Z, ldz) );
    CUDA_CALL( cudaDeviceSynchronize() );

    // for k=1:q -> Z = X*(X'*Z) -> end
    int ldtemp = n;
    // double temp_array[ldtemp * (rank + oversamples)];
    double *d_temp_array;
    CUDA_CALL( cudaMalloc ((void**) &d_temp_array, sizeof(double) * ldtemp * (rank + oversamples)) );
    for (int i = 0; i < powiter; i++)  // powiter will never be very large, so just keep this 
                                   // in the cpp format.  
    {
        // first get temp = X'*Z
        CUDA_CALL( cublasDgemm(cublasH, transOn, transOff, 
            n, rank + oversamples, m, &one, 
            d_X, ldx, 
            d_Z, ldz, 
            &zero, d_temp_array, ldtemp) );
        CUDA_CALL( cudaDeviceSynchronize() );
        
        // Then Z = X * temp
        CUDA_CALL( cublasDgemm(cublasH, transOff, transOff, 
            m, rank + oversamples, n, &one, 
            d_X, ldx, 
            d_temp_array, ldtemp, 
            &zero, d_Z, ldz) );
        CUDA_CALL( cudaDeviceSynchronize() );
    }


    //-----------------------------------------------------------------------------//
    // Step 3:  Find Rank Revealing Q matrix [Q,R] = qr(Z,0). Z is m-by-(r+p)
    //  To do this, you need to use cuSOLVER functions geqrf and orgqr.  
    //  Following an example from nvidia docs
    //  https://docs.nvidia.com/cuda/cusolver/index.html#ormqr-example1

    // Setting up the important variables
    double *d_tau;      // scaling factor for qr decomposition.
    
    int *d_info;        // gpu info to assert successful qr.
    int h_info;         // host copy
    
    int lwork_geqrf;    // size of working array for geqrf.
    int lwork_orgqr;    // size of working array for orgqr.
    int lwork;          // size of working array.  max of the above 2.
    double *d_work;     // working space.  array of size lwork_geqrf

    CUDA_CALL( cudaMalloc((void**)&d_tau, sizeof(double) * (rank + oversamples)) );
    CUDA_CALL( cudaMalloc((void**)&d_info, sizeof(int)) );

    // query the workspace size for both functions
    CUSOLVER_CALL( cusolverDnDgeqrf_bufferSize(
        cusolverH,
        m,                  // first dim
        rank + oversamples, // second dim
        d_Z,
        ldz,
        &lwork_geqrf) );

    CUSOLVER_CALL( cusolverDnDorgqr_bufferSize(
        cusolverH,
        m,                  // first dimension
        rank + oversamples, // second dimension
        rank + oversamples, // elementary reflections defining the matrix Q
        d_Z,
        ldz,
        d_tau,
        &lwork_orgqr) );
    
    // max of the above two.
    lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;

    // Allocate workspace size.
    CUDA_CALL( cudaMalloc((void**)&d_work, sizeof(double) * lwork) );

    // Compute the qr decomposition using geqrf. 
    CUSOLVER_CALL( cusolverDnDgeqrf(
        cusolverH, 
        m,                  // first dimension
        rank + oversamples, // second dimension
        d_Z,                // array being used (overwritten)
        ldz,                // leading dimension
        d_tau,              // scaling factor
        d_work,             // workspace
        lwork,              // workspace size
        d_info) );          // info
    CUDA_CALL( cudaDeviceSynchronize() );
    
    // Check for successful qr factorization.
    CUDA_CALL( cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost) );
    assert(0 == h_info);

    // Get Q from the result of geqrf using ormqr
    CUSOLVER_CALL( cusolverDnDorgqr(
        cusolverH,
        m,                  // first dim
        rank + oversamples, // second dim
        rank + oversamples, // elementary reflections
        d_Z,                // array that will become Q
        ldz,                // leading dimension of Q
        d_tau,              // scaling factors
        d_work,             // working space    
        lwork,              // working space size
        d_info) );          // info
    CUDA_CALL( cudaDeviceSynchronize() );

    // Check for successful qr factorization.
    CUDA_CALL( cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost) );
    assert(0 == h_info);

    // Y = Q'*X
    const int ldy = rank + oversamples;
    // double Y[ldy * n];
    double *d_Y;
    CUDA_CALL( cudaMalloc ((void**) &d_Y, sizeof(double) * ldy * n) );
    CUDA_CALL( cublasDgemm(cublasH, transOn, transOff, 
        rank + oversamples, n, m, &one, 
        d_Z, ldz, 
        d_X, ldx, 
        &zero, d_Y, ldy) );
    CUDA_CALL( cudaDeviceSynchronize() );


    //-----------------------------------------------------------------------------//
    // Step 4:  Compute the SVD of the much smaller matrix Y

    gesvdjInfo_t gesvdj_params;             // Jacobi SVD parameters

    // Allocate space for Uy, S, and V
    const int lduy = rank + oversamples;
    const int ldv = n;
    const int minsz = min(rank + oversamples, n);
    // double Uy[lduy * minsz];        // Uy: (r+p)-by-min((r+p),n)
    double V[ldv * minsz];          // V: n-by-min((r+p),n)
    double S[minsz];                // S: min((r+p),n)
    double  *d_Uy, *d_V, *d_S;  
    CUDA_CALL( cudaMalloc ((void**) &d_Uy, sizeof(double) * lduy * minsz) );
    CUDA_CALL( cudaMalloc ((void**) &d_V, sizeof(double) * ldv * minsz) );
    CUDA_CALL( cudaMalloc ((void**) &d_S, sizeof(double) * minsz) );

    // Configure gesvdj ( jacobian SVD )
    const double tol = 1.e-14;
    const int max_sweeps = 100;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eig vecs
    int econ = 1;    // economy size on
    
    // Configure gesvdj
    CUSOLVER_CALL( cusolverDnCreateGesvdjInfo(&gesvdj_params) );
    CUSOLVER_CALL( cusolverDnXgesvdjSetMaxSweeps( gesvdj_params, max_sweeps) );
    CUSOLVER_CALL( cusolverDnXgesvdjSetTolerance( gesvdj_params, tol) );

    // Get the buffer size
    CUSOLVER_CALL( cusolverDnDgesvdj_bufferSize(
        cusolverH,
        jobz,   // CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
        econ,   // 1 for economy size
        rank + oversamples,      // num rows
        n,      // num cols
        d_Y,    // data on device (r+p)-by-n
        ldy,    // leading dimension of data
        d_S,    // min((r+p),n) sing vals on device
        d_Uy,   // (r+p)-by-min((r+p),n) for econ = 1
        lduy,   // leading dimension of U
        d_V,    // n-by-min((r+p),n) for 
        ldv,    // leading dimension of V
        &lwork,
        gesvdj_params) );

    // Allocate the workspace
    CUDA_CALL( cudaMalloc((void**) &d_work, sizeof(double) * lwork) );

    // Do the SVD: [Uy, S, V] = svd(Y,'econ');
    CUSOLVER_CALL( cusolverDnDgesvdj(
        cusolverH,
        jobz,   // CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
        econ,   // 1 for economy size
        rank + oversamples,      // num rows
        n,      // num cols
        d_Y,    // data on device (r+p)-by-n
        ldy,    // leading dimension of data
        d_S,    // min((r+p),n) sing vals on device
        d_Uy,   // (r+p)-by-min((r+p),n) for econ = 1
        lduy,   // leading dimension of U
        d_V,    // n-by-min((r+p),n) for 
        ldv,    // leading dimension of V 
        d_work,
        lwork,
        d_info,
        gesvdj_params) );
    
    // Check for successful qr factorization.
    CUDA_CALL( cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost) );
    assert(0 == h_info);

    // Synchronize to be safe
    CUDA_CALL( cudaDeviceSynchronize() );

    
    //-----------------------------------------------------------------------------//
    // Step 5:  Get U by projecting Uy onto Q.  U = Q*Uy

    // Allocate space for U
    const int ldu = m;
    double U[ldu * minsz];
    double *d_U;
    CUDA_CALL( cudaMalloc ((void**) &d_U, sizeof(double) * ldu * minsz) );

    // Q*Uy
    CUDA_CALL( cublasDgemm(cublasH, transOff, transOff, 
        m, minsz, rank + oversamples, &one, 
        d_Z, ldz, // Q
        d_Uy, lduy, 
        &zero, d_U, ldu) );

    CUDA_CALL( cudaMemcpy(U, d_U, sizeof(double) * ldu * minmn, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(V, d_V, sizeof(double) * ldv * minmn, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(S, d_S, sizeof(double) * minmn, cudaMemcpyDeviceToHost) );

    //-----------------------------------------------------------------------------//
    // Step 6:  Convert back to MatrixXd format for additional work in the class.
    U_ = Eigen::Map
        <Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic> >
        (&U[0], m, minsz).block(0, 0, (*X_).rows(), rank);
    V_ = Eigen::Map
        <Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic> >
        (&V[0], n, minsz).block(0, 0, (*X_).cols(), rank);
    S_ = Eigen::Map<Eigen::VectorXd>(&S[0], minsz).head(rank);

    
    //-----------------------------------------------------------------------------//
    // Step 7:  Free variables
    if (d_X         ) CUDA_CALL( cudaFree(d_X) );
    if (d_U         ) CUDA_CALL( cudaFree(d_U) );
    if (d_Uy        ) CUDA_CALL( cudaFree(d_Uy) );
    if (d_Z         ) CUDA_CALL( cudaFree(d_Z) );
    if (d_V         ) CUDA_CALL( cudaFree(d_V) );
    if (d_S         ) CUDA_CALL( cudaFree(d_S) );
    if (d_P         ) CUDA_CALL( cudaFree(d_P) );
    if (d_Y         ) CUDA_CALL( cudaFree(d_Y) );
    if (d_tau       ) CUDA_CALL( cudaFree(d_tau) );
    if (d_info      ) CUDA_CALL( cudaFree(d_info) );
    if (d_work      ) CUDA_CALL( cudaFree(d_work) );
    if (d_temp_array) CUDA_CALL( cudaFree(d_temp_array) );

    if (cusolverH    ) CUSOLVER_CALL( cusolverDnDestroy(cusolverH) );
    if (cublasH      ) CUBLAS_CALL( cublasDestroy(cublasH) );
    if (stream       ) CUDA_CALL( cudaStreamDestroy(stream) );
    if (gesvdj_params) CUSOLVER_CALL( cusolverDnDestroyGesvdjInfo(gesvdj_params) );

    cudaDeviceReset();
    return;

}

#if 0 
/* 
  cuSOLVER from version 11.3 (perhaps a little earlier) contains an implementation of rSVD
  which I have implemented here.  Since the version of cuda on the machine is 9.1, I cannot
  run it for this application.  However, it should work otherwise.
*/ 

// Computing the SVD
void rSVD_gpu::ComputeSVD()
/* 
  Note that I am following an example found on stack exchange for this implementation
  https://docs.nvidia.com/cuda/cusolver/index.html#gesvdj-example1
*/
{
    //-----------------------------------------------------------------------------//
    // Step 0:  Initialize variables to work with Cuda.

    // cusolver types.
    cusolverDnHandle_t cusolverH;           // cusolver handle
    cudaStream_t stream;                    // stream handle
    cusolverDnParams_t params_gesvdr;       // Randomized SVD parameters

    // Input data
    const int64_t m = (*X_).rows(), n = (*X_).cols(), minmn = min(m,n);
    const int64_t ldx = m;  // leading dimension X
    const int64_t rank = (int64_t) rsvd_params.rank;
    double* X = (*X_).data();    // m x n

    // Allocate space for U, S, and V
    const int64_t ldu = m;  // leading dimension U
    double U[ldu*rank];    // m x m
    const int64_t ldv = n;  // leading dimension V
    double V[ldv*rank];    // n x n   
    double S[rank]; 

    // create device variables 
    void *d_X, *d_S, *d_U, *d_V;  // device SVD variables
    int* d_info;                    // error info
    int lwork, h_info;                // size of workspace and host copy of error info
    double* d_work_gesvdr;                 // device workspace for gesvdr
    double* h_work_gesvdr;                 // device workspace for gesvdr
    size_t workspaceInBytesOnDevice_gesvdr;
    size_t workspaceInBytesOnHost_gesvdr;

    // Configure gesvdr
    signed char jobu = 'S';     // compute left singular vectors
    signed char jobv = 'S';     // compute right singular vectors
    const int64_t iters = (int64_t) rsvd_params.powiter;
    const int64_t p = (int64_t) rsvd_params.oversamples;

    //-----------------------------------------------------------------------------//
    // Step 1:  Create a cusolver handle and bind a stream.
    CUSOLVER_CALL( cusolverDnCreate(&cusolverH) );
    CUDA_CALL( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
    CUSOLVER_CALL( cusolverDnSetStream(cusolverH, stream) );

    //-----------------------------------------------------------------------------//
    // Step 2:  Configuration of gesvdr
    CUSOLVER_CALL( cusolverDnCreateParams(&params_gesvdr) );

    //-----------------------------------------------------------------------------//
    // Step 3:  Allocate device memory and copy data to it

    // Allocate the memory on the device
    CUDA_CALL( cudaMalloc ((void**) &d_X, sizeof(double) * ldx * n) );
    CUDA_CALL( cudaMalloc ((void**) &d_U, sizeof(double) * ldu * rank) );
    CUDA_CALL( cudaMalloc ((void**) &d_V, sizeof(double) * ldv * n) );
    CUDA_CALL( cudaMalloc ((void**) &d_S, sizeof(double) * rank) );
    CUDA_CALL( cudaMalloc ((void**) &d_info, sizeof(int)) );

    // Copy the initial array to the device.
    CUDA_CALL( cudaMemcpy(d_X, X, sizeof(double) * ldx * n, cudaMemcpyHostToDevice) );

    //-----------------------------------------------------------------------------//
    // Step 4:  Query the workspace of SVD
    
    // Get the buffer size
    CUSOLVER_CALL( cusolverDnXgesvdr_bufferSize(
        cusolverH,      // cusolver Handle
        params_gesvdr,  // cusolver Parameters structure
        jobu,           // 'S' returns leakding rank-r left sing vecs
        jobv,           // 'S' returns leakding rank-r right sing vecs
        m,              // rows of data
        n,              // columns
        rank,           // rank of the decomposition
        p,              // oversampling parameter
        iters,          // power iterations
        CUDA_R_64F,     // data type of data
        d_X,            // Array X: ldx-by-n
        ldx,            // leading dimension of X
        CUDA_R_64F,     // data type of singular values
        d_S,            // Array S: rank
        CUDA_R_64F,     // data type of left singular vectors
        d_U,            // Array U: ldu-by-rank
        ldu,            // leading dimension of U
        CUDA_R_64F,     // data type of right singular vectors
        d_V,            // Array V: ldv(rank)-by-n
        ldv,            // leading dimension of V
        CUDA_R_64F,     // data type of computation
        &workspaceInBytesOnDevice_gesvdr,       // device buffer
        &workspaceInBytesOnHost_gesvdr ) );    // host buffer

    // Allocate the workspace
    h_work_gesvdr = (double*) malloc( workspaceInBytesOnHost_gesvdr );
    assert( h_work_gesvdr != NULL );

    CUDA_CALL( cudaMalloc((void**) &d_work_gesvdr, workspaceInBytesOnDevice_gesvdr) );
    CUDA_CALL( cudaDeviceSynchronize() );

    //-----------------------------------------------------------------------------//
    // Step 5:  Compute SVD

    // Do the SVD
    CUSOLVER_CALL( cusolverDnXgesvdr_bufferSize(
        cusolverH,      // cusolver Handle
        params_gesvdr,  // cusolver Parameters structure
        jobu,           // 'S' returns leakding rank-r left sing vecs
        jobv,           // 'S' returns leakding rank-r right sing vecs
        m,              // rows of data
        n,              // columns
        rank,           // rank of the decomposition
        p,              // oversampling parameter
        iters,          // power iterations
        CUDA_R_64F,     // data type of data (real double precision)
        d_X,            // Array X: ldx-by-n
        ldx,            // leading dimension of X
        CUDA_R_64F,     // data type of singular values
        d_S,            // Array S: rank
        CUDA_R_64F,     // data type of left singular vectors
        d_U,            // Array U: ldu-by-rank
        ldu,            // leading dimension of U
        CUDA_R_64F,     // data type of right singular vectors
        d_V,            // Array V: ldv-by-rank
        ldv,            // leading dimension of V
        CUDA_R_64F,     // data type of computation
        d_work_gesvdr,  // device workspace
        workspaceInBytesOnDevice_gesvdr,       // device buffer
        h_work_gesvdr,  // host workspace
        workspaceInBytesOnHost_gesvdr ) );     // host buffer );

    // Memcopy back to the host.
    CUDA_CALL( cudaMemcpy(U, d_U, sizeof(double) * ldu * rank, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(V, d_V, sizeof(double) * ldv * rank, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(S, d_S, sizeof(double) * rank, cudaMemcpyDeviceToHost) );
    CUDA_CALL( cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost) );

    //-----------------------------------------------------------------------------//
    // Step 6:  Convert back to MatrixXd format for additional work in the class.
    U_ = Eigen::Map
        <Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic> >
        (&U[0], m, minmn);
    V_ = Eigen::Map
        <Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic> >
        (&V[0], n, minmn);
    S_ = Eigen::Map<Eigen::VectorXd>(&S[0], minmn);
    
    //-----------------------------------------------------------------------------//
    // Step 7:  Free variables
    if (d_X   ) CUDA_CALL( cudaFree(d_X) );
    if (d_U   ) CUDA_CALL( cudaFree(d_U) );
    if (d_V   ) CUDA_CALL( cudaFree(d_V) );
    if (d_S   ) CUDA_CALL( cudaFree(d_S) );
    if (d_info) CUDA_CALL( cudaFree(d_info) );
    if (d_work) CUDA_CALL( cudaFree(d_work_gesvdr) );

    if ( h_work_gesvdr ) free( h_work_gesvdr );

    if (cusolverH    ) CUSOLVER_CALL( cusolverDnDestroy(cusolverH) );
    if (stream       ) CUDA_CALL( cudaStreamDestroy(stream) );
    if (gesvdj_params) CUSOLVER_CALL( cusolverDnDestroyParams(params_gesvdr) );

    CUDA_CALL( cudaDeviceReset() );
    return;

}
#endif
