

#include "utils.hpp"
#include "svds.hpp"


/***********************************************************************************/
/*
    SVD Superclass
*/

// Constructor
SVD::SVD(MatrixXd* Xptr) 
    : X_(Xptr), U_(), V_(), S_(), R_() {}

// Destructor
SVD::~SVD()
{
    std::cout << "Destroyed SVD" << std::endl;
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
    std::cout << "test 2.1" << std::endl;
    std::cout << *X_ << std::endl;
    ComputeSVD();
};

// Destructor
SVD_cpu::~SVD_cpu()
{
    std::cout << "SVD_cpu destroyed" << std::endl;
    return;
}

// Computing the SVD
void SVD_cpu::ComputeSVD()
{
    MatrixXd X = *X_;
    std::cout << "test 2.2" << std::endl;
    Eigen::JacobiSVD<MatrixXd> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    std::cout << "test 2.3" << std::endl;
    U_ = svd.matrixU();
    V_ = svd.matrixV();
    S_ = svd.singularValues();
    std::cout << "test 2.4" << std::endl;
    return;
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
    Class for randomized SVD computed using a CPU.
*/

// Constructor

// Destructor

// Accessors

// Mutators

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