
#include "utils.hpp"


/********************************************************************************/
// Argument utilities

// Parsing input arguments
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
// Data Class utilities

// Class Constructor
data::data(int M, int N, double* data) : m(), n(), arr() {
    m = M;
    n = N;
    arr = data;
    matXd = Eigen::Map
        <Eigen::Matrix <double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >
        (&data[0], m, n);
}

// Class Destructor
data::~data() {
    delete[] arr;
}

// Accessors
int data::getM() { return m; }
int data::getN() { return n; }
double* data::getData_ptr() { return arr; }
MatrixXd data::getData_mXd() { return matXd; }

// Mutators
void data::setM(int numRows) { 
    m = numRows; 
    return;
}
void data::setN(int numCols) { 
    n = numCols; 
    return;
}
void data::setData_Ptr(double *data) { 
    arr = data;
    return;
}
void data::setData_mXd(MatrixXd data) { 
    matXd = data;
    return;
}

// function to print the array
void data::printData(void) 
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

/********************************************************************************/
// Handling imported data.
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
// Matrix printing utility for testing GPU SVD results
void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %20.16E\n", name, row+1, col+1, Areg);
        }
    }
}