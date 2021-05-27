%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
README file for CS179 Final Project by Tanner Harms
CPU and GPU implementation of Randomized SVD

%--------------------------------------------------------------------------------------%
File Structure:

./bin  -> Folder for executable files.  Should only contain rSVD.exe.
./data -> Folder containing text files with matrices to perform SVD on.  
    test_matrix_01.txt -> Small 5-by-4 matrix to test functionality
    test_matrix_02.txt* -> Mid-sized 100-by-1000 flat matrix.
    test_matrix_03.txt* -> Larger 20000-by-3000 matrix resembling real data.
    eigenfaces.txt*     -> YaleB Face Data set organized as a matrix and normalized.
    CylinderFlow.txt*   -> Snapshots of flow around a cylinder.  Typical test set 
                            for data analytics in fluid mechanics.  
./obj  -> Folder containing all object files created during compilation and linking.
./src  -> Folder containing source files:
    rSVD.cpp      -> Primary file containing entry point.
    svds.cpp      -> Class structure for computing SVDs.  Each type of SVD is 
                        assigned a class name and is a child class of the SVD super 
                        class.  Types of SVD being performed are:
                        Standard SVD:
                        Randomized SVD:
                        Standard GPU SVD:
                        Randomized GPU SVD:
    svds.hpp      -> Class declarations for svds.cpp.
    utils.cpp     -> Various utilities for rSVD. Includes:
                        svd_params_s struct:  Stores rsvd parameters.
                        data class:  Stores input data.
                        functions for importing data:
    utils.hpp     -> Declarations for utils.cpp.
    helper_cuda.h -> Helper files for working with cuda libraries.  Same as from
                        assignments 5 and 6.  

* Not yet included in class structure

%--------------------------------------------------------------------------------------%
Usage:

To implement rSVD, the command line should be structured 
    ./bin/rSVD <input file> <rank> <power iterations> <oversaples> 

Example: ./bin/rSVD data/test_matrix_01.txt 2 1 1
    This uses the matrix contained in test_matrix_01.txt to perform all four types of 
    SVD.  it will use a rank-2 approximation of the data with 1 power iteration and 
    1 oversample.  

%--------------------------------------------------------------------------------------%
Background:



%--------------------------------------------------------------------------------------%
Results:












%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%