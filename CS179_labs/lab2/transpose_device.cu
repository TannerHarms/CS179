#include <cassert>
#include <cuda_runtime.h>
#include "transpose_device.cuh"

/*
 * TODO for all kernels (including naive):
 * Leave a comment above all non-coalesced memory accesses and bank conflicts.
 * Make it clear if the suboptimal access is a read or write. If an access is
 * non-coalesced, specify how many cache lines it touches, and if an access
 * causes bank conflicts, say if its a 2-way bank conflict, 4-way bank
 * conflict, etc.
 *
 * Comment all of your kernels.
 */


/*
 * Each block of the naive transpose handles a 64x64 block of the input matrix,
 * with each thread of the block handling a 1x4 section and each warp handling
 * a 32x4 section.
 *
 * If we split the 64x64 matrix into 32 blocks of shape (32, 4), then we have
 * a block matrix of shape (2 blocks, 16 blocks).
 * Warp 0 handles block (0, 0), warp 1 handles (1, 0), warp 2 handles (0, 1),
 * warp n handles (n % 2, n / 2).
 *
 * This kernel is launched with block shape (64, 16) and grid shape
 * (n / 64, n / 64) where n is the size of the square matrix.
 *
 * You may notice that we suggested in lecture that threads should be able to
 * handle an arbitrary number of elements and that this kernel handles exactly
 * 4 elements per thread. This is OK here because to overwhelm this kernel
 * it would take a 4194304 x 4194304    matrix, which would take ~17.6TB of
 * memory (well beyond what I expect GPUs to have in the next few years).
 */
__global__
void naiveTransposeKernel(const float *input, float *output, int n) {
    // TODO: do not modify code, just comment on suboptimal accesses

    // This is slow because it requires that the GPU communicate more frequently with
    // the global memory.  Using the shared memory in the next section allows for 
    // speed increases by putting the data in a faster location.  As the output is 
    // computed here, it must pull the input directly from global and put it out 
    // directly to global.  Fixing this, as we do in the next segment, affords a 
    // large speed boost.

    const int i = threadIdx.x + 64 * blockIdx.x;
    int j = 4 * threadIdx.y + 64 * blockIdx.y;
    const int end_j = j + 4;

    // Though it is barely a consideration, loop unrolling may add a tiny speed
    // boost...
    for (; j < end_j; j++)
        output[j + n * i] = input[i + n * j];
}

__global__
void shmemTransposeKernel(const float *input, float *output, int n) {
    // TODO: Modify transpose kernel to use shared memory. All global memory
    // reads and writes should be coalesced. Minimize the number of shared
    // memory bank conflicts (0 bank conflicts should be possible using
    // padding). Again, comment on all sub-optimal accesses.

    // Store the 64x64 sub-matrix in shared memory for easy access with padding.
    __shared__ float data[64 * 65];  // This could lead to bank conflicts, if 
                                     // padding is not allowed in the range.  

    // Indexing 64 by 64 blocks.
    // Assign each thread a constant index according to the row.
    const int i = threadIdx.x + 64 * blockIdx.x;  // Get the row index of the input
    const int ii = threadIdx.x;  // Get the row index of the sub-matrix
    const int iii = threadIdx.x + 64 * blockIdx.y; // transpose the blocks
    
    // Allow j to be a variable that iterates rows in the input and columns in the 
    // output.  Do four column indices at a time.  
    int j = 4 * threadIdx.y + 64 * blockIdx.y;  // Get the column index of the input
    int jj = 4 * threadIdx.y;  // Get the column index of the sub-matrix
    int jjj = 4 * threadIdx.y + 64 * blockIdx.x; // transpose the blocks
    const int end_j = j + 4;

    for (; j < end_j; j++){
        data[ii + 65 * jj] = input[i + n * j];
        jj++;
    }
    
    // Reset j and jj
    j = 4 * threadIdx.y + 64 * blockIdx.y;  
    jj = 4 * threadIdx.y;  

    // synchronize the threads here
    __syncthreads();    // We have to wait for threads to sync, which slows us down a 
                        // little.  

    // Perform the transpose and save the data to the output
    for (; j < end_j; j++) {
        output[iii + n * jjj] = data[jj + 65 * ii]; // Here data is accessed again,
                                                    // which leads to sub-optimality.
        jj++;
        jjj++;
    }

}

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.
    // Store the 64x64 sub-matrix in shared memory for easy access with padding.
    __shared__ float data[64 * 65];

    // Indexing 64 by 64 blocks.
    // Assign each thread a constant index according to the row.
    const int i = threadIdx.x + 64 * blockIdx.x;  // Get the row index of the input
    const int ii = threadIdx.x;  // Get the row index of the sub-matrix
    const int iii = threadIdx.x + 64 * blockIdx.y; // transpose the blocks
    
    // Allow j to be a variable that iterates rows in the input and columns in the 
    // output.  Do four column indices at a time.  
    int j = 4 * threadIdx.y + 64 * blockIdx.y;  // Get the column index of the input
    int jj = 4 * threadIdx.y;  // Get the column index of the sub-matrix
    int jjj = 4 * threadIdx.y + 64 * blockIdx.x; // transpose the blocks

    // Unroll the loops
    data[ii + 65 * (jj)] = input[i + n * (j)];
    data[ii + 65 * (jj + 1)] = input[i + n * (j + 1)];
    data[ii + 65 * (jj + 2)] = input[i + n * (j + 2)];
    data[ii + 65 * (jj + 3)] = input[i + n * (j + 3)];
    
    // reset j and jj
    j = 4 * threadIdx.y + 64 * blockIdx.y;  
    jj = 4 * threadIdx.y;  

    // synchronize the threads here
    __syncthreads();

    // Perform the transpose and save the data to the output
    output[iii + n * (jjj)] = data[(jj) + 65 * ii];
    output[iii + n * (jjj + 1)] = data[(jj + 1) + 65 * ii];
    output[iii + n * (jjj + 2)] = data[(jj + 2) + 65 * ii];
    output[iii + n * (jjj + 3)] = data[(jj + 3) + 65 * ii];    

}

void cudaTranspose(
    const float *d_input,
    float *d_output,
    int n,
    TransposeImplementation type)
{
    if (type == NAIVE) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        naiveTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == SHMEM) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        shmemTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    else if (type == OPTIMAL) {
        dim3 blockSize(64, 16);
        dim3 gridSize(n / 64, n / 64);
        optimalTransposeKernel<<<gridSize, blockSize>>>(d_input, d_output, n);
    }
    // Unknown type
    else
        assert(false);
}
