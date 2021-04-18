/* CUDA blur
 * Kevin Yuh, 2014 */

#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>
#include <cufft.h>

#include "fft_convolve.cuh"


/* 
Atomic-max function. You may find it useful for normalization.

We haven't really talked about this yet, but __device__ functions not
only are run on the GPU, but are called from within a kernel.

Source: 
http://stackoverflow.com/questions/17399119/
cant-we-use-atomic-operations-for-floating-point-variables-in-cuda
*/
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}



__global__
void
cudaProdScaleKernel(const cufftComplex *raw_data, const cufftComplex *impulse_v, 
    cufftComplex *out_data, int padded_length) {
    
    // It makes sense to just use one dimension here.  Also, problem oriented this way
    uint thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Need to scale by record length due to FFT implementation in cuFFT.  
    float inverse_padded_length = 1.0 / padded_length;

    while (thread_index < padded_length) {
        // Do complex multiplication
        // Store temporary variables

        float a = raw_data[thread_index].x;
        float b = raw_data[thread_index].y;
        float c = impulse_v[thread_index].x;
        float d = impulse_v[thread_index].y;

        cufftComplex output;

        // Update the output
        output.x = ((a * c) - (b * d)) * inverse_padded_length;
        output.y = ((a * d) + (b * c)) * inverse_padded_length;

        out_data[thread_index] = output;

        // Update the grid stride index
        thread_index += blockDim.x * gridDim.x;
    }

    /* TODO: Implement the point-wise multiplication and scaling for the
    FFT'd input and impulse response. 

    Recall that these are complex numbers, so you'll need to use the
    appropriate rule for multiplying them. 

    Also remember to scale by the padded length of the signal
    (see the notes for Question 1).

    As in Assignment 1 and Week 1, remember to make your implementation
    resilient to varying numbers of threads.

    */
}

__global__
void
cudaMaximumKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the maximum-finding.

    There are many ways to do this reduction, and some methods
    have much better performance than others. 

    For this section: Please explain your approach to the reduction,
    including why you chose the optimizations you did
    (especially as they relate to GPU hardware).

    You'll likely find the above atomicMax function helpful.
    (CUDA's atomicMax function doesn't work for floating-point values.)
    It's based on two principles:
        1) From Week 2, any atomic function can be implemented using
        atomic compare-and-swap.
        2) One can "represent" floating-point values as integers in
        a way that preserves comparison, if the sign of the two
        values is the same. (see http://stackoverflow.com/questions/
        29596797/can-the-return-value-of-float-as-int-be-used-to-
        compare-float-in-cuda)

    */

    // Here I will just implement the vanilla Interleaved Addressing reduction 
    // as a baseline.  I am referencing M. Harris' slides
    // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
    // If I have time, I will try to write an optimized version of this.  

    // Set up shared data
    extern __shared__ float sdata[];

    // Load one element from global to shared memory with each thread
    unsigned int thread_index = threadIdx.x; // thread index in current block
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    sdata[thread_index] = abs(out_data[i].x);   // load data and compute absolute value
    __syncthreads(); 

    // Allow for records of varying length
    while (thread_index < padded_length) {

        // Do the reduction in shared memory per block
        for(unsigned int s = 1; s < blockDim.x; s *= 2) {
            if (thread_index % (2 * s) == 0) {
                // Use the atomicMax function defined above
                sdata[thread_index] = atomicMax(&sdata[thread_index], sdata[thread_index + s]);
            }
            __syncthreads();
        }

        // write the result for this to the global memory
        // If the first index of the block
        if (thread_index == 0) {
            // If the previous max absolute value is less than that of this thread
            if (*max_abs_val < sdata[0]) {
                *max_abs_val = sdata[0]; // store the value of the new max abs val
            }
        }

        // update with grid stride
        thread_index += blockDim.x * gridDim.x;
    }

}

__global__
void
cudaDivideKernel(cufftComplex *out_data, float *max_abs_val,
    int padded_length) {

    /* TODO 2: Implement the division kernel. Divide all
    data by the value pointed to by max_abs_val. 

    This kernel should be quite short.
    */

    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    float scaling = 1.0 / *max_abs_val; 

    while (thread_index < padded_length) { 
        // scale each value.  Don't need y.
        out_data[thread_index].x *= scaling;

        // Update the thread index with a grid stride
        thread_index += gridDim.x * blockDim.x;
    }
}


void cudaCallProdScaleKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        const cufftComplex *raw_data,
        const cufftComplex *impulse_v,
        cufftComplex *out_data,
        const unsigned int padded_length) {
        
    /* TODO: Call the element-wise product and scaling kernel. */

    cudaProdScaleKernel<<<blocks, threadsPerBlock>>>(raw_data, impulse_v, out_data, padded_length);
}

void cudaCallMaximumKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        

    /* TODO 2: Call the max-finding kernel. */

    cudaMaximumKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}


void cudaCallDivideKernel(const unsigned int blocks,
        const unsigned int threadsPerBlock,
        cufftComplex *out_data,
        float *max_abs_val,
        const unsigned int padded_length) {
        
    /* TODO 2: Call the division kernel. */

    cudaDivideKernel<<<blocks, threadsPerBlock>>>(out_data, max_abs_val, padded_length);
}
