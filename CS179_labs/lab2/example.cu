/**

* For a matrix of size 32 x 32, computes (a_ij)^(pow) for each element a_ij

* and stores in res_ij.

*

* Shared memory is necessary here because we are reading and writing

* to memory many times...

*

* Note that __syncthreads is not needed here because each row in shared

* memory is exclusively read and written to by a single warp.

*/

__global__ void pow_rows(const float *a, uint pow, float *res) {

    // store entire matrix in shared memory for fast reads.

    __shared__ float s_a[32 * 32];

    // store result in shared memory for fast writes.

    __shared__ float s_res[32 * 32];

    // assign each thread an index so that threads in the same warp process

    // elements in the same row.

    const uint row_i = threadIdx.x + 32 * threadIdx.y;

    // copy matrix from global memory to shared memory in a coalesced fashion.

    s_a[row_i] = a[row_i];

    // intialize result as a matrix where each element is 1.0.

    s_res[row_i] = 1.0;

    // a single block computes the power of the entire matrix.

    // each warp in the block computes the power of a single row.

    // each thread in the warp computes the power of a single element.

    while (pow > 0) {

        s_res[row_i] *= s_a[row_i];

        pow -= 1;

    }

    // copy result from shared memory to global memory in a coalesced fashion.

    res[row_i] = s_res[row_i];

};

/**

* For a matrix of size 32 x 32, computes (a_ij)^(pow) for each element a_ij

* and stores in res_ij.

*

* After reading the matrix a into local memory row by row, we

* compute the power of each element on a column by column basis

* in order to cause a bank conflict.

*

* Note that __syncthreads is necessary here because the same shared

* memory is accessed by multiple warps.

*/

__global__ void pow_cols(const float *a, uint pow, float *res) {

    // store entire matrix in shared memory for fast reads.

    __shared__ float s_a[32 * 32];

    // store result in shared memory for fast writes.

    __shared__ float s_res[32 * 32];

    // assign each thread an index so that threads in the same warp process

    // elements in the same row.

    const uint row_i = threadIdx.x + 32 * threadIdx.y;

    // copy matrix from global memory to shared memory in a coalesced fashion.

    s_a[row_i] = a[row_i];

    // intialize result as a matrix where each element is 1.0.

    s_res[row_i] = 1.0;

    // in order to process the matrix column-by-column... all warps must

    // finish initializing shared memory row-by-row.

    __syncthreads();

    // assign each thread an index so that threads in the same warp process

    // elements in the same column.

    const uint col_i = threadIdx.y + 32 * threadIdx.x;

    // a single block computes the power of the entire matrix.

    // each warp in the block computes the power of a single column.

    // each thread in the warp computes the power of a single element.

    while (pow > 0) {

        // Note that col_i % 32 = threadIdx.y.

        // Since all threads in the same warp have the same threadIdx.y, this

        // is a 32-way bank conflict!

        s_res[col_i] *= s_a[col_i];

        pow -= 1;

    }

    // in order to read the matrix row-by-row... all warps must

    // finish initializing shared memory column-by-column.

    __syncthreads();

    // copy result from shared memory to global memory in a coalesced fashion.

    res[row_i] = s_res[row_i];

};

/**

* For a matrix of size 32 x 32, computes (a_ij)^(pow) for each element a_ij

* and stores in res_ij.

*

* After reading the matrix a into local memory row by row, we

* compute the power of each element on a column by column basis.

* Due to zero padding, we don't have a bank conflict.

*

* Note that __syncthreads is necessary here because the same shared

* memory is accessed by multiple warps.

*/

__global__ void pow_cols_pad(const float *a, uint pow, float *res) {

    // store entire matrix in shared memory for fast reads.

    __shared__ float s_a[33 * 33];

    // store result in shared memory for fast writes.

    __shared__ float s_res[33 * 33];

    // assign each thread an index so that threads in the same warp process

    // elements in the same column.

    const uint row_i = threadIdx.x + 32 * threadIdx.y;

    // copy matrix from global memory to shared memory in a coalesced fashion.

    s_a[row_i] = a[row_i];

    // intialize result as a matrix where each element is 1.0.

    s_res[row_i] = 1.0;

    // assign each thread an index so that threads in the same warp process

    // elements in the same column.

    const uint col_i = threadIdx.y + 33 * threadIdx.x;

    // in order to process the matrix column-by-column... all warps must

    // finish initializing shared memory row-by-row.

    __syncthreads();

    // a single block computes the power of the entire matrix.

    // each warp in the block computes the power of a single column.

    // each thread in the warp computes the power of a single element.

    while (pow > 0) {

        // Results from number theory: Additive group of integers mod n is

        // generated by all integers m relatively prime to n. A warp conflict occurs

        // if two threads in a warp access the same address mod 32. We

        // minimize bank conflicts by reading and writing data to shared memory

        // with a stride m relatively prime to n.

        //

        // Even though we are reading data column-by-column, we don't have

        // bank conflicts since our stride is relatively prime to 32.

        // For larger matrices (size n), we should choose a stride that is

        // relatively prime to 32. It is useful to note that for any integer n,

        // gcd(n, n + 1) = 1.

        s_res[col_i] *= s_a[col_i];

        pow -= 1;

    }

    // in order to read the matrix row-by-row... all warps must

    // finish initializing shared memory column-by-column.

    __syncthreads();

    // copy result from shared memory to global memory in a coalesced fashion.

    res[row_i] = s_res[row_i];

};
