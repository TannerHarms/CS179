PART 1

Question 1.1: Latency Hiding
-------------------------------------------------------------------------------------

For the GK110, there are 4 warp schedulers in each SM and 2 dispatchers.  Therefore, 
the GPU is able to start instructions in up to 8 warps at once.  Then, since the 
latency of each warp add is about 10 cycles (according to the notes) it requires 80 
warp instructions to hide the latency of a warp add.  

Question 1.2: Thread Divergence
-------------------------------------------------------------------------------------

Part a:
This code is not divergent since the it is indexing in y rather than in x.  Once the 
scheduler assigns the trheads according to the index, each warp will be executing 
either the if or the else statement, but none will be executing both.  Therefore, 
there is now warp divergence.  

Part b:
Here it can be argued that the warp is divergent since each iteration of the loop 
computes a different number of computations.  Therefore, since some will have to run
multiple times even though they only do one computation, the warp is divergent.


Question 1.3: Coalesced Memory Access
-------------------------------------------------------------------------------------

Part a:
This write is coalesced.  It will write to one 128 byte cache line per warp.  
Then there will be 32 128 byte cache lines.  

Part b:
In this case the write is not coalesced.  Since it indexes in y rather than x, each 
access to the data is done on a new cache line, creating 32 128 byte cache lines per 
warp.  Therefore, there are 32 * 32 = 1024 128 byte cache lines created in the 
computation.

Part c:
This is not coalesced, since the '1 +' at the start of the argument misaligns the 
memory allocation.  Therefore, for data that fits into 128 bytes, the stored data
is written to an additional 128 byte cache line.  So, there will be 64 128 byte cache
lines created for the computation.


Question 1.4: Bank Conflicts and Instruction Dependencies
-------------------------------------------------------------------------------------

Part a:
In this example there are no bank conflicts.  Multiple threads are not accessing the 
same bank at different locations.  However, there is some broadcasting that happens

Part b:
    line 1: output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
1        float L0 = lhs[i + 32 * k];          // load in the lhs for the first line.
2        float R0 = rhs[k + 128 * j];         // load in the rhs for the first line.
3        float result0 = output[i + 32 * j];  // load in the stored output value.  
4        result0 = L0 * R0 + result0;         // Do the computation (FMA operation)
5        output[i + 32 * j] = result0;        // Output the result.  
        
    line 2: output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
6        float L1 = lhs[i + 32 * (k + 1)];    // load in the lhs for the second line.
7        float R1 = rhs[(k + 1) + 128 * j];   // load in the rhs for the second line.
8        float result1 = output[i + 32 * j];  // load in the stored output value.  
9        result1 = L1 * R1 + result1;         // Do the computation (FMA operation)
10       output[i + 32 * j] = result1;        // Output the result.  

Together, the result can be simplified:
    Inside Expanded:
11       float L0 = lhs[i + 32 * k];          // load in the lhs for the first line.
12       float R0 = rhs[k + 128 * j];         // load in the rhs for the first line.
13       float L1 = lhs[i + 32 * (k + 1)];    // load in the lhs for the second line.
14       float R1 = rhs[(k + 1) + 128 * j];   // load in the rhs for the second line.
15       float result = output[i + 32 * j];   // load in the stored output value.  
16       result = L0 * R0 + L1 * R1 + result; // Do the computation (FMA operation)
17       output[i + 32 * j] = result;         // Output the result.  

Part c:
I already kind of did this in simplifying above.  If lines 1 to 10 were written together, 
then result1 would depend on result0, and output would depend on result 1.  These are 
all unnecessary dependencies, as output can be computed directly from L0, R0, L1, R1.  

Part d:
Therefore, to simplify the code even further, the initial code can be written with fewer
dependencies in the following way: 
    int i = threadIdx.x;
    int j = threadIdx.y;
    for (int k = 0; k < 128; k += 2) {
        output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j] 
            + lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
    }

Part e:
Loop unrolling could be applied here.  Either all 128 instances could be unrolled, or 
the increment size in the for loop could be made larger.  


Additional:  Time spent on assignment
-------------------------------------------------------------------------------------
All together: ~15 hours
Part 1: ~ 7 hours
Part 2: ~ 8 hours
*** The TA's (I worked with Ivy) were very helpful with this assignment.  If not for 
    time in office hours, these numbers would have been much larger.  Thanks!