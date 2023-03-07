//======================================================================================
// This function performs 32-bit Matmul and assumes no padding. A and B are 32-bit-array,
// C is normal array in row-major. The dot product is among A-row and B-row.
// A(A_width, A_height) * B(B_height, B_width) = C(A_height, B_width), A_width = B_height
//======================================================================================

// ---------------------------------------------------------------------------
// process 32 Arows in a warp. (througputs)
// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------

// ToBit32Row<float><<<dim3(N/32,N/32), 32>>>(fB, tB, N, N);
// ToBit32Col<float><<<dim3(N/32,N/32), 32>>>(fA, tA, N, N);
// BMM32_Arow_Brow<float><<<dim3(N/32,N/32), 32>>>(tA, tB, fC, N, N/32, N);

template <typename T>
__global__ void __launch_bounds__(32, 32)
    BMM32_Arow_Brow(const unsigned *__restrict__ A, const unsigned *__restrict__ B,
                    T *C, const int A_height, const int A_width, const int B_width) // M, N/32, K
{
    GET_LANEID;
    const unsigned *Asub = &A[blockIdx.x * 32];
    const unsigned *Bsub = &B[blockIdx.y * 32];
    T *Csub = &C[blockIdx.x * B_width * 32 + blockIdx.y * 32];
    register unsigned Cm[32] = {0};

    for (int i = 0; i < A_width; i++)
    {
        unsigned r0 = Asub[i * A_height + laneid]; // process 32 row a time, *A_width if we want horizontal layout
        unsigned r1 = Bsub[i * B_width + laneid];

#pragma unroll
        for (int j = 0; j < 32; j++)
        {
            unsigned r2 = __shfl(r1, j); //from lane-j, r1 of matrix B
            Cm[j] += __popc(r0 ^ r2);    //can remove C to exploit register reuse
        }
    }

    //This is for saving as st.128 to improve memory access efficiency
    //It is to optimize the following original version:
    //for (int i=0; i<32; i++)
    //Csub[laneid*B_width+i] = 32*A_width - (T)(Cm[i])*2;
    for (int i = 0; i < 32; i += 4)
    {
        store128((void *)&Csub[laneid * B_width + i],
                 32 * A_width - (T)(Cm[i + 0]) * 2,
                 32 * A_width - (T)(Cm[i + 1]) * 2,
                 32 * A_width - (T)(Cm[i + 2]) * 2,
                 32 * A_width - (T)(Cm[i + 3]) * 2);
    }
}


// ToBit32Row<float><<<dim3(N/32,N/32), 32>>>(fB, tB, N, N);
// ToBit32Col<float><<<dim3(N/32,N/32), 32>>>(fA, tA, N, N);
// BMM32_MT_M_S<<<dim3(N/32,N/32), 1024>>>(tA, tB, fC, N, N/32, N);
__global__ void BMM32_MT_M_S(const unsigned *__restrict__ A, const unsigned *__restrict__ B, float *C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    const unsigned *Asub = &A[blockIdx.x * 32];
    const unsigned *Bsub = &B[blockIdx.y * 32];
    float *Csub = &C[blockIdx.x * k * 32 + blockIdx.y * 32];
    register unsigned Cc = 0;

    for (int i = 0; i < n; i++)
    {
        unsigned r0 = Asub[i * m + warpid];
        unsigned r1 = Bsub[i * k + laneid];
        Cc += __popc(r0 ^ r1);
    }
    Csub[warpid * k + laneid] = 32 * n - (float)Cc * 2;
}


// BMM32_BIN<<<dim3(N/32,N/32), 32>>>(tA, tB, uC, N, N, N);
__global__ void BMM32_BIN(const unsigned *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned *Asub = &A[blockIdx.x * 32];
    const unsigned *Bsub = &B[blockIdx.y * 32];
    unsigned *Csub = &C[blockIdx.y * m + blockIdx.x * 32];
    register int Cm[32] = {0};
    for (int i = 0; (i * 32) < n; i++)
    {
        unsigned r0 = Asub[i * m + laneid];
        unsigned r1 = Bsub[i * k + laneid];
#pragma unroll
        for (int j = 0; j < 32; j++)
        {
            unsigned r2 = __shfl(r1, j); //from lane-j, r1 of weight matrix
            Cm[j] += __popc(r0 ^ r2);
        }
    }
    unsigned c = 0;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        float t = n - 2 * (float)Cm[i];
        c = c + c + (t >= 0);
    }
    Csub[laneid] = c;
}

// BMM32S_BIN<<<dim3(N/32,N/32), 1024>>>(tA, tB, uC, N, N, N);
__global__ void BMM32S_BIN(const unsigned *__restrict__ A, const unsigned *__restrict__ B, unsigned *C, int m, int n, int k)
{
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    const unsigned *Asub = &A[blockIdx.x * 32];
    const unsigned *Bsub = &B[blockIdx.y * 32];
    unsigned *Csub = &C[blockIdx.x * k * 32 + blockIdx.y * 32];
    register unsigned Cc = 0;

    for (int i = 0; (32 * i) < n; i++)
    {
        unsigned r0 = Asub[i * m + warpid];
        unsigned r1 = Bsub[i * k + laneid];
        Cc += __popc(r0 ^ r1);
    }
    unsigned r2 = __ballot(n - 2 * (float)Cc >= 0);
    Csub[warpid] = __brev(r2);
}