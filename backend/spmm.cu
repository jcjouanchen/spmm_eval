#include <stdio.h>
#include <assert.h>

typedef unsigned char uchar;       // 8
typedef unsigned short ushort;     // 16
typedef unsigned long long ullong; // 64

#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X)+BITWIDTH-1)>>LOG_BITWIDTH)
#define FEIL(X) ((((X)+BITWIDTH-1)>>LOG_BITWIDTH)<<LOG_BITWIDTH)

// A faster way to obtain lane id in a warp
#define GET_LANEID              \
    unsigned laneid;            \
    asm("mov.u32 %0, %%laneid;" \
        : "=r"(laneid));

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store64(const void *addr, T a, T b)
{
    *((float2 *)addr) = make_float2(*(float *)(&a), *(float *)(&b));
}
//For higher memory access efficiency

template <typename T>
__device__ __inline__ void store128(const void *addr, T a, T b, T c, T d)
{
    *((float4 *)addr) = make_float4(*(float *)(&a), *(float *)(&b), *(float *)(&c), *(float *)(&d));
}

#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X)+BITWIDTH-1)>>LOG_BITWIDTH)
#define FEIL(X) ((((X)+BITWIDTH-1)>>LOG_BITWIDTH)<<LOG_BITWIDTH)

#define BITWIDTH64 64
#define LOG_BITWIDTH64 6
#define CEIL64(X) (((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)
#define FEIL64(X) ((((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)<<LOG_BITWIDTH64)


//======================================================================================
// bit-packing
//======================================================================================
// for dense unpack
/** @brief Unpack 32-bit row-major unsigned activation matrix into floating-point.
 *
 *  Unpack compact 32-bit unsigned layer output activation matrix into floating-point for 
 *  validation purpose.
 *
 *  @return Void.
 */

// for dense pack
template <typename T>
__global__ void ToBit32RowDenseBin(const T *__restrict__ A, unsigned *B,
                                const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval = 0;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = ((by * 32 + i < A_width) && (bx * 32 + laneid < A_height)) ? A[(bx * 32 + laneid) * A_width + by * 32 + i] : (T)-1;
        Bval = (Bval << 1) + (f0 > 0); // already binarized B
    }
    if (bx * gridDim.y * 32 + laneid * gridDim.y + by < A_height * gridDim.y)
        B[bx * gridDim.y * 32 + laneid * gridDim.y + by] = Bval;
}

template <typename T>
__global__ void ToBit32RowDenseSign(const T *__restrict__ A, unsigned *B, 
                                    const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval = 0;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = ((by * 32 + i < A_width) && (bx * 32 + laneid < A_height)) ? A[(bx * 32 + laneid) * A_width + by * 32 + i] : (T)-1;
        Bval = (Bval << 1) + (f0 >= 0);
    }
    if (bx * gridDim.y * 32 + laneid * gridDim.y + by < A_height * gridDim.y)
        B[bx * gridDim.y * 32 + laneid * gridDim.y + by] = Bval;
}

template <typename T>
__global__ void ToBit32ColUd(const T *__restrict__ A, unsigned *B,
                             const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = ((by * 32 + laneid < A_width) && (bx * 32 + i < A_height)) ? A[(bx * 32 + i) * A_width + by * 32 + laneid] : (T)-1;
        unsigned r0 = __brev(__ballot(f0 >= 0));
        if (laneid == i)
            Bval = r0;
    }
    if (laneid < A_height * A_width)
        B[by * gridDim.x * 32 + bx * 32 + laneid] = Bval;
}

//======================================================================================
// From column-major 32-bit-array to row-major normal array. No padding.
//======================================================================================
template <typename T>
__global__ void Bit32ColTo(const unsigned *__restrict__ A, T *B,
                           const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Aval = A[by * A_height + bx * 32 + laneid];
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        unsigned r0 = __shfl(Aval, i); //from lane-i
        B[(32 * bx + i) * A_width * 32 + by * 32 + laneid] = (T)((r0 >> (31 - laneid)) & 0x1);
    }
}

// col-major packing bit 2
template <typename T>
__global__ void ToBit2Col(const T *__restrict__ A, uchar *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/64)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
    T f0;

#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        f0 = by * 16 * 64 + i * 16 * 2 + laneid < nblocks * 16 ? A[by * 16 * 64 + i * 16 * 2 + laneid] : 0; // <-- laneid will get consecutive 32 (2-blocks)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0)); //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }

    // layout block0 at high-16
    B[by * 2 * 64 + laneid * 2 * 2] = (Bval & 0xF0000000) >> 28;
    B[by * 2 * 64 + laneid * 2 * 2 + 1] = (Bval & 0x0F000000) >> 24;
    B[by * 2 * 64 + laneid * 2 * 2 + 2] = (Bval & 0x00F00000) >> 20;
    B[by * 2 * 64 + laneid * 2 * 2 + 3] = (Bval & 0x000F0000) >> 16;

    // layout block1 at low-16
    B[by * 2 * 64 + laneid * 2 * 2 + 4] = (Bval & 0x0000F000) >> 12;
    B[by * 2 * 64 + laneid * 2 * 2 + 5] = (Bval & 0x00000F00) >> 8;
    B[by * 2 * 64 + laneid * 2 * 2 + 6] = (Bval & 0x000000F0) >> 4;
    B[by * 2 * 64 + laneid * 2 * 2 + 7] = (Bval & 0x0000000F);
}

// col-major packing bit 4
template <typename T>
__global__ void ToBit4Col(const T *__restrict__ A, uchar *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/64)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
    T f0;

#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        f0 = by * 16 * 64 + i * 16 * 2 + laneid < nblocks * 16 ? A[by * 16 * 64 + i * 16 * 2 + laneid] : 0; // <-- laneid will get consecutive 32 (2-blocks)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));                                    //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }

    // layout block0 at high-16
    B[by * 4 * 64 + laneid * 4 * 2] = (Bval & 0xF0000000) >> 28;
    B[by * 4 * 64 + laneid * 4 * 2 + 1] = (Bval & 0x0F000000) >> 24;
    B[by * 4 * 64 + laneid * 4 * 2 + 2] = (Bval & 0x00F00000) >> 20;
    B[by * 4 * 64 + laneid * 4 * 2 + 3] = (Bval & 0x000F0000) >> 16;

    // layout block1 at low-16
    B[by * 4 * 64 + laneid * 4 * 2 + 4] = (Bval & 0x0000F000) >> 12;
    B[by * 4 * 64 + laneid * 4 * 2 + 5] = (Bval & 0x00000F00) >> 8;
    B[by * 4 * 64 + laneid * 4 * 2 + 6] = (Bval & 0x000000F0) >> 4;
    B[by * 4 * 64 + laneid * 4 * 2 + 7] = (Bval & 0x0000000F);
}

// row-major packing bit 4
template <typename T>
__global__ void ToBit4Row(const T *__restrict__ A, uchar *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 4))
    {
        unsigned Bval = 0;
        T f0;

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            if (i % 8 < 4)
                f0 = (T)(0); // high-4 bit remain 0
            else
                f0 = A[bx * 4 * 4 + (i - 4 * ((i / 8) + 1))];

            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx * 4] = (Bval & 0xFF000000) >> 24;
        B[bx * 4 + 1] = (Bval & 0x00FF0000) >> 16;
        B[bx * 4 + 2] = (Bval & 0x0000FF00) >> 8;
        B[bx * 4 + 3] = Bval & 0x000000FF;
    }
}

// col-major packing bit 8
// process 4 8x8x4 at the same time
template <typename T>
__global__ void ToBit8Col(const T *__restrict__ A, uchar *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/16)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = by * 8 * 8 * 4 * 4 + i * 32 + laneid < nblocks * 8 * 8 ? A[by * 8 * 8 * 4 * 4 + i * 32 + laneid] : 0; // <-- laneid will get consecutive 32 (half-block)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0));                                             //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }

    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4] = (Bval & 0xFF000000) >> 24;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 1] = (Bval & 0x00FF0000) >> 16;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 2] = (Bval & 0x0000FF00) >> 8;
    B[by * 8 * 4 * 4 + (laneid / 2) * 8 + laneid % 2 * 4 + 3] = Bval & 0x000000FF;
}

// row-major packing bit 8
template <typename T>
__global__ void ToBit8Row(const T *__restrict__ A, uchar *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 4))
    {
        unsigned Bval = 0;

#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 8 * 4 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx * 4] = (Bval & 0xFF000000) >> 24;
        B[bx * 4 + 1] = (Bval & 0x00FF0000) >> 16;
        B[bx * 4 + 2] = (Bval & 0x0000FF00) >> 8;
        B[bx * 4 + 3] = Bval & 0x000000FF;
    }
}

// col-major packing bit 16
template <typename T>
__global__ void ToBit16Col(const T *__restrict__ A, ushort *B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/4)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = by * 16 * 16 * 4 + i * 16 * 2 + laneid < nblocks * 16 * 16 ? A[by * 16 * 16 * 4 + i * 16 * 2 + laneid] : 0;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0)); //__brev(__ballot(f0>0));

        if (laneid == i)
            Bval = r0;
    }

    B[by * 16 * 4 + laneid * 2] = (Bval & 0xFFFF0000) >> 16;
    B[by * 16 * 4 + laneid * 2 + 1] = (Bval & 0x0000FFFF);
}
// 4 16x16 at the same time

// row-major packing bit 16
template <typename T>
__global__ void ToBit16Row(const T *__restrict__ A, ushort *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows / 2))
    {
        unsigned Bval = 0;
#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 32 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }

        B[bx * 2] = (Bval & 0xFFFF0000) >> 16;
        B[bx * 2 + 1] = (Bval & 0x0000FFFF);
    }
}

// weight should be col-major packing, layout is 32 * (32*numofblocks)
// input should be row-major packing, layout is whatever it is originally

// col-major packing bit 32
template <typename T>
__global__ void ToBit32Col(const T *__restrict__ A, unsigned *B, const int A_height, const int A_width) // blocksize, nblocks * blocksize
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // nblocks
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = A[by * 32 * 32 + i * 32 + laneid];
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0 > 0)); //__brev(__ballot(f0>0));
        if (laneid == i)
            Bval = r0;
    }
    B[by * 32 + laneid] = Bval;
}

// row-major packing bit 32
template <typename T>
__global__ void ToBit32Row(const T *__restrict__ A, unsigned *B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows)
    {
        unsigned Bval = 0;
#pragma unroll
        for (int i = 0; i < 32; i++)
        {
            T f0 = A[bx * 32 + i];
            Bval = (Bval << 1) + (f0 > 0);
        }
        B[bx] = Bval;
    }
}

// col-major packing bit 64
template <typename T>
__global__ void ToBit64Col(const T *__restrict__ A, ullong *B, const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; //nblocks
    const unsigned bx = blockIdx.x; // 2 <- set this
    ullong Bval;
#pragma unroll
    for (int i = 0; i < 32; i++)
    {
        T f0 = A[by * 64 * 64 + bx * 64 * 32 + i * 64 + laneid];
        T f1 = A[by * 64 * 64 + bx * 64 * 32 + i * 64 + 32 + laneid];
        unsigned r0 = __ballot_sync(0xFFFFFFFF, f0 > 0);
        unsigned r1 = __ballot_sync(0xFFFFFFFF, f1 > 0);

        //        unsigned r0 = __ballot(f0>0);
        //        unsigned r1 = __ballot(f1>0);

        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};"
                     : "=l"(l0)
                     : "r"(r0), "r"(r1)); //lo,hi
        if (laneid == i)
            Bval = __brevll(l0);
    }
    B[by * 64 + bx * 32 + laneid] = Bval;
}

// row-major packing bit 64
template <typename T>
__global__ void ToBit64Row(const T *__restrict__ A, ullong *B, const int A_height, const int A_width, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows)
    {
        GET_LANEID;

        ullong Bval = 0;
#pragma unroll
        for (int i = 0; i < 64; i++)
        {
            T f0 = A[bx * 64 + i];
            Bval = (Bval << 1) | (f0 > 0);
        }
        B[bx] = Bval;
    }
}

//======================================================================================
// spmm FBF
//======================================================================================
// ---------------------------------------------------------------------------
// [[Old-FBF-8]  
// shared mem used 32KB per tb (reach max)
// ---------------------------------------------------------------------------
// [cora] b2sr: 14.746 / csr-float: 46.784 / speedup: 3.173
// [pubmed] b2sr: 48.096 / csr-float: 53.280 / speedup: 1.108
// [citeseer] b2sr: 11.674 / csr-float: 39.712 / speedup: 3.402
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_1_1024_lessthan8(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                                const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                                const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32 + warpid;
    
    // 32-thread in a warp
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        __shared__ float B_shared[32*32*8];
        register float Ctemp = 0;

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<B_cols; j++)
            {
                unsigned ind = (colindsub[i+(laneid/4)]*4+(laneid%4))*B_cols+j;
                B_shared[warpid*32*8+laneid*8+j] = (i+laneid/4 < load) ? Bsub[ind] : 0;
            }

            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            // every 4 lane work on one B_cols dimension
            // so B_cols cannot exceed 8
            for(int k=0; k<32; k++)
            {
                float fval = B_shared[warpid*32*8+k*8+(laneid/4)]; 
                Ctemp += ((a >> (31-k)) & 0x1)?fval:0;
            }
        }

        // store
        if((laneid/4)<B_cols) Csub[(laneid%4)*B_cols+(laneid/4)] = Ctemp;
    }
}

// ---------------------------------------------------------------------------
// [Old-FBF-32]
// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_1_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 32;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid)*32+laneid < B_cols)
        {
            Csub[(warpid)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid)*32+laneid] = Ctemp[3];
        }
    }
}

// ---------------------------------------------------------------------------
// [Old-FBF-64]  
// ---------------------------------------------------------------------------
// [reddit] b2sr: 449085.840 / csr-float: 2097836.426 / speedup: 4.671
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_2_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 16 + warpid/2;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid%2)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid%2)*32+laneid < B_cols)
        {
            Csub[(warpid%2)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid%2)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid%2)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid%2)*32+laneid] = Ctemp[3];
        }
    }
}

// ---------------------------------------------------------------------------
// [Old-FBF-128] 
// ---------------------------------------------------------------------------
// 
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_4_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 8 + warpid/4;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid%4)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid%4)*32+laneid < B_cols)
        {
            Csub[(warpid%4)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid%4)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid%4)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid%4)*32+laneid] = Ctemp[3];
        }
    }
}

// ---------------------------------------------------------------------------
// [Old-FBF-256]
// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_8_1024(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                       const int *__restrict__ rowptr, const int *__restrict__ colind, const int nblockrows,
                                       const int B_cols)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    GET_LANEID;
    const unsigned warpid = (threadIdx.x >> 5);
    int row = bx * 4 + warpid/8;
    
    if (row < nblockrows)
    {
        int row_start, row_end, load = 0;
        row_start = rowptr[row];
        row_end = rowptr[row + 1];
        load = row_end - row_start;

        const uchar *Asub = &(A[row_start * 4]);
        const float *Bsub = &(B[0]);
        const int *colindsub = &(colind[row_start]);
        float *Csub = &(C[row*4*B_cols]);

        register unsigned A_shared = 0;
        register unsigned a = 0;
        register float b[32] = {0}; // cannot be 32*2 and use 1 warp
        register float Ctemp[4] = {0};

        for(int i=0; i<((load+8-1)/8)*8; i+=8) 
        {
            // preload A, B
            A_shared = (i+laneid/4 < load) ? (Asub[(i+(laneid/4))*4+(laneid%4)] & 0x0000000F) : 0;
            for(int j=0; j<32; j++)
            {
                b[j] = (i+j/4 < load) ? Bsub[(colindsub[i+(j/4)]*4+(j%4))*B_cols+(warpid%8)*32+laneid] : 0;
            }
            
            // layout A as 8 uchar into 1 unsigned
            // every 4 lane has the same copy of data
            // r0, r1, r2, r3 | r0, r1, r2, r3 | r0, r1, r2, r3 ...
            a = __shfl_sync(0xFFFFFFFF, A_shared, 0+laneid%4) << 28 | __shfl_sync(0xFFFFFFFF, A_shared, 4+laneid%4) << 24 
                | __shfl_sync(0xFFFFFFFF, A_shared, 8+laneid%4) << 20 | __shfl_sync(0xFFFFFFFF, A_shared, 12+laneid%4) << 16
                | __shfl_sync(0xFFFFFFFF, A_shared, 16+laneid%4) << 12 | __shfl_sync(0xFFFFFFFF, A_shared, 20+laneid%4) << 8 
                | __shfl_sync(0xFFFFFFFF, A_shared, 24+laneid%4) << 4 | __shfl_sync(0xFFFFFFFF, A_shared, 28+laneid%4);

            // compute
            for(int j=0; j<4; j++)
            {
                for(int k=0; k<32; k++)
                {
                    Ctemp[j] += (((__shfl_sync(0xFFFFFFFF, a, j)) >> (31-k)) & 0x1) ? b[k] : 0;
                }
            }
        }

        // store
        if ((warpid%8)*32+laneid < B_cols)
        {
            Csub[(warpid%8)*32+laneid] = Ctemp[0];
            Csub[B_cols+(warpid%8)*32+laneid] = Ctemp[1];
            Csub[B_cols*2+(warpid%8)*32+laneid] = Ctemp[2];
            Csub[B_cols*3+(warpid%8)*32+laneid] = Ctemp[3];
        }
    }
}


// ---------------------------------------------------------------------------
// [New-FBF-ver1] 1thrd-per-tile, 32-tile-per-iter, 1-tilerow-at-a-time
// ---------------------------------------------------------------------------
// Reddit-64 166.419ms -> 2469.361ms
// Flickr-32         -> 20.484ms
// Flickr-64 1.781ms -> 32.078ms
// Flickr-128        -> 59.797ms
// Flickr-256        -> 118.552ms 
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_new(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, 
                                    const int nblockrows, const int B_cols)
{
    GET_LANEID;
    int K = B_cols / 32; // K = 2 for B_cols = 64
    int tiles_per_tb = 1024 / B_cols; // tiles_per_tb = 16 for B_cols = 64

    const unsigned warpid = (threadIdx.x >> 5);
    int row = blockIdx.x * tiles_per_tb + (warpid/K);
    
    // Compute the column index of `mat` in which the thread is operating.
    int mat_col_idx = (warpid%K) * 32 + laneid;

    if (row < nblockrows)
    {
        int row_start = __ldg(rowptr + (row % nblockrows));
        int row_end = __ldg(rowptr + (row % nblockrows) + 1);
        int col_idx = row_start + laneid;
        // if (threadIdx.x == 0) {printf("row: %d, row_start: %d, row_end: %d, col_idx: %d, rowptr[row]:%d , rowptr[row+1]:%d\n", 
        // row, row_start, row_end, col_idx, rowptr[row], rowptr[row+1]);}

        register float result[4] = {0};

        // process 32 tiles on a row at a time, each by a lane
        for (int c = row_start; c < row_end; c += 32)
        {
            register int col[4][4] = {0};
            register int mat_rows[4][4][32] = {0};
            register float val = 0;

            if (col_idx < row_end)
            {
                // decode bit-tiles nonzero index
                int ind = __ldg(colind + col_idx);
                // if (threadIdx.x == 0) {printf("ind: %d\n", ind);}
                for (int i=0; i<4; i++)
                {
                    uchar r = __ldg(A + col_idx*4 + i);
                    for (int j=0; j<4; j++)
                    {
                        col[i][j] = ((r >> (3-j)) & 0x1) ? (ind*4+j): -1;
                        // if (threadIdx.x == 0) {printf("col[%d][%d]: %d\n", i, j, col[i][j]);}
                    }
                }
            } else 
            {
                for(int i=0; i<4; i++)
                {
                    for(int j=0; j<4; j++)
                    {
                        col[i][j] = -1;
                    }
                }
            }
            
            col_idx += 32;

            #pragma unroll
            for (int i=0; i<4; i++)
            {
                for (int j=0; j<4; j++)
                {
                    for (int l=0; l <32; l++)
                    {
                        mat_rows[i][j][l] = __shfl_sync(0xffffffff, col[i][j], l);
                        // if (threadIdx.x == 0) {printf("mat_rows[%d][%d][%d]: %d\n", i, j, l, mat_rows[i][j][l]);}
                    }
                }
            }

            for (int i=0; i<4; i++)
            {
                for (int j=0; j<4; j++)
                {
                    for(int l=0; l<32; l++)
                    {    
                        if (mat_rows[i][j][l] != -1)
                        {
                            // Coalesced memory access into `mat`.
                            val = __ldg(B + mat_rows[i][j][l]*B_cols + mat_col_idx);
                            // if (threadIdx.x == 0) {printf("B[%d] = %.f\n", (mat_rows[i][j][l]*B_cols + mat_col_idx), B[(mat_rows[i][j][l]*B_cols + mat_col_idx)]);}
                            // if (threadIdx.x == 0) {printf("val = %.f\n", val);}
                            result[i] = result[i] + val;
                            // if (threadIdx.x == 0) {printf("result[%d] = %.f\n", i, result[i]);}
                        }
                    }
                }
            }
        }

        //store
        for(int i=0; i<4; i++)
        {
            if (result[i] != 0)
            {
                int out_idx = (row*4+i) * B_cols + mat_col_idx;
                *(C + out_idx) = result[i];
            }
        }
    } // if (row < nblockrows)
}

// ---------------------------------------------------------------------------
// [New-FBF-ver2 (current best)] 4thrd-per-tile, 8-tile-per-iter, 1-tilerow-at-a-time
// ---------------------------------------------------------------------------
// Reddit-64 166.419ms -> 97.027ms
// Reddit-128          -> 166.101ms
// Reddit-256          -> 290.319ms
// Flickr-32         -> 1.626 ms
// Flickr-64 1.781ms -> 1.654ms
// Flickr-128        -> 1.955 ms
// Flickr-256        -> 2.648ms
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_new2(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, 
                                    const int nblockrows, const int B_cols)
{
    GET_LANEID;
    int K = B_cols / 32; // K = 2 for B_cols = 64
    int tiles_per_tb = 1024 / B_cols; // tiles_per_tb = 16 for B_cols = 64

    const unsigned warpid = (threadIdx.x >> 5);
    int row = blockIdx.x * tiles_per_tb + (warpid/K);
    
    // Compute the column index of `mat` in which the thread is operating.
    int mat_col_start_idx = (warpid%K) * 32 + (laneid/4) * 4;

    if (row < nblockrows)
    {
        int row_start = __ldg(rowptr + (row % nblockrows));
        int row_end = __ldg(rowptr + (row % nblockrows) + 1);
        int col_tile_idx = row_start + laneid/4;

        register float result[4] = {0};

        // process 8 tiles on a row at a time, each by 4 lanes
        for (int c = row_start; c < row_end; c += 8, col_tile_idx += 8)
        {
            register int col_idx[4] = {0};
            if (col_tile_idx < row_end) {
                // decode bit-tiles nonzero index
                int ind = __ldg(colind + col_tile_idx);
                uchar r = __ldg(A + col_tile_idx*4 + (laneid%4));
                for (int j=0; j<4; j++) {
                    col_idx[j] = ((r >> (3-j)) & 0x1) ? (ind*4+j): -1;
                    // if (threadIdx.x == 0) {printf("col_idx[%d]: %d\n", j, col_idx[j]);}
                }
            } 
            else {
                for(int j=0; j<4; j++) {
                    col_idx[j] = -1;
                    // if (threadIdx.x == 0) {printf("col_idx[%d]: %d\n", j, col_idx[j]);}
                }
            }

            register int all_lane_col_idx[4][8] = {0};
            #pragma unroll
            for (int j=0; j<4; j++) // for 4 column (tile size)
            {
                for (int l=0; l<8; l++) // laneid/4 (tile size)
                {
                    all_lane_col_idx[j][l] = __shfl_sync(0xffffffff, col_idx[j], (l*4+(laneid%4)));
                    // if (threadIdx.x == 0) {printf("mat_rows[%d][%d]: %d\n", j, l, mat_rows[j][l]);}
                }
            }

            register float val = 0;
            for (int j=0; j<4; j++){
                for(int l=0; l<8; l++){  
                    if (all_lane_col_idx[j][l] != -1) {
                        // Coalesced memory access into `mat`.
                        // cannot use ldg here (only works for m = 32)
                        // val = __ldg(B + all_lane_col_idx[j][l]*B_cols + mat_col_start_idx + m); 
                        for(int m=0; m<4; m++) {
                            val = B[all_lane_col_idx[j][l]*B_cols + mat_col_start_idx + m]; 
                            result[m] = result[m] + val;
                        }
                    }
                }
            }
        }

        //store
        for (int m=0; m<4; m++)
        {
            if (result[m] != 0)
            {
                int out_idx = (row*4+(laneid%4)) * B_cols + mat_col_start_idx + m;
                *(C + out_idx) = result[m];
            }
        }
    } // if (row < nblockrows)
}


// ---------------------------------------------------------------------------
// [New-FBF-ver3] 4thrd-per-tile, 8-tile-per-iter, 1-tilerow-at-a-time
// ---------------------------------------------------------------------------
// Reddit-64 166.419ms -> 96.055ms
// Reddit-128       -> 169.694ms
// Reddit-256       -> 305.326ms
// Flickr-32        -> 2.119ms
// Flickr-64 1.781ms -> 2.466ms
// Flickr-128     -> 2.717ms
// Flickr-256     -> 2.973ms
// ---------------------------------------------------------------------------
__global__ void spmm4_full_full_new3(const uchar *__restrict__ A, const float *__restrict__ B, float *C, 
                                    const int *__restrict__ rowptr, const int *__restrict__ colind, 
                                    const int nblockrows, const int B_cols)
{
    GET_LANEID;
    int K = B_cols / 32; // K = 2 for B_cols = 64
    int tiles_per_tb = (1024 / B_cols)*2; // tiles_per_tb = 32 for B_cols = 64

    const unsigned warpid = (threadIdx.x >> 5);
    int row = blockIdx.x * tiles_per_tb + (warpid/K)*2 + (laneid/16);
    // if (threadIdx.x%16 == 0) printf("threadIdx.x=%d, row=%d\n", threadIdx.x, row);
    
    // Compute the column index of `mat` in which the thread is operating.
    // int mat_col_start_idx = (warpid%K) * 32 + ((laneid-((laneid/16)*16))/4)*8;
    int mat_col_start_idx = (warpid%K) * 32 + (laneid%16)/4*8;

    if (row < nblockrows)
    {
        int row_start = __ldg(rowptr + (row % nblockrows));
        int row_end = __ldg(rowptr + (row % nblockrows) + 1);
        int col_tile_idx = row_start + (laneid%16)/4;

        register float result[8] = {0};

        // process 8 tiles on a row at a time, each by 4 lanes
        for (int c = row_start; c < row_end; c += 4, col_tile_idx += 4)
        {
            register int col_idx[4] = {0};
            if (col_tile_idx < row_end) {
                // decode bit-tiles nonzero index
                int ind = __ldg(colind + col_tile_idx);
                uchar r = __ldg(A + col_tile_idx*4 + (laneid%4));
                for (int j=0; j<4; j++) {
                    col_idx[j] = ((r >> (3-j)) & 0x1) ? (ind*4+j): -1;
                    // if (threadIdx.x == 0) {printf("col_idx[%d]: %d\n", j, col_idx[j]);}
                }
            } 
            else {
                for(int j=0; j<4; j++) {
                    col_idx[j] = -1;
                    // if (threadIdx.x == 0) {printf("col_idx[%d]: %d\n", j, col_idx[j]);}
                }
            }

            register int all_lane_col_idx[4][4] = {0};
            #pragma unroll
            for (int j=0; j<4; j++) // for 4 column (tile size)
            {
                for (int l=0; l<4; l++) // laneid/4 (tile size)
                {
                    all_lane_col_idx[j][l] = __shfl_sync(0xffffffff, col_idx[j], ((laneid/16)*16+l*4+(laneid%4)));
                    // if (threadIdx.x == 0) {printf("mat_rows[%d][%d]: %d\n", j, l, mat_rows[j][l]);}
                }
            }

            register float val = 0;
            for (int j=0; j<4; j++){
                for(int l=0; l<4; l++){  
                    if (all_lane_col_idx[j][l] != -1) {
                        // Coalesced memory access into `mat`.
                        // cannot use ldg here (only works for m = 32)
                        for(int m=0; m<8; m++) {
                            val = B[all_lane_col_idx[j][l]*B_cols + mat_col_start_idx + m]; 
                            result[m] = result[m] + val;
                        }
                    }
                }
            }
        }

        //store
        for (int m=0; m<8; m++)
        {
            if (result[m] != 0)
            {
                int out_idx = (row*4+(laneid%4)) * B_cols + mat_col_start_idx + m;
                *(C + out_idx) = result[m];
            }
        }
    } // if (row < nblockrows)
}