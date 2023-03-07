#include <iostream>
#include <sys/time.h>
#include <bitset>

#define TEST_TIMES 1
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <vector>
#include "backend/readMtx.hpp"
#include "backend/csr2bsr_batch_bsrbmv.cu"

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

/// ======================
// csr metadata
int nrows, ncols, nnz;

// csr host
int *h_csrRowPtr, *h_csrColInd;
float *h_csrVal;

// csr device
int *csrRowPtr, *csrColInd;
float *csrVal;

// csc host
int *h_cscRowInd, *h_cscColPtr;

// csc device
int *cscRowInd, *cscColPtr;
float *cscVal;

// b2sr metadata
int mb, nb, nblockrows;
int nblocks;
int tiledim = 4;

// b2sr
int *bsrRowPtr, *bsrColInd;

// b2sr val
uchar *tA;

// result mat
float* fC;
float* hC;
float* hC_row_major;

// Bmat host
float *B;
float *B_col_major;
int nBrows, nBcols;
int outunit;

// Bmat device
float *fB;
unsigned *tB;

// cuSPARSE vec
float *dX, *dY;

// cuSPARSE result host
float *result_cusparsecsrspmmfloat;

// b2sr result host
float *result_b2srspmm;

// cusparse handles
cusparseMatDescr_t csr_descr = 0;
cusparseMatDescr_t bsr_descr = 0;
cudaStream_t streamId = 0;
cusparseHandle_t handle = 0;

/// ======================
void readMtxCSR(const char *filename)
{
    // graphblast mmio interface
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;
    char *dat_name;
    readMtx(filename, &row_indices, &col_indices, &values,
            &nrows, &ncols, &nnz, 0, false, &dat_name); // directed, mtxinfo

    h_csrRowPtr = (int *)malloc(sizeof(int) * (nrows + 1));
    h_csrColInd = (int *)malloc(sizeof(int) * nnz);
    h_csrVal = (float *)malloc(sizeof(float) * nnz);
    coo2csr(h_csrRowPtr, h_csrColInd, h_csrVal,
            row_indices, col_indices, values, nrows, ncols);

    // copy csr to device
    cudaMalloc(&csrRowPtr, sizeof(int) * (nrows + 1));
    cudaMalloc(&csrColInd, sizeof(int) * nnz);
    cudaMalloc(&csrVal, sizeof(float) * nnz);
    cudaMemcpy(csrRowPtr, h_csrRowPtr, sizeof(int) * (nrows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColInd, h_csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(csrVal, h_csrVal, sizeof(float) * nnz, cudaMemcpyHostToDevice);

    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    cudaMemset(csrVal, 1.0, nnz * sizeof(float));
}

void CSR2B2SR()
{
    // transform from csr to bsr using cuSPARSE API
    mb = (nrows + tiledim - 1) / tiledim;
    nb = (ncols + tiledim - 1) / tiledim;
    nblockrows = mb;

    // cuSPARSE API metadata setup
    cusparseCreateMatDescr(&csr_descr);
    cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreateMatDescr(&bsr_descr);
    cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseCreate(&handle);
    cusparseSetStream(handle, streamId);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    cudaMalloc((void **)&bsrRowPtr, sizeof(int) * (nblockrows + 1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, tiledim, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void **)&bsrColInd, sizeof(int) * nblocks);

    // print size
    // printf("Storage Info --------------\n");
    // unsigned bytes = (nrows + 1) * 4 + nnz * 4 * 2; // size in bytes
    // printf("nrows: %d, nnz: %d\n", nrows, nnz);
    // printf("CSR size: ");
    // printBytes(bytes);
    // printf("\n");
    // bytes = (nblockrows + 1) * 4 + nblocks * 4 + nblocks * tiledim * ((tiledim + 8 - 1) / 8); // size in bytes
    // printf("B2SR size: ");
    // printBytes(bytes);
    // printf("\n");

    // malloc packed matrix & pack
    cudaMalloc((void **)&tA, nblocks * tiledim * sizeof(uchar));
    csr2bsr_batch_4(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                    bsrRowPtr, bsrColInd, tA, tiledim, nblockrows, nblocks);
}

void readBMtx(char *filename, int nBrows, int nBcols)
{
    FILE *cf = fopen(filename, "r");
    B = (float *)malloc(sizeof(float) * nBrows * nBcols);
    if (cf == NULL)
    {
        fprintf(stderr, "NULL pointer to file.\n");
        exit(1);
    }
    for (int i = 0; i < nBrows * nBcols; i++)
        fscanf(cf, "%f", &B[i]);
}

void binarizeBMtxSign()
{
    for (int i = 0; i < nBrows * nBcols; i++) 
    {
        if (B[i] >= 0) B[i] = 1.0;
        else B[i] = -1.0;
    }

    // print B
    // printf("B (float) --------------\n");
    // for (int i = 0; i < 5; i++) 
    // {
    //     for (int j=0; j<nBcols; j++) 
    //     {

    //         printf("%.f", B[i*nBcols+j]); //print +1/-1
    //     }
    //     printf("\n");
    // }
}

void packBasSign32full()
{
    // copy to device
    cudaMalloc(&fB, nrows * nBcols * sizeof(float));
    cudaMemcpy(fB, B, nrows * nBcols * sizeof(float), cudaMemcpyHostToDevice); // the rest are paddings
}

double evalCSRSpmmFloatCuSPARSE() // cusparse spmm
{
    // covert B from row-major to col-major
    B_col_major = (float *)malloc(sizeof(float) * nBrows * nBcols);
    int cnt = 0;
    for (int i=0; i<nBcols; i++) 
    {
        for (int j=0; j<nBrows; j++)
        {
            B_col_major[cnt++] = B[j*nBcols+i];
        }
    }

    // Host problem definition
    int A_num_rows = nrows;
    int A_num_cols = ncols;
    int A_nnz = nnz;
    int B_num_rows = nBrows;
    int B_num_cols = nBcols;
    int ldb = B_num_rows;
    int ldc = A_num_rows;
    int B_size = ldb * B_num_cols;
    int C_size = ldc * B_num_cols;
    int *hA_csrOffsets = h_csrRowPtr;
    int *hA_columns = h_csrColInd;
    float *hA_values = h_csrVal;
    float *hB = B_col_major;
    hC = (float *)malloc(sizeof(float) * C_size);
    for (int i = 0; i < C_size; i++)
    {
        hC[i] = 0.0f;
    } 

#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;

    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&dB, B_size * sizeof(float)))
    CHECK_CUDA(cudaMalloc((void **)&dC, C_size * sizeof(float)))

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB, hB, B_size * sizeof(float),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dC, hC, C_size * sizeof(float),
                          cudaMemcpyHostToDevice))
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL))
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    // execute SpMM
    GpuTimer csr_timer;
    csr_timer.Start();
    CHECK_CUSPARSE(cusparseSpMM(handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
    csr_timer.Stop();
    double cusparsecsrspmmfloat_time = csr_timer.ElapsedMillis();

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA(cudaMemcpy(hC, dC, C_size * sizeof(float),
                          cudaMemcpyDeviceToHost))
    // hC from col-major to row-major
    hC_row_major = (float *)malloc(sizeof(float) * C_size);
    cnt = 0;
    for (int i=0; i<nBrows; i++) 
    {
        for (int j=0; j<nBcols; j++)
        {
            hC_row_major[cnt++] = hC[j*nBrows+i];
        }
    }

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dB))
    CHECK_CUDA(cudaFree(dC))

    return cusparsecsrspmmfloat_time;
#endif
}

double evalB2SRSpmmFull32()
{
    // init C (result storage)
    cudaMalloc(&fC, nrows * nBcols * sizeof(float));
    cudaMemset(fC, 0, nrows * nBcols * sizeof(float));

    // define thread blocks & thread
    int tiles_per_tb = (1024 / nBcols);
    dim3 BLOCKS = dim3((nblockrows+tiles_per_tb-1)/tiles_per_tb);
    int THREADS = 1024;

    // ------
    GpuTimer b2sr_timer;
    b2sr_timer.Start();

    for (int i = 0; i < TEST_TIMES; i++)
    {
        spmm4_full_full_new2<<<BLOCKS, THREADS>>>(tA, fB, fC, bsrRowPtr, bsrColInd, nblockrows, nBcols);
    }

    b2sr_timer.Stop();
    double b2sr_time = b2sr_timer.ElapsedMillis() / double(TEST_TIMES);
    // ------

    return b2sr_time;
}

/// ======================
void freeCSR()
{
    // free csr mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);
}

void freeB2SR()
{
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free storage
    cudaFree(tA);

    // free vec mem
    free(B);
    cudaFree(tB);
    cudaFree(fB);
    cudaFree(fC);

    // free indexing sys
    cudaFree(bsrRowPtr);
    cudaFree(bsrColInd);
}

void freeResult()
{
    free(result_b2srspmm);
    free(result_cusparsecsrspmmfloat);
}

/// ======================
void verifyResultFull()
{
    // copy result to host for verification
    result_cusparsecsrspmmfloat = hC_row_major;
    result_b2srspmm = (float *)malloc(nrows * nBcols * sizeof(float));
    cudaMemcpy(result_b2srspmm, fC, nrows * nBcols * sizeof(float), cudaMemcpyDeviceToHost);

    // verify
    bool pass = true;
    for(int i=0; i<nrows; i++)
    {
        for(int j=0; j<nBcols; j++)
        {
            if (result_b2srspmm[i*nBcols+j] != result_cusparsecsrspmmfloat[i*nBcols+j]) 
                {pass = false;}
        }
    }

    // printf("-------------- Verify Result Sign Full --------------\n");
    printf("TEST PASSED: %d\n", pass);

    // printf check
    // printf("nrows: %d, nBcols: %d\n", nrows, nBcols);
    // printf("hostB (float) --------------\n");
    float *hostB = (float *)malloc(nrows * nBcols * sizeof(float));
    cudaMemcpy(hostB, fB, nrows * nBcols * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0; i<5; i++)
    // {
    //     for (int j=0; j<nBcols; j++)
    //     {
    //         std::cout << hostB[i * nBcols + j];
    //     }
    //     printf("\n");
    // }
    free(hostB);

    // printf("--------------B2SR SpMM--------------\n");

    // for (int i=0; i<5; i++) 
    // {
    //     for (int j=0; j<nBcols; j++)
    //     {
    //         std::cout << result_b2srspmm[i * nBcols + j] << " ";
    //     }
    //     printf("\n");
    // }

    // printf("--------------cuSPARSE SpMM--------------\n");
    // for (int i=0; i<5; i++) 
    // {
    //     for (int j=0; j<nBcols; j++)
    //     {
    //         std::cout << result_cusparsecsrspmmfloat[i * nBcols + j] << " ";
    //     }
    //     printf("\n");
    // }

    // free mem
    freeResult();
}

/// ======================
int main(int argc, char *argv[])
{
    char *Amtxfile = argv[1]; // e.g. "G43.mtx"
    nBcols = atoi(argv[2]); // e.g. "64"

    cudaSetDevice(0);

    // read A matrix as CSR
    readMtxCSR(Amtxfile);

    // randomized B matrix
    nBrows = nrows; 
    B = (float *)malloc(sizeof(float) * nBrows * nBcols);
    srand(time(0));
    for (int i = 0; i < nBrows * nBcols; i++)
    {
        float x = (float)rand() / RAND_MAX;
        B[i] = (x > 0.5) ? 1 : -1;
    }
    
    // baseline
    double spmmtime_baseline = evalCSRSpmmFloatCuSPARSE();

    // preprocess A into B2SR
    CSR2B2SR();
    packBasSign32full();

    double spmmtime_b2sr = evalB2SRSpmmFull32();

    // print result
    // printf("--------------Time --------------\n");
    printf("b2sr: %.3f, csr-float: %.3f\n", spmmtime_b2sr, spmmtime_baseline); // ms
    // printf("speedup: %.3f\n", spmmtime_baseline/spmmtime_b2sr);

    // verify result
    verifyResultFull();

    // free mem
    freeCSR();
    freeB2SR();
}
