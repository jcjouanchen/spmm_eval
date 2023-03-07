// #include "bsrbmv.cu"
#include "spmm.cu"
#include "utility.cu"


/**
* batch the process of csr2bsr, blocksize = 2
* assume csr val are only 0 or 1
* each value store at the lower 2 bit
*/
void csr2bsr_batch_2(const int *h_csrRowPtr, const int *h_csrColInd,
                     const int nrows, const int ncols, const int nnz,
                     int *bsrRowPtr, int *bsrColInd, uchar *bsrVal,
                     const int blocksize, const int nblockrows, const int nblocks)
{
    int *stats;
    cudaMalloc((void **)&stats, sizeof(int)*4);
    setDeviceValArr<int, int><<<1, 1>>>(stats, 4, 0);
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n");
    printHostIndArr<int>(h_csrRowPtr, (nrows + 1));
    printf("h_csrColInd:\n");
    printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1, 1>>>(bsrRowPtr, (nblockrows + 1), 0);
    setDeviceIndArr<int><<<1, 1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, uchar><<<1, 1>>>(bsrVal, nblocks * blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    uchar *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 2 < nrows ? h_csrRowPtr[(i + 1) * 2] : nnz), temp_rowstart = h_csrRowPtr[i * 2];
        int temp_nnz = temp_rowend - temp_rowstart;
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (2 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 2, sizeof(int) * ((nrows + 1) - (i * 2)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 2)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 2)), (2 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 2, sizeof(int) * (2 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (2 + 1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
            printf("temp_csrrowptr: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (2 + 1));
#endif

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
            printf("temp_csrcolind: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrcolind, temp_nnz);
#endif

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1, 1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                               temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                               temp_bsrrowptr, &temp_nblocks));
            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
#ifdef VERBOSE
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
#endif
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(uchar) * ceil((float)temp_nblocks / 64) * 64 * blocksize);
            // ToBit2Col<float><<<dim3(1, ceil((float)temp_nblocks / 64)), 32>>>(temp_bsrval, temp_bsrval_packed, temp_nblocks);

            // printTempBSRVal<<<1,1>>>(temp_bsrval, blocksize, temp_nblocks);
            // printGlobalBSRBlock2<<<1,1>>>(temp_bsrval_packed, blocksize, temp_nblocks);
                    
            countBlockStats2<<<1,1>>>(stats, temp_bsrval, blocksize, temp_nblocks);
            // printDeviceIndArr<int><<<1, 1>>>(stats, 4);
            // int k;
            // std::cin >> k;

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind + temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind + temp_nblocks);
#endif
            cudaMemcpy(bsrColInd + last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + last_bsrrowind * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
        else
        { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR8<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k;
        std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
#ifdef VERBOSE
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else
        printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);
#endif

#ifdef DEBUG
    //printout global bsr to verify
    printGlobalBSR8<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock8<<<1, 1>>>(bsrVal, blocksize, nblocks);
#endif
    // printf("stats: \n");
    printDeviceIndArr<int><<<1, 1>>>(stats, 4);
    cudaFree(stats);
}

/**
* batch the process of csr2bsr, blocksize = 4
* assume csr val are only 0 or 1
* each value store at the lower 4 bit
*/
void csr2bsr_batch_4(const int *h_csrRowPtr, const int *h_csrColInd,
                     const int nrows, const int ncols, const int nnz,
                     int *bsrRowPtr, int *bsrColInd, uchar *bsrVal,
                     const int blocksize, const int nblockrows, const int nblocks)
{
    // int *stats;
    // cudaMalloc((void **)&stats, sizeof(int)*16);
    // setDeviceValArr<int, int><<<1, 1>>>(stats, 16, 0);
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n");
    printHostIndArr<int>(h_csrRowPtr, (nrows + 1));
    printf("h_csrColInd:\n");
    printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1, 1>>>(bsrRowPtr, (nblockrows + 1), 0);
    setDeviceIndArr<int><<<1, 1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, uchar><<<1, 1>>>(bsrVal, nblocks * blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    uchar *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 4 < nrows ? h_csrRowPtr[(i + 1) * 4] : nnz), temp_rowstart = h_csrRowPtr[i * 4];
        int temp_nnz = temp_rowend - temp_rowstart;
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (4 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 4, sizeof(int) * ((nrows + 1) - (i * 4)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 4)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 4)), (4 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 4, sizeof(int) * (4 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (4 + 1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
            printf("temp_csrrowptr: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (4 + 1));
#endif

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
            printf("temp_csrcolind: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrcolind, temp_nnz);
#endif

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1, 1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                               temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                               temp_bsrrowptr, &temp_nblocks));
            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
#ifdef VERBOSE
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
#endif
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(uchar) * ceil((float)temp_nblocks / 64) * 64 * blocksize);
            ToBit4Col<float><<<dim3(1, ceil((float)temp_nblocks / 64)), 32>>>(temp_bsrval, temp_bsrval_packed, temp_nblocks);
            // <-- padding is not a problem here since we did not copy them later

            // each of the block will use 2x location to store
                    //    printTempBSRVal<<<1,1>>>(temp_bsrval, blocksize, temp_nblocks);
                    //    printGlobalBSRBlock4<<<1,1>>>(temp_bsrval_packed, blocksize, temp_nblocks);
                       
            // countBlockStats<<<1,1>>>(stats, temp_bsrval_packed, blocksize, temp_nblocks);
                    //    int k;
                    //    std::cin >> k;

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind + temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind + temp_nblocks);
#endif
            cudaMemcpy(bsrColInd + last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + last_bsrrowind * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
        else
        { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR8<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k;
        std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
#ifdef VERBOSE
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else
        printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);
#endif

#ifdef DEBUG
    //printout global bsr to verify
    printGlobalBSR8<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock8<<<1, 1>>>(bsrVal, blocksize, nblocks);
#endif
    // printf("stats: \n");
    // printDeviceIndArr<int><<<1, 1>>>(stats, 16);
    // cudaFree(stats);
}

/**
* batch the process of csr2bsr, blocksize = 8
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_8(const int *h_csrRowPtr, const int *h_csrColInd,
                     const int nrows, const int ncols, const int nnz,
                     int *bsrRowPtr, int *bsrColInd, uchar *bsrVal,
                     const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n");
    printHostIndArr<int>(h_csrRowPtr, (nrows + 1));
    printf("h_csrColInd:\n");
    printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1, 1>>>(bsrRowPtr, (nblockrows + 1), 0);
    setDeviceIndArr<int><<<1, 1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, uchar><<<1, 1>>>(bsrVal, nblocks * blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    uchar *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 8 < nrows ? h_csrRowPtr[(i + 1) * 8] : nnz), temp_rowstart = h_csrRowPtr[i * 8];
        int temp_nnz = temp_rowend - temp_rowstart;
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (8 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 8, sizeof(int) * ((nrows + 1) - (i * 8)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 8)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 8)), (8 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 8, sizeof(int) * (8 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (8 + 1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
            printf("temp_csrrowptr: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (8 + 1));
#endif

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
            printf("temp_csrcolind: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrcolind, temp_nnz);
#endif

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1, 1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                               temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                               temp_bsrrowptr, &temp_nblocks));
            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
#ifdef VERBOSE
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
#endif
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(uchar) * ceil((float)temp_nblocks / 16) * 16 * blocksize);
            ToBit8Col<float><<<dim3(1, ceil((float)temp_nblocks / 16)), 32>>>(temp_bsrval, temp_bsrval_packed, temp_nblocks);
            // <-- padding is not a problem here since we did not copy them later

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind + temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind + temp_nblocks);
#endif
            cudaMemcpy(bsrColInd + last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + last_bsrrowind * blocksize, temp_bsrval_packed, sizeof(uchar) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
        else
        { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR8<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k;
        std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
#ifdef VERBOSE
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else
        printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);
#endif

#ifdef DEBUG
    //printout global bsr to verify
    printGlobalBSR8<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock8<<<1, 1>>>(bsrVal, blocksize, nblocks);
#endif
}

/**
* size estimation before
* convert 8-bin packed bsrval to aligned unsigned blocks
* every 8x8x4 uchar -> 8x1 unsigned
*/
__global__ void countUcharUnsignedSize(const int *bsrRowPtr, const int nblockrows, int *count)
{

    int cnt = 0, temp_nblocks = 0;
    for (int row = 0; row < nblockrows; row++)
    {
        temp_nblocks = bsrRowPtr[row + 1] - bsrRowPtr[row];
        cnt += (int)ceil((float)temp_nblocks / 4) * 8;
    }

    count[0] = cnt;

#ifdef DEBUG
    printf("[countUnsignedSize] result total blocks = %d (8 * %d)\n", cnt, cnt / 8);
#endif
}

/**
* convert 8-bin packed bsrval to aligned unsigned blocks
* every 8x8x4 uchar -> 8x1 unsigned
* also convert bsrrowptr, bsrcolind
*/
__global__ void packUchar2AlignedUnsigned(const int *bsrRowPtr, const int *bsrColInd, uchar *bsrval_packed, const int nblockrows,
                                          int *new_bsrrowptr, int *new_bsrcolind, unsigned *new_bsrval_packed)
{
    // [uchar0 uchar1 ... uchar7] , [uchar8 uchar9 ... uchar15] , [uchar16 uchar17 ... uchar23] ...
    // become [(uchar0, uchar8, uchar16, uchar24) => unsigned, ...]
    // every 8x8x4 uchar -> 8x1 unsigned
    new_bsrrowptr[0] = 0;

    int cnt = 0, temp_nblocks = 0, row_start = 0;
    int colcnt = 0;
    for (int row = 0; row < nblockrows; row++)
    {
        temp_nblocks = bsrRowPtr[row + 1] - bsrRowPtr[row];
        row_start = bsrRowPtr[row];
        for (int i = 0; i < (int)ceil((float)temp_nblocks / 4) * 4; i += 4)
        {
            for (int j = 0; j < 8; j++)
            {
                uchar a0 = i * 8 + j < temp_nblocks * 8 ? bsrval_packed[row_start * 8 + i * 8 + j] : 0;
                uchar a1 = i * 8 + 8 + j < temp_nblocks * 8 ? bsrval_packed[row_start * 8 + i * 8 + 8 + j] : 0;
                uchar a2 = i * 8 + 16 + j < temp_nblocks * 8 ? bsrval_packed[row_start * 8 + i * 8 + 16 + j] : 0;
                uchar a3 = i * 8 + 24 + j < temp_nblocks * 8 ? bsrval_packed[row_start * 8 + i * 8 + 24 + j] : 0;
                unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;
                new_bsrval_packed[cnt++] = r0;
            }
        }

        for (int i = bsrRowPtr[row]; i < bsrRowPtr[row] + temp_nblocks; i++)
            new_bsrcolind[colcnt++] = bsrColInd[i];
        for (int i = 0; i < (int)ceil((float)temp_nblocks / 4) * 4 - temp_nblocks; i++)
            new_bsrcolind[colcnt++] = 0;

        new_bsrrowptr[row + 1] = new_bsrrowptr[row] + (int)ceil((float)temp_nblocks / 4) * 4;
        //        printf("row:%d, bsrRowPtr[row]:%d, bsrRowPtr[row+1]:%d\n", row, bsrRowPtr[row], bsrRowPtr[row+1]);
        //        printf("row:%d, new_bsrrowptr[row]:%d, new_bsrrowptr[row+1]:%d\n", row, new_bsrrowptr[row], new_bsrrowptr[row+1]);
    }

#ifdef DEBUG
    printf("[PackUchar2AlignedUnsigned] result total unsigned = %d (8 * %d), colind size = %d (4 * %d)\n", cnt, cnt / 8, colcnt, colcnt / 4);
#endif
}

/**
* batch the process of csr2bsr, blocksize = 16
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_16(const int *h_csrRowPtr, const int *h_csrColInd,
                      const int nrows, const int ncols, const int nnz,
                      int *bsrRowPtr, int *bsrColInd, ushort *bsrVal,
                      const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n");
    printHostIndArr<int>(h_csrRowPtr, (nrows + 1));
    printf("h_csrColInd:\n");
    printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1, 1>>>(bsrRowPtr, (nblockrows + 1), 0);
    setDeviceIndArr<int><<<1, 1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, ushort><<<1, 1>>>(bsrVal, nblocks * blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    ushort *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 16 < nrows ? h_csrRowPtr[(i + 1) * 16] : nnz), temp_rowstart = h_csrRowPtr[i * 16];
        int temp_nnz = temp_rowend - temp_rowstart;
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (16 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 16, sizeof(int) * ((nrows + 1) - (i * 16)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 16)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 16)), (16 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 16, sizeof(int) * (16 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (16 + 1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
            printf("temp_csrrowptr: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (16 + 1));
#endif

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
            printf("temp_csrcolind: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrcolind, temp_nnz);
#endif

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1, 1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                               temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                               temp_bsrrowptr, &temp_nblocks));
            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
#ifdef VERBOSE
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
#endif
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(ushort) * ceil((float)temp_nblocks / 4) * 4 * blocksize);
            ToBit16Col<float><<<dim3(1, ceil((float)temp_nblocks / 4)), 32>>>(temp_bsrval, temp_bsrval_packed, temp_nblocks);
            // <-- padding is not a problem here since we did not copy them later

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind + temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind + temp_nblocks);
#endif
            cudaMemcpy(bsrColInd + last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + last_bsrrowind * blocksize, temp_bsrval_packed, sizeof(ushort) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
        else
        { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR16<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k;
        std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
#ifdef VERBOSE
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else
        printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);
#endif

#ifdef DEBUG
    //printout global bsr to verify
    printGlobalBSR16<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock16<<<1, 1>>>(bsrVal, blocksize, nblocks);
#endif
}

/**
* size estimation before
* convert 16-bin packed bsrval to aligned unsigned blocks
* every 16x16x2 ushort -> 16x1 unsigned
*/
__global__ void countUshortUnsignedSize(const int *bsrRowPtr, const int nblockrows, int *count)
{

    int cnt = 0, temp_nblocks = 0;
    for (int row = 0; row < nblockrows; row++)
    {
        temp_nblocks = bsrRowPtr[row + 1] - bsrRowPtr[row];
        cnt += (int)ceil((float)temp_nblocks / 2) * 16;
    }

    count[0] = cnt;

#ifdef DEBUG
    printf("[countUnsignedSize] result total blocks = %d (16 * %d)\n", cnt, cnt / 16);
#endif
}

/**
* convert 16-bin packed bsrval to aligned unsigned blocks
* every 16x16x2 ushort -> 16x1 unsigned
* also convert bsrrowptr, bsrcolind
*/
__global__ void packUshort2AlignedUnsigned(const int *bsrRowPtr, const int *bsrColInd, ushort *bsrval_packed, const int nblockrows,
                                           int *new_bsrrowptr, int *new_bsrcolind, unsigned *new_bsrval_packed)
{
    new_bsrrowptr[0] = 0;

    int cnt = 0, temp_nblocks = 0, row_start = 0;
    int colcnt = 0;
    for (int row = 0; row < nblockrows; row++)
    {
        temp_nblocks = bsrRowPtr[row + 1] - bsrRowPtr[row];
        row_start = bsrRowPtr[row];
        for (int i = 0; i < (int)ceil((float)temp_nblocks / 2) * 2; i += 2)
        {
            for (int j = 0; j < 16; j++)
            {
                ushort a0 = i * 16 + j < temp_nblocks * 16 ? bsrval_packed[row_start * 16 + i * 16 + j] : 0;
                ushort a1 = i * 16 + 16 + j < temp_nblocks * 16 ? bsrval_packed[row_start * 16 + i * 16 + 16 + j] : 0;
                unsigned r0 = a0 << 16 | a1;
                new_bsrval_packed[cnt++] = r0;
            }
        }

        for (int i = bsrRowPtr[row]; i < bsrRowPtr[row] + temp_nblocks; i++)
            new_bsrcolind[colcnt++] = bsrColInd[i];
        for (int i = 0; i < (int)ceil((float)temp_nblocks / 2) * 2 - temp_nblocks; i++)
            new_bsrcolind[colcnt++] = 0;

        new_bsrrowptr[row + 1] = new_bsrrowptr[row] + (int)ceil((float)temp_nblocks / 2) * 2;
    }
}

/**
* batch the process of csr2bsr, blocksize = 32
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_32(const int *h_csrRowPtr, const int *h_csrColInd,
                      const int nrows, const int ncols, const int nnz,
                      int *bsrRowPtr, int *bsrColInd, unsigned *bsrVal,
                      const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n");
    printHostIndArr<int>(h_csrRowPtr, (nrows + 1));
    printf("h_csrColInd:\n");
    printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1, 1>>>(bsrRowPtr, (nblockrows + 1), 0);
    setDeviceIndArr<int><<<1, 1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, unsigned><<<1, 1>>>(bsrVal, nblocks * blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    unsigned *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 32 < nrows ? h_csrRowPtr[(i + 1) * 32] : nnz), temp_rowstart = h_csrRowPtr[i * 32];
        int temp_nnz = temp_rowend - temp_rowstart;
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (32 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 32, sizeof(int) * ((nrows + 1) - (i * 32)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 32)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 32)), (32 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 32, sizeof(int) * (32 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (32 + 1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
            printf("temp_csrrowptr: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (32 + 1));
#endif

            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
            printf("temp_csrcolind: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrcolind, temp_nnz);
#endif

            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1, 1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                               temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                               temp_bsrrowptr, &temp_nblocks));
            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
#ifdef VERBOSE
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
#endif
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(unsigned) * temp_nblocks * blocksize);
            ToBit32Col<float><<<dim3(1, temp_nblocks), 32>>>(temp_bsrval,
                                                             temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind + temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind + temp_nblocks);
#endif
            cudaMemcpy(bsrColInd + last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + last_bsrrowind * blocksize, temp_bsrval_packed, sizeof(unsigned) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
        else
        { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR32<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k;
        std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
#ifdef VERBOSE
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else
        printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);
#endif

#ifdef DEBUG
    //printout global bsr to verify
    printGlobalBSR32<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock32<<<1, 1>>>(bsrVal, blocksize, nblocks);
#endif
}

/**
* batch the process of csr2bsr, blocksize = 64
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_64(const int *h_csrRowPtr, const int *h_csrColInd,
                      const int nrows, const int ncols, const int nnz,
                      int *bsrRowPtr, int *bsrColInd, ullong *bsrVal,
                      const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n");
    printHostIndArr<int>(h_csrRowPtr, (nrows + 1));
    printf("h_csrColInd:\n");
    printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1, 1>>>(bsrRowPtr, (nblockrows + 1), 0);
    setDeviceIndArr<int><<<1, 1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, ullong><<<1, 1>>>(bsrVal, nblocks * blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    ullong *temp_bsrval_packed;

    for (int i = 0; i < nblockrows; i++)
    { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i + 1) * 64 < nrows ? h_csrRowPtr[(i + 1) * 64] : nnz), temp_rowstart = h_csrRowPtr[i * 64];
        int temp_nnz = temp_rowend - temp_rowstart;
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&csr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(csr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(csr_descr, CUSPARSE_INDEX_BASE_ZERO));
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE(cusparseCreateMatDescr(&bsr_descr));
            CHECK_CUSPARSE(cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CHECK_CUSPARSE(cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO));

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE(cusparseCreate(&handle));
            CHECK_CUSPARSE(cusparseSetStream(handle, streamId));

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void **)&temp_csrrowptr, sizeof(int) * (64 + 1));
            if (i == nblockrows - 1)
            { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 64, sizeof(int) * ((nrows + 1) - (i * 64)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 64)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, ((nrows + 1) - (i * 64)), (64 + 1), temp_nnz);
            }
            else
            { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr + i * 64, sizeof(int) * (64 + 1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (64 + 1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
            printf("temp_csrrowptr: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrrowptr, (64 + 1));
#endif
            // 2) set buffer csr colind
            cudaMalloc((void **)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd + temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
            printf("temp_csrcolind: \n");
            printDeviceIndArr<int><<<1, 1>>>(temp_csrcolind, temp_nnz);
#endif
            // 3) set buffer csr val
            cudaMalloc((void **)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1, 1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void **)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                               temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                               temp_bsrrowptr, &temp_nblocks));
            cudaMalloc((void **)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void **)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE(cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind));
#ifdef VERBOSE
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
#endif
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
#ifdef DEBUG
            printTempBSRVal<<<1, 1>>>(temp_bsrval, blocksize, temp_nblocks);
#endif
            cudaMalloc((void **)&temp_bsrval_packed, sizeof(ullong) * temp_nblocks * blocksize);
            ToBit64Col<float><<<dim3(2, temp_nblocks), 32>>>(temp_bsrval,
                                                             temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind + temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind + temp_nblocks);
#endif
            cudaMemcpy(bsrColInd + last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal + last_bsrrowind * blocksize, temp_bsrval_packed, sizeof(ullong) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr);
            temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind);
            temp_csrcolind = NULL;
            cudaFree(temp_csrval);
            temp_csrval = NULL;
            cudaFree(temp_bsrrowptr);
            temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind);
            temp_bsrcolind = NULL;
            cudaFree(temp_bsrval);
            temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed);
            temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE(cusparseDestroyMatDescr(csr_descr));
            CHECK_CUSPARSE(cusparseDestroyMatDescr(bsr_descr));
            CHECK_CUSPARSE(cusparseDestroy(handle));
        }
        else
        { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr + i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1, 1>>>(bsrRowPtr, (i + 1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i + 1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR64<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k;
        std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
#ifdef VERBOSE
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else
        printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);
#endif

#ifdef DEBUG
    // printout global bsr to verify
    printGlobalBSR64<<<1, 1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock64<<<1, 1>>>(bsrVal, blocksize, nblocks);
#endif DEBUG
}