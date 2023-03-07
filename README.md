# spmm_eval

* CUDA 11.3
* The baseline in the code is CuSPARSE SpMM. It is only for verification purpose, not for time comparison.
* The current FBF version is adapted from the TorchSparse implementation (current baseline): https://github.com/rusty1s/pytorch_sparse/blob/3bf43eb09a68efb684d5a09e33a2e2114dd81689/csrc/cuda/spmm_cuda.cu#L13
* All kernel implementations are located in 'spmm_eval/backend/spmm.cu'

* Test the result using `{kernel_name} {matrix.mtx} {hidden_size}` e.g. `spmm/spmmfbf data/flickr.mtx 32`
* `flickr.mtx` and `reddit.mtx` are large, may need to download from here: https://drive.google.com/drive/folders/1FLQcV8nB3-Y0tTWmOYjiW502xBAijviw?usp=sharing