CC = nvcc

FLAGS = -arch=sm_70 -O3 -std=c++11 -w
LINK = -lcublas -lcusparse -lcudart

all: spmmfbf

# spmm
spmmfbf: backend/spmm.cu backend/utility.cu backend/csr2bsr_batch_bsrbmv.cu
		$(CC) $(FLAGS) $(LINK) test_spmm_full.cu -o spmm/$@

test:
	@echo "\n==== [Fin-Fout] ====\n"
	spmm/spmmfbf data/flickr.mtx 32
	spmm/spmmfbf data/flickr.mtx 64
	spmm/spmmfbf data/flickr.mtx 128
	spmm/spmmfbf data/flickr.mtx 256

	spmm/spmmfbf data/reddit.mtx 32
	spmm/spmmfbf data/reddit.mtx 64
	spmm/spmmfbf data/reddit.mtx 128
	spmm/spmmfbf data/reddit.mtx 256

clean:
	rm -f spmm/*