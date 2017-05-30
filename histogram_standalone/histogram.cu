#include "histogram.h"

__global__
void histogram_atomics(unsigned int* d_out,
						unsigned int* d_in,
						unsigned int d_in_len)
{
	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (glbl_t_idx >= d_in_len)
		return;

	atomicAdd(&(d_out[d_in[glbl_t_idx]]), 1);
}

void gpu_histogram(unsigned int* h_out,
	unsigned int* h_in,
	unsigned int num_elems,
	unsigned int num_bins)
{
	unsigned int block_sz = MAX_BLOCK_SZ;
	unsigned int grid_sz = (unsigned int)std::ceil((float)num_elems / (float)block_sz);

	unsigned int* d_in;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(unsigned int) * num_elems));
	checkCudaErrors(cudaMemcpy(d_in, h_in, sizeof(unsigned int) * num_elems, cudaMemcpyHostToDevice));
	
	unsigned int* d_out;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * num_bins));
	histogram_atomics<<<grid_sz, block_sz>>>(d_out, d_in, num_elems);
	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(unsigned int) * num_bins, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_in));
}
