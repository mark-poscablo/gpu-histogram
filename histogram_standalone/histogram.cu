#include "histogram.h"
#include "sort.h"

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

void gpu_histogram_atomics(unsigned int* h_out,
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

__global__
void map_input_to_kv(uint2* d_out, unsigned int* d_in, unsigned int d_in_len)
{
	unsigned int glbl_t_idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (glbl_t_idx >= d_in_len)
		return;

	d_out[glbl_t_idx] = make_uint2(d_in[glbl_t_idx], 1);
}

__global__
void sum_reduce_by_key(unsigned int* d_out, unsigned int* d_in, unsigned int d_in_len, unsigned int key)
{

}

void gpu_histogram_sortreduce(unsigned int* h_out,
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

	// Map input elements 1 to 1 to new key-value pairs
	//  Key is the element and value is 1, e.g. (592, 1)
	uint2* d_kv_pairs;
	checkCudaErrors(cudaMalloc(&d_kv_pairs, sizeof(uint2) * num_elems));
	map_input_to_kv<<<grid_sz, block_sz>>>(d_kv_pairs, d_in, num_elems);

	// Sort k-v pairs by their keys
	//  Use our radix sort
	//  In a way, the histogram performance can be improved by
	//   improving the 2-way split of our radix sort
	// Since our radix_sort() isn't templated, need to change data types within radix_sort()
	uint2* d_kv_pairs_sorted;
	checkCudaErrors(cudaMalloc(&d_kv_pairs_sorted, sizeof(uint2) * num_elems));
	unsigned int* d_radix_sort_preds;
	checkCudaErrors(cudaMalloc(&d_radix_sort_preds, sizeof(unsigned int) * num_elems));
	unsigned int* d_radix_sort_scanned_preds;
	checkCudaErrors(cudaMalloc(&d_radix_sort_scanned_preds, sizeof(unsigned int) * num_elems));
	radix_sort(d_kv_pairs_sorted, d_kv_pairs, d_radix_sort_preds, d_radix_sort_scanned_preds, num_elems);

	// Reduce k-v pairs by their values
	//  Use streams to launch multiple reductions
	//  How do you tell the different keys apart though..?
	cudaStream_t* streams = new cudaStream_t[num_bins];
	for (int i = 0; i < num_bins; ++i)
	{
		// Idea: instead of using k-v pairs, keep using unsigned ints
		// During reduction, divide the element by itself to get 1s - or just somehow map them to 1s
		// Use shared memory for the reduction
		cudaStreamCreate(&streams[i]);
		sum_reduce_by_key<<<grid_sz, block_sz, 0, streams[i]>>>(&d_out[i], d_in, num_elems, i);
	}
	delete[] streams;

	checkCudaErrors(cudaMemcpy(h_out, d_out, sizeof(unsigned int) * num_bins, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_kv_pairs_sorted));
	checkCudaErrors(cudaFree(d_kv_pairs));
	checkCudaErrors(cudaFree(d_out));
	checkCudaErrors(cudaFree(d_in));
}