#ifndef HISTOGRAM_H__
#define HISTOGRAM_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void gpu_histogram(unsigned int* h_out,
	unsigned int* h_in,
	unsigned int num_elems,
	unsigned int range);

#endif
