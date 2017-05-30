#ifndef HISTOGRAM_H__
#define HISTOGRAM_H__

#include <cmath>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"

#define MAX_BLOCK_SZ 1024

void gpu_histogram(unsigned int* h_out,
	unsigned int* h_in,
	unsigned int num_elems,
	unsigned int range);

#endif
