#include <stdio.h>
#include <iostream>
#include <cstring>

#include "histogram.h"

void cpu_histogram(unsigned int* h_out, 
					unsigned int* h_in, 
					unsigned int num_elems, 
					unsigned int num_bins)
{
	memset(h_out, 0, num_bins * sizeof(unsigned int));

	for (int i = 0; i < num_elems; ++i)
	{
		h_out[h_in[i]]++;
	}
}

unsigned int* generate_input(unsigned int range, unsigned int num_elems)
{
	unsigned int* h_out = new unsigned int[num_elems];

	unsigned int curr_idx = 0;

	for (int i = 0; i < range; ++i)
	{
		for (int j = 0; j <= i; ++j)
		{
			h_out[curr_idx] = i;
			curr_idx++;
		}
	}

	return h_out;
}

int main()
{
	// Set up clock for timing comparisons
	srand(time(NULL));
	std::clock_t start;
	double duration;

	unsigned int range = 1024;
	unsigned int num_elems = (range * (range + 1)) / 2;
	unsigned int* input = generate_input(range, num_elems);

	//for (unsigned int i = 0; i < num_elems; ++i)
	//{
	//	std::cout << input[i] << " ";
	//}
	//std::cout << std::endl;

	unsigned int* cpu_output = new unsigned int[range];
	start  = std::clock();
	cpu_histogram(cpu_output, input, num_elems, range);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "CPU time: " << duration << std::endl;
	//for (unsigned int i = 0; i < range; ++i)
	//{
	//	std::cout << cpu_output[i] << " ";
	//}
	//std::cout << std::endl;

	unsigned int* gpu_output = new unsigned int[range];
	start = std::clock();
	gpu_histogram_atomics(gpu_output, input, num_elems, range);
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	std::cout << "GPU time: " << duration << std::endl;
	//for (unsigned int i = 0; i < range; ++i)
	//{
	//	std::cout << gpu_output[i] << " ";
	//}
	//std::cout << std::endl;

	// Check for any mismatches between outputs of CPU and GPU
	bool match = true;
	int index_diff = 0;
	for (int i = 0; i < range; ++i)
	{
		if (cpu_output[i] != gpu_output[i])
		{
			match = false;
			index_diff = i;
			break;
		}
	}
	std::cout << "Match: " << match << std::endl;

	// Detail the mismatch if any
	if (!match)
	{
		std::cout << "Difference in index: " << index_diff << std::endl;
		std::cout << "CPU: " << cpu_output[index_diff] << std::endl;
		std::cout << "GPU: " << gpu_output[index_diff] << std::endl;
		int window_sz = 10;

		std::cout << "Contents: " << std::endl;
		std::cout << "CPU: ";
		for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
		{
			std::cout << cpu_output[index_diff + i] << ", ";
		}
		std::cout << std::endl;
		std::cout << "GPU: ";
		for (int i = -(window_sz / 2); i < (window_sz / 2); ++i)
		{
			std::cout << gpu_output[index_diff + i] << ", ";
		}
		std::cout << std::endl;
	}

	if (gpu_output != NULL)
		delete[] gpu_output;
	if (cpu_output != NULL)
		delete[] cpu_output;
	if (input != NULL)
		delete[] input;
}
