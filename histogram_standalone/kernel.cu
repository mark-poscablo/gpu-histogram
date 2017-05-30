
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstring>

void cpu_histogram(unsigned int* h_out, 
					unsigned int* h_in, 
					unsigned int num_elems, 
					unsigned int range)
{
	memset(h_out, 0, range * sizeof(unsigned int));

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
	unsigned int range = 8;
	unsigned int num_elems = (range * (range + 1)) / 2;
	unsigned int* input = generate_input(range, num_elems);

	for (unsigned int i = 0; i < num_elems; ++i)
	{
		std::cout << input[i] << " ";
	}
	std::cout << std::endl;

	unsigned int* output = new unsigned int[range];
	cpu_histogram(output, input, num_elems, range);
	for (unsigned int i = 0; i < range; ++i)
	{
		std::cout << output[i] << " ";
	}
	std::cout << std::endl;

	if (output != NULL)
		delete[] output;
	if (input != NULL)
		delete[] input;
}
