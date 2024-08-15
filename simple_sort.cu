// Discente: Gustavo Henrique Ferreira Cruz
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <stdio.h>
#include "bitonic_sort.h"

#define THREADS_PER_BLOCK 1024
#define MAX_RAND 10000

__global__ void partition_insert(uint *vertical_scan, uint *global_scan, uint n_histograms, uint *input, uint *output, uint n_elements, uint smallest, uint histogram_factor)
{
  extern __shared__ uint hist_counter[];

  uint thread_id = threadIdx.x;
  uint block_id = blockIdx.x;

  uint curr_line_offset_mult = block_id * n_histograms;
  uint curr_input_offset_mult = block_id * blockDim.x;

  if (thread_id < n_histograms)
    hist_counter[thread_id] = vertical_scan[thread_id + curr_line_offset_mult] + global_scan[thread_id];

  __syncthreads();

  if (thread_id < n_elements)
  {
    uint input_value = input[thread_id + curr_input_offset_mult];
    uint interval_to_insert = min((input_value - smallest) / histogram_factor, n_histograms - 1);
    uint position_to_insert = atomicAdd(&hist_counter[interval_to_insert], 1);

    output[position_to_insert] = input_value;
  }

  __syncthreads();
}

__global__ void vertical_scan(uint *line_histogram, uint *vertical_scan, uint n_histograms, uint line_amount)
{
  extern __shared__ uint scan[];

  uint thread_id = threadIdx.x;
  uint col = blockIdx.x;
  uint line = n_histograms;

  if (thread_id == 0)
    scan[col + (line * thread_id)] = 0;

  if (thread_id < line_amount)
  {
    scan[col + (line * (thread_id + 1))] = line_histogram[col + (line * thread_id)];
    __syncthreads();

    for (uint stride = 1; stride < line_amount; stride *= 2)
    {
      uint index = (thread_id + 1) * stride * 2 - 1;
      if (index < line_amount)
        scan[col + (line * index)] += scan[col + (line * (index - stride))];

      __syncthreads();
    }

    for (uint stride = blockDim.x; stride > 0; stride /= 2)
    {
      __syncthreads();
      uint index = (thread_id + 1) * stride * 2 - 1;
      if (index + stride < line_amount)
        scan[col + (line * (index + stride))] += scan[col + (line * index)];
    }
    __syncthreads();

    vertical_scan[col + (line * thread_id)] = scan[col + (line * thread_id)];
    __syncthreads();
  }
}

__global__ void prefix_sum(uint *global_histogram, uint *global_histogram_scan, uint n_histograms)
{
  extern __shared__ uint scan[];

  uint thread_id = threadIdx.x;

  if (thread_id == 0)
    scan[thread_id] = 0;

  if (thread_id < n_histograms)
  {
    scan[thread_id + 1] = global_histogram[thread_id];

    __syncthreads();

    for (uint stride = 1; stride < n_histograms; stride *= 2)
    {
      uint index = (thread_id + 1) * stride * 2 - 1;
      if (index < n_histograms)
        scan[index] += scan[index - stride];

      __syncthreads();
    }

    for (uint stride = blockDim.x; stride > 0; stride /= 2)
    {
      __syncthreads();
      uint index = (thread_id + 1) * stride * 2 - 1;
      if (index + stride < n_histograms)
        scan[index + stride] += scan[index];
    }
    __syncthreads();

    global_histogram_scan[thread_id] = scan[thread_id];
    __syncthreads();
  }
}

__global__ void block_histogram(uint *input, uint n_elements, uint *global_histogram, uint *line_histogram, uint histogram_factor, uint n_histograms, uint smallest)
{
  extern __shared__ uint local_histogram[];

  uint thread_id = threadIdx.x;
  uint block_id = blockIdx.x;
  uint input_index = thread_id + block_id * blockDim.x;

  if (thread_id == 0)
    for (uint i = 0; i < n_histograms; i++)
      local_histogram[i] = 0;

  __syncthreads();

  if (input_index < n_elements)
  {
    uint histogram_position = min((input[input_index] - smallest) / histogram_factor, n_histograms - 1);

    atomicAdd(&global_histogram[histogram_position], 1);
    atomicAdd(&local_histogram[histogram_position], 1);
  }

  __syncthreads();

  if (thread_id == 0)
    memcpy(line_histogram + (block_id * n_histograms), local_histogram, n_histograms * sizeof(uint));
}

void initialize_input_vector(uint *input, unsigned short n_elements, uint *smallest, uint *biggest)
{
  *smallest = std::numeric_limits<uint>::max();
  *biggest = std::numeric_limits<uint>::min();

  for (unsigned short i = 0; i < n_elements; i++)
  {
    int a = rand() % (MAX_RAND + 1);
    uint v = static_cast<uint>(a);

    if (v > *biggest)
      *biggest = v;

    if (v < *smallest)
      *smallest = v;

    input[i] = v;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    printf("Correct usage: ./simple_sort <number_elements> <number_histograms> <number_repetitions>\n");
    return -1;
  }

  uint n_elements = std::atoi(argv[1]);
  uint n_histograms = std::atoi(argv[2]);
  uint n_repetitions = std::atoi(argv[3]);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  uint histogram_size = n_histograms * sizeof(uint);

  uint input_size = n_elements * sizeof(uint);

  uint potency = 2;
  while (potency < n_elements)
    potency *= 2;

  uint max_size_aux = max(potency, 1024U);

  uint *d_input, *h_input = (uint *)malloc(input_size), *d_output, *h_output = (uint *)malloc(input_size), smallest, biggest, *d_global_histogram, *d_line_histogram, *d_global_histogram_scan, *h_global_histogram_scan = (uint *)malloc(histogram_size), *d_vertical_scan, *h_aux = (uint *)malloc(max_size_aux * sizeof(uint)), *d_aux;

  uint blocks_per_grid = (n_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  initialize_input_vector(h_input, n_elements, &smallest, &biggest);

  uint histogram_factor = (biggest - smallest) / n_histograms;
  if (histogram_factor == 0)
    histogram_factor = 1;

  // ------------------------------ Prints
  printf("Biggest Number:%u\n", biggest);
  printf("Smallest Number:%u\n", smallest);
  printf("Histogram Value Range Width:%u\n", histogram_factor);
  // ------------------------------ Prints

  cudaEventRecord(start);
  for (uint r = 0; r < n_repetitions; r++)
  {
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, input_size);
    cudaMalloc(&d_global_histogram, histogram_size);
    cudaMemset(d_global_histogram, 0, histogram_size);
    cudaMalloc(&d_global_histogram_scan, histogram_size);

    cudaMalloc(&d_line_histogram, blocks_per_grid * histogram_size);
    cudaMemset(d_line_histogram, 0, blocks_per_grid * histogram_size);
    cudaMalloc(&d_vertical_scan, blocks_per_grid * histogram_size);
    cudaMalloc(&d_aux, max_size_aux * sizeof(uint));

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);
    block_histogram<<<blocks_per_grid, THREADS_PER_BLOCK, histogram_size>>>(d_input, n_elements, d_global_histogram, d_line_histogram, histogram_factor, n_histograms, smallest);
    cudaDeviceSynchronize();

    prefix_sum<<<1, THREADS_PER_BLOCK, histogram_size>>>(d_global_histogram, d_global_histogram_scan, n_histograms);
    cudaMemcpy(h_global_histogram_scan, d_global_histogram_scan, histogram_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    vertical_scan<<<n_histograms, THREADS_PER_BLOCK, blocks_per_grid * histogram_size>>>(d_line_histogram, d_vertical_scan, n_histograms, blocks_per_grid);
    cudaDeviceSynchronize();

    partition_insert<<<blocks_per_grid, THREADS_PER_BLOCK, n_histograms>>>(d_vertical_scan, d_global_histogram_scan, n_histograms, d_input, d_output, n_elements, smallest, histogram_factor);
    cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (uint i = 0; i < n_histograms; i++)
    {
      uint start_index = h_global_histogram_scan[i];
      uint end_index = n_elements;
      if (i + 1 < n_histograms)
        end_index = h_global_histogram_scan[i + 1];

      uint k = 0;
      for (uint j = start_index; j < end_index; j++)
        h_aux[k++] = h_output[j];

      for (uint j = end_index - start_index; j < max_size_aux; j++)
        h_aux[j] = MAX_RAND + 1;

      cudaMemcpy(d_aux, h_aux, max_size_aux * sizeof(uint), cudaMemcpyHostToDevice);
      bitonicSort(d_aux, d_aux, d_aux, d_aux, 1, max_size_aux, 1);
      cudaMemcpy(h_aux, d_aux, max_size_aux * sizeof(uint), cudaMemcpyDeviceToHost);

      k = 0;
      for (uint j = start_index; j < end_index; j++)
        h_output[j] = h_aux[k++];
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_global_histogram);
    cudaFree(d_global_histogram_scan);
    cudaFree(d_line_histogram);
    cudaFree(d_vertical_scan);
    cudaFree(d_aux);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float simple_sort_milliseconds = 0;
  cudaEventElapsedTime(&simple_sort_milliseconds, start, stop);

  cudaEventRecord(start);

  for (uint r = 0; r < n_repetitions; r++)
  {
    thrust::device_vector<uint> d_vec(h_input, h_input + n_elements);
    thrust::sort(d_vec.begin(), d_vec.end());
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float thrust_milliseconds = 0;
  cudaEventElapsedTime(&thrust_milliseconds, start, stop);

  double ops = static_cast<double>(n_elements) / ((simple_sort_milliseconds / n_repetitions) / 1000);

  printf("Simple Sort Time: %.2lfms\n", (simple_sort_milliseconds / n_repetitions));
  printf("Throughput: %.2lf\n", ops);

  printf("Thrust Sort Time: %.2lfms\n", (thrust_milliseconds / n_repetitions));

  free(h_input);
  free(h_output);
  free(h_global_histogram_scan);
  free(h_aux);

  return 0;
}