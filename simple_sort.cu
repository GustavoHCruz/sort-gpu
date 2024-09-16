// Discente: Gustavo Henrique Ferreira Cruz
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <stdio.h>
#include <limits.h>

#define THREADS_PER_BLOCK 1024
#define SHARED_SIZE_LIMIT 1024U

namespace cg = cooperative_groups;

__global__ void blockAndGlobalHisto(uint *input, uint n_elements, uint *global_histogram, uint *line_histogram, uint histogram_factor, uint n_histograms, uint smallest)
{
  extern __shared__ uint local_histogram[];

  uint tid = threadIdx.x;
  uint bid = blockIdx.x;
  uint idx = tid + bid * blockDim.x;
  uint histogram_position;

  for (uint i = tid; i < n_histograms; i += blockDim.x)
    local_histogram[i] = 0;

  __syncthreads();

  if (idx < n_elements)
  {
    histogram_position = min((input[idx] - smallest) / histogram_factor, n_histograms - 1);
    atomicAdd(&local_histogram[histogram_position], 1);
  }

  __syncthreads();

  for (uint i = tid; i < n_histograms; i += blockDim.x)
  {
    line_histogram[i + bid * n_histograms] = local_histogram[i];
    atomicAdd(&global_histogram[i], local_histogram[i]);
  }
}

__global__ void globalHistoScan(uint *global_histogram, uint *global_histogram_scan, uint n_histograms)
{
  uint tid = threadIdx.x;

  if (tid == 0)
  {
    uint sum = 0;
    for (uint i = 0; i < n_histograms; i++)
    {
      global_histogram_scan[i] = sum;
      sum += global_histogram[i];
    }
  }
}

__global__ void verticalScanHH(uint *line_histogram, uint *vertical_scan, uint n_histograms, uint blocks_per_grid)
{
  uint tid = threadIdx.x;
  uint bid = blockIdx.x;
  uint idx = tid + blockDim.x * bid;

  uint sum = 0;

  if (tid + bid * blockDim.x < n_histograms)
    for (uint i = 0; i < blocks_per_grid; i++)
    {
      vertical_scan[idx + (i * n_histograms)] = sum;
      sum += line_histogram[idx + (i * n_histograms)];
    }
}

__global__ void partitionKernel(uint *vertical_scan, uint *global_scan, uint n_histograms, uint *input, uint *output, uint n_elements, uint smallest, uint histogram_factor)
{
  extern __shared__ uint hist_counter[];

  uint tid = threadIdx.x;
  uint bid = blockIdx.x;

  uint curr_line_index_mult = bid * n_histograms;
  uint curr_input_offset_mult = bid * blockDim.x;

  for (uint i = tid; i < n_histograms; i += blockDim.x)
    hist_counter[i] = vertical_scan[i + curr_line_index_mult] + global_scan[i];

  __syncthreads();

  if (tid + curr_input_offset_mult < n_elements)
  {
    uint input_value = input[tid + curr_input_offset_mult];
    uint interval_to_insert = min((input_value - smallest) / histogram_factor, n_histograms - 1);
    uint position_to_insert = atomicAdd(&hist_counter[interval_to_insert], 1);

    output[position_to_insert] = input_value;
  }

  __syncthreads();
}

__device__ inline void compareAndSwap(uint *data, uint a, uint b, uint dir)
{
  uint temp;
  if ((data[a] > data[b]) == dir)
  {
    temp = data[a];
    data[a] = data[b];
    data[b] = temp;
  }
}

__global__ void blockBitonicSort(uint *global_histogram_scan, uint *output, uint n_histograms, uint n_elements)
{
  cg::thread_block cta = cg::this_thread_block();

  uint bid = blockIdx.x;
  uint block_dim = blockDim.x;
  uint tid = threadIdx.x;

  __shared__ uint aux[SHARED_SIZE_LIMIT];

  uint start_index = global_histogram_scan[bid];
  uint end_index = n_elements;
  if (bid + 1 < n_histograms)
    end_index = global_histogram_scan[bid + 1];

  if (start_index + tid < end_index)
    aux[tid] = output[start_index + tid];
  else
    aux[tid] = UINT_MAX;

  __syncthreads();

  uint dir = 1;

  for (uint size = 2; size < block_dim; size <<= 1)
  {
    uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

    for (uint stride = size / 2; stride > 0; stride >>= 1)
    {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      if (pos + stride < block_dim)
        compareAndSwap(aux, pos, pos + stride, ddd);
    }
  }

  for (uint stride = block_dim / 2; stride > 0; stride >>= 1)
  {
    cg::sync(cta);
    uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    if (pos + stride < block_dim)
      compareAndSwap(aux, pos, pos + stride, dir);
  }

  __syncthreads();

  if (start_index + tid < end_index)
    output[start_index + tid] = aux[tid];
}

bool verify_sort(uint *a, uint *b, uint size)
{
  bool flag = true;
  for (uint i = 0; i < size; i++)
    if (a[i] != b[i])
      flag = false;

  return flag;
}

void initialize_input_vector(uint *input, uint n_elements, uint *smallest, uint *biggest)
{
  *smallest = UINT_MAX;
  *biggest = 0;

  for (uint i = 0; i < n_elements; i++)
  {
    int a = rand();
    int b = rand();

    int c = a * 20 * b;

    uint v = static_cast<uint>(c);

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

  printf("============== Starting Execution ==============\n");

  if (n_elements / n_histograms > THREADS_PER_BLOCK - 150)
    printf("WARNING: the number of elements is too high for the histograms to be divided, this may result in bitonic sorting having to deal with more elements than a block of threads (%i) can support, which may lead to an incorrect sorting result\n", THREADS_PER_BLOCK);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  uint histogram_size = n_histograms * sizeof(uint);

  uint input_size = n_elements * sizeof(uint);

  uint potency = 2;
  while (potency < n_elements)
    potency *= 2;

  uint max_size_aux = max(potency, 1024U);

  uint *d_input, *h_input = (uint *)malloc(input_size), *d_output, *h_output = (uint *)malloc(input_size), *thrust_output = (uint *)malloc(input_size), smallest, biggest, *d_global_histogram, *d_line_histogram, *d_global_histogram_scan, *h_global_histogram_scan = (uint *)malloc(histogram_size), *d_vertical_scan, *h_aux = (uint *)malloc(max_size_aux * sizeof(uint));
  float simple_sort_milliseconds = 0, thrust_milliseconds = 0;

  uint blocks_per_grid = (n_elements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  uint histograms_per_grid = (n_histograms + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  int limit_shared_memory;
  cudaDeviceGetAttribute(&limit_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  unsigned long int memory_usage = n_histograms * sizeof(uint);
  if (memory_usage > limit_shared_memory)
    printf("WARNING: The number of histogram divisions is too high, this will result in excessive shared memory usage (%lu used out of %i available) which will lead to incorrect sorting result\n", memory_usage, limit_shared_memory);

  initialize_input_vector(h_input, n_elements, &smallest, &biggest);

  uint histogram_factor = (biggest - smallest) / n_histograms;
  if (histogram_factor == 0)
    histogram_factor = 1;

  // ------------------------------ Prints
  printf("Biggest Number:%u\n", biggest);
  printf("Smallest Number:%u\n", smallest);
  printf("Histogram Value Range Width:%u\n", histogram_factor);
  // ------------------------------ Prints

  cudaMalloc(&d_input, input_size);
  cudaMalloc(&d_output, input_size);
  cudaMalloc(&d_global_histogram, histogram_size);
  cudaMalloc(&d_global_histogram_scan, histogram_size);

  cudaMalloc(&d_line_histogram, blocks_per_grid * histogram_size);
  cudaMalloc(&d_vertical_scan, blocks_per_grid * histogram_size);

  for (uint r = 0; r < n_repetitions; r++)
  {
    // ================= warm-up
    cudaMemset(d_line_histogram, 0, blocks_per_grid * histogram_size);
    cudaMemset(d_global_histogram, 0, histogram_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    blockAndGlobalHisto<<<blocks_per_grid, THREADS_PER_BLOCK, histogram_size>>>(d_input, n_elements, d_global_histogram, d_line_histogram, histogram_factor, n_histograms, smallest);
    // =================

    cudaMemset(d_line_histogram, 0, blocks_per_grid * histogram_size);
    cudaMemset(d_global_histogram, 0, histogram_size);

    cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);

    blockAndGlobalHisto<<<blocks_per_grid, THREADS_PER_BLOCK, histogram_size>>>(d_input, n_elements, d_global_histogram, d_line_histogram, histogram_factor, n_histograms, smallest);

    globalHistoScan<<<1, THREADS_PER_BLOCK>>>(d_global_histogram, d_global_histogram_scan, n_histograms);

    verticalScanHH<<<histograms_per_grid, THREADS_PER_BLOCK>>>(d_line_histogram, d_vertical_scan, n_histograms, blocks_per_grid);

    partitionKernel<<<blocks_per_grid, THREADS_PER_BLOCK, histogram_size>>>(d_vertical_scan, d_global_histogram_scan, n_histograms, d_input, d_output, n_elements, smallest, histogram_factor);

    blockBitonicSort<<<n_histograms, THREADS_PER_BLOCK>>>(d_global_histogram_scan, d_output, n_histograms, n_elements);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    simple_sort_milliseconds += time;
  }
  cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_global_histogram);
  cudaFree(d_global_histogram_scan);
  cudaFree(d_line_histogram);
  cudaFree(d_vertical_scan);

  for (uint r = 0; r < n_repetitions; r++)
  {
    thrust::device_vector<uint> d_vec(h_input, h_input + n_elements);

    cudaEventRecord(start);

    thrust::sort(d_vec.begin(), d_vec.end());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    thrust_milliseconds += time;

    thrust::copy(d_vec.begin(), d_vec.end(), thrust_output);
  }

  bool sort_validation = verify_sort(h_output, thrust_output, n_elements);

  if (sort_validation)
    printf("The Sort Is Valid\n");
  else
    printf("The Sort Is Invalid\n");

  double flops = static_cast<double>(n_elements) / ((simple_sort_milliseconds / n_repetitions) / 1000);

  double mflops = flops / 1e6;

  printf("Simple Sort Time: %.2lfms\n", (simple_sort_milliseconds / n_repetitions));
  printf("Throughput: %.2lf MFLOPS\n", mflops);

  printf("Thrust Sort Time: %.2lfms\n", (thrust_milliseconds / n_repetitions));

  double accleration = thrust_milliseconds / simple_sort_milliseconds;

  printf("Acceleration: %.2lf\n", accleration);

  free(h_input);
  free(h_output);
  free(thrust_output);
  free(h_global_histogram_scan);
  free(h_aux);

  printf("\n\n");

  return 0;
}