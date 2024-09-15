// Discente: Gustavo Henrique Ferreira Cruz
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <stdio.h>
#include <limits.h>

#define CUDA_CHECK(call)                                               \
  {                                                                    \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess)                                            \
    {                                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  }

#define THREADS_PER_BLOCK 1024
#define SHARED_SIZE_LIMIT 1024U

namespace cg = cooperative_groups;

__global__ void blockAndGlobalHisto(uint *input, uint n_elements, uint *global_histogram, uint *line_histogram, uint histogram_factor, uint n_histograms, uint smallest)
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

__global__ void globalHistoScan(uint *global_histogram, uint *global_histogram_scan, uint n_histograms)
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

__global__ void verticalScanHH(uint *line_histogram, uint *vertical_scan, uint n_histograms, uint blocks_per_grid)
{
  uint tid = threadIdx.x;

  if (tid == 0)
  {
    for (uint i = 0; i < blocks_per_grid; i++)
    {
      for (uint j = 0; j < n_histograms; j++)
      {
        if (i == 0)
        {
          vertical_scan[j] = 0;
        }
        else
        {
          vertical_scan[i * n_histograms + j] = vertical_scan[(i - 1) * n_histograms + j] + line_histogram[(i - 1) * n_histograms + j];
        }
      }
    }
  }
}

__global__ void partitionKernel(uint *vertical_scan, uint *global_scan, uint n_histograms, uint *input, uint *output, uint n_elements, uint smallest, uint histogram_factor)
{
  extern __shared__ uint hist_counter[];

  uint thread_id = threadIdx.x;
  uint block_id = blockIdx.x;

  uint curr_line_index_mult = block_id * n_histograms;
  uint curr_input_offset_mult = block_id * blockDim.x;

  if (thread_id < n_histograms)
    hist_counter[thread_id] = vertical_scan[thread_id + curr_line_index_mult] + global_scan[thread_id];

  __syncthreads();

  if (thread_id + curr_input_offset_mult < n_elements)
  {
    uint input_value = input[thread_id + curr_input_offset_mult];
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

  uint block_id = blockIdx.x;
  uint block_dim = blockDim.x;
  uint thread_id = threadIdx.x;

  __shared__ uint aux[SHARED_SIZE_LIMIT];

  uint start_index = global_histogram_scan[block_id];
  uint end_index = n_elements;
  if (block_id + 1 < n_histograms)
    end_index = global_histogram_scan[block_id + 1];

  if (start_index + thread_id < end_index)
    aux[thread_id] = output[start_index + thread_id];
  else
    aux[thread_id] = UINT_MAX;

  __syncthreads();

  uint dir = 1;

  for (uint size = 2; size < block_dim; size <<= 1)
  {
    // Bitonic merge
    uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

    for (uint stride = size / 2; stride > 0; stride >>= 1)
    {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      if (pos + stride < block_dim)
        compareAndSwap(aux, pos, pos + stride, ddd);
    }
  }

  {
    for (uint stride = block_dim / 2; stride > 0; stride >>= 1)
    {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      if (pos + stride < block_dim)
        compareAndSwap(aux, pos, pos + stride, dir);
    }
  }

  __syncthreads();

  if (start_index + thread_id < end_index)
    output[start_index + thread_id] = aux[thread_id];
}

bool verify_sort(uint *a, uint *b, uint size)
{
  bool flag = true;
  for (uint i = 0; i < size; i++)
    if (a[i] != b[i])
    {
      flag = false;
      printf("Mismatch at [%u]: %u X %u\n", i, a[i], b[i]);
    }

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

  std::srand(0);

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

  uint *d_input, *h_input = (uint *)malloc(input_size), *d_output, *h_output = (uint *)malloc(input_size), *thrust_output = (uint *)malloc(input_size), smallest, biggest, *d_global_histogram, *d_line_histogram, *d_global_histogram_scan, *h_global_histogram_scan = (uint *)malloc(histogram_size), *d_vertical_scan, *h_aux = (uint *)malloc(max_size_aux * sizeof(uint)), *d_aux;

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

  cudaEventRecord(start);
  for (uint r = 0; r < n_repetitions; r++)
  {
    blockAndGlobalHisto<<<blocks_per_grid, THREADS_PER_BLOCK, histogram_size>>>(d_input, n_elements, d_global_histogram, d_line_histogram, histogram_factor, n_histograms, smallest);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Execução 1 finalizada\n");

    globalHistoScan<<<1, THREADS_PER_BLOCK, histogram_size>>>(d_global_histogram, d_global_histogram_scan, n_histograms);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Execução 2 finalizada\n");

    verticalScanHH<<<1, THREADS_PER_BLOCK>>>(d_line_histogram, d_vertical_scan, n_histograms, blocks_per_grid);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Execução 3 finalizada\n");

    partitionKernel<<<blocks_per_grid, THREADS_PER_BLOCK, histogram_size>>>(d_vertical_scan, d_global_histogram_scan, n_histograms, d_input, d_output, n_elements, smallest, histogram_factor);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Execução 4 finalizada\n");

    blockBitonicSort<<<n_histograms, THREADS_PER_BLOCK>>>(d_global_histogram_scan, d_output, n_histograms, n_elements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Execução 5 finalizada\n");
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_global_histogram);
  cudaFree(d_global_histogram_scan);
  cudaFree(d_line_histogram);
  cudaFree(d_vertical_scan);
  cudaFree(d_aux);

  float simple_sort_milliseconds = 0;
  cudaEventElapsedTime(&simple_sort_milliseconds, start, stop);

  cudaEventRecord(start);
  for (uint r = 0; r < n_repetitions; r++)
  {
    thrust::device_vector<uint> d_vec(h_input, h_input + n_elements);
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), thrust_output);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  bool sort_validation = verify_sort(h_output, thrust_output, n_elements);

  if (sort_validation)
    printf("The Sort Is Valid\n");
  else
    printf("The Sort Is Invalid\n");

  float thrust_milliseconds = 0;
  cudaEventElapsedTime(&thrust_milliseconds, start, stop);

  double ops = static_cast<double>(n_elements) / ((simple_sort_milliseconds / n_repetitions) / 1000);

  printf("Simple Sort Time: %.2lfms\n", (simple_sort_milliseconds / n_repetitions));
  printf("Throughput: %.2lf\n", ops);

  printf("Thrust Sort Time: %.2lfms\n", (thrust_milliseconds / n_repetitions));

  free(h_input);
  free(h_output);
  free(thrust_output);
  free(h_global_histogram_scan);
  free(h_aux);

  return 0;
}