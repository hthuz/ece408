// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
    __shared__ float T[2 *  BLOCK_SIZE]; 
    // Loading shared mem
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    T[2 * threadIdx.x] = i < len ? input[i] : 0;
    T[2 * threadIdx.x + 1] = i + 1 < len ? input[i + 1] : 0;
    __syncthreads();

    // Reduction step
    int stride = 1;
    while(stride < 2 * BLOCK_SIZE) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < 2 * BLOCK_SIZE && index - stride >= 0)
            T[index] += T[index - stride];
        stride *= 2;
    }

    // Post scan step
    stride = BLOCK_SIZE / 2;
    while(stride > 0) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride + 2 - 1;
        if(index + stride < 2 * BLOCK_SIZE)
            T[index + stride] += T[index];
        stride /= 2;
    }

    if (i < len)
        output[i] = T[2 * threadIdx.x];
    if (i + 1 < len)
        output[i + 1] = T[2 * threadIdx.x + 1];

}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  wbLog(TRACE, "DIM GRID: ", ceil(numElements * 1.0 / (2.0 * BLOCK_SIZE)));
  dim3 DimGrid(ceil(numElements * 1.0 / (2.0 * BLOCK_SIZE)), 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  wbLog(TRACE, "Output[0] ",hostOutput[0]);
  wbLog(TRACE, "Output[1] ",hostOutput[1]);
  wbLog(TRACE, "Output[2] ",hostOutput[2]);
  wbLog(TRACE, "Output[3] ",hostOutput[3]);
  free(hostInput);
  free(hostOutput);

  return 0;
}
