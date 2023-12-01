#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        std::cout<<"CUDA error: "<<cudaGetErrorString(err)<<std::endl;  \
        exit(-1);                                                         \
    }                                                                     \
  } while (0)


/* convolution using tiled matrix multiplication */
/* X, Y dimension maps to matrix dimesion, Z is image number */
__global__ void conv_forward_kernel_multiply(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{

    int H_out = (H - K) / S + 1;
    int W_out = (W - K) / S + 1;

    // A is mask, B is input_unrolled, C is multiplication result
    int numARows = M;
    int numAColumns = K * K * C;
    int numBRows = numAColumns;
    int numBColumns = H_out * W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]

    int b = blockIdx.z;

    // Naive approach
    // int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // int Col = blockIdx.x * blockDim.x + threadIdx.x;
    //
    // if(Row < numCRows && Col < numCColumns) 
    // {
    //     float Cval = 0;
    //     for(int i = 0; i < numAColumns; i++) {
    //         Cval += mask[Row * numAColumns + i] * 
    //             in_4d(b, i / (K * K), (Col / H_out) * S + (i % (K * K)) / K, (Col % H_out) * S + (i % (K * K)) % K);
    //         // Cval += mask[Row * numAColumns + i] * input[b * numBRows * numBColumns + i * numBColumns + Col];
    //     }
    //     output[b * numCRows * numCColumns + Row * numCColumns + Col] = Cval;
    // }

    // Tiled approach
    __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Cval = 0;
    int i;

    for(int q = 0; q < (numAColumns - 1) / TILE_WIDTH + 1; q++) {
        if(Row < numARows && q * TILE_WIDTH + tx < numAColumns)
            subTileA[ty][tx] = mask[Row * numAColumns + q * TILE_WIDTH + tx];
        else
            subTileA[ty][tx] = 0;
        if(q * TILE_WIDTH + ty < numBRows && Col < numBColumns) {
            i = q * TILE_WIDTH + ty;
            subTileB[ty][tx] = in_4d(b, i / (K * K), (Col / H_out) * S + (i % (K * K)) / K, (Col % H_out) * S + (i % (K * K)) % K);
            // subTileB[ty][tx] = input[b * numBRows * numBColumns + (q * TILE_WIDTH + ty) * numBColumns + Col];
        }
        else
            subTileB[ty][tx] = 0;
        __syncthreads();
        if(Row < numCRows && Col < numCColumns) {
            for(int k = 0; k < TILE_WIDTH; k++)
                Cval += subTileA[ty][k] * subTileB[k][tx];
        }
        __syncthreads();
    }

    if(Row < numCRows && Col < numCColumns)
        output[b * numCRows * numCColumns + Row * numCColumns + Col] = Cval;

}
	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int num_input_elts = B * C * H * W;
    int num_output_elts = B * M * H_out * W_out;
    int num_mask_elts = M * C * K * K;

    wbCheck(cudaMalloc((void**)device_input_ptr, num_input_elts * sizeof(float)));
    wbCheck(cudaMalloc((void**)device_output_ptr, num_output_elts * sizeof(float)));
    wbCheck(cudaMalloc((void**)device_mask_ptr, num_mask_elts * sizeof(float)));
    wbCheck(cudaMemcpy(*device_input_ptr, host_input, num_input_elts * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(*device_mask_ptr, host_mask, num_mask_elts * sizeof(float), cudaMemcpyHostToDevice));
   
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int W_grid = ceil((W_out * 1.0) / (TILE_WIDTH * 1.0));
    int H_grid = ceil((H_out * 1.0) / (TILE_WIDTH * 1.0));

    // matrix multiplication
    dim3 gridDim_m(ceil((1.0 * H_out * W_out) / (1.0 * TILE_WIDTH)), ceil((1.0 * M) / (1.0 * TILE_WIDTH)),B);
    dim3 blockDim_m(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel_multiply<<<gridDim_m, blockDim_m>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int num_output_elts = B * M * H_out * W_out;
    wbCheck(cudaMemcpy(host_output, device_output, num_output_elts * sizeof(float), cudaMemcpyDeviceToHost));
    // Free device memory
    wbCheck(cudaFree(device_input));
    wbCheck(cudaFree(device_output));
    wbCheck(cudaFree(device_mask));

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
