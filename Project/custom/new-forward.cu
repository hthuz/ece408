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

__global__ void unroll_input_kernel(const int B, const int C, const int H, const int W, const int K, const float* input, float* output)
{

    int H_out = (H - K) + 1;
    int W_out = (W - K) + 1;
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define unroll_4d(i2,i1,i0) output[(i2) * (K * K * C * H_out * W_out) + (i1) * (H_out * W_out) + i0]

    int W_grid = ceil((W_out * 1.0) / (TILE_WIDTH * 1.0));
    int H_grid = ceil((H_out * 1.0) / (TILE_WIDTH * 1.0));
    // int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;
    if(h < H_out && w < W_out) 
    {
        for(int c = 0; c < C; c++) {
            int w_base = c * (K * K);
            for(int p = 0; p < K; ++p) {
                for(int q = 0; q < K; ++q) {
                    int h_unroll = w_base + p * K + q;
                    int w_unroll = h * W_out + w;
                    unroll_4d(b,h_unroll, w_unroll) = in_4d(b,c,h + p, w + q);
                }
            }
        }
    }
    #undef in_4d
    #undef unroll_4d
}

/* convolution using tiled matrix multiplication */
/* X, Y dimension maps to matrix dimesion, Z is image number */
__global__ void conv_forward_kernel_multiply(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{

    int H_out = (H - K) + 1;
    int W_out = (W - K) + 1;

    // A is mask, B is input_unrolled, C is multiplication result
    int numARows = M;
    int numAColumns = K * K * C;
    int numBRows = numAColumns;
    int numBColumns = H_out * W_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    int b = blockIdx.z;

    // Direct approach
    // if(Row < numCRows && Col < numCColumns) 
    // {
    //     float Cval = 0;
    //     for(int i = 0; i < numAColumns; i++) 
    //         Cval += mask[b * numARows * numAColumns + Row * numAColumns + i]
    //             * input[b * numBRows * numBColumns + i * numBColumns + Col];
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

    for(int q = 0; q < (numAColumns - 1) / TILE_WIDTH + 1; q++) {
        if(Row < numARows && q * TILE_WIDTH + tx < numAColumns)
            subTileA[ty][tx] = mask[b * numARows * numAColumns + Row * numAColumns + q * TILE_WIDTH + tx];
        else
            subTileA[ty][tx] = 0;
        if(q * TILE_WIDTH + ty < numBRows && Col < numBColumns)
            subTileB[ty][tx] = input[b * numBRows * numBColumns + (q * TILE_WIDTH + ty) * numBColumns + Col];
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

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil((W_out * 1.0) / (TILE_WIDTH * 1.0));
    int H_grid = ceil((H_out * 1.0) / (TILE_WIDTH * 1.0));
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int b = blockIdx.z;
    if(w < W_out && h < H_out) {
        float acc = 0.0f;
        for(int c = 0 ; c < C; c++) {
            for(int p = 0; p < K; p++) {
                for(int q = 0; q < K; q++) {
                    acc += in_4d(b, c, h * S + p, w * S + q) * mask_4d(m,c,p,q);
                }
            }
        }
        out_4d(b,m,h,w) = acc;
    }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int num_input_elts = B * C * H * W;
    int num_output_elts = B * M * H_out * W_out;
    int num_mask_elts = M * C * K * K;
    // for(int i = 0; i < 5;i++)
    //     std::cout << i << " " << host_input[i] << std::endl;

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

    // Call unroll kernel
    float* unroll_input;
    int num_unroll_input_elts = K * K * C * H_out * W_out;
    wbCheck(cudaMalloc((void**)&unroll_input, num_unroll_input_elts * sizeof(float)));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH,1);
    dim3 gridDim(M, W_grid * H_grid, B);

    // unroll: C * K * K, H_out * W_out
    unroll_input_kernel<<<gridDim, blockDim>>>(B,C,H,W,K,device_input, unroll_input);

    // Verification
    // float* host_unroll_input;
    // host_unroll_input = (float*)malloc(num_unroll_input_elts * sizeof(float));
    // cudaMemcpy(host_unroll_input, unroll_input, num_unroll_input_elts * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 5;i++)
    //     std::cout << i << " " << host_unroll_input[i] << std::endl;

    dim3 gridDim_m(ceil((1.0 * H_out * W_out) / (1.0 * TILE_WIDTH)), ceil((1.0 * M) / (1.0 * TILE_WIDTH)),1);
    dim3 blockDim_m(TILE_WIDTH, TILE_WIDTH, 1);
    conv_forward_kernel_multiply<<<gridDim_m, blockDim_m>>>(device_output, unroll_input, device_mask, B, M, C, H, W, K, S);



    

    // conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    int num_output_elts = B * M * H_out * W_out;
    wbCheck(cudaMemcpy(host_output, device_output, num_output_elts * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "OUTPUT" << std::endl;
    for(int i = 0; i < 5;i++)
        std::cout << i << " " << host_output[i] << std::endl;
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
