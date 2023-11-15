// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256


//@@ insert code here
__global__ void float_2uchar(float *input , unsigned char* output, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size)
        output[i] = (unsigned char)(255 * input[i]);
}

__global__ void rgb_2gray(unsigned char* input, unsigned char* output, int gsize, int imageChannels)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < gsize)
        output[i] = (unsigned char)(0.21 * input[imageChannels * i] + 0.71 * input[imageChannels * i + 1] + 0.07 * input[imageChannels * i + 2]);
}

__global__ void histo_kernel(unsigned char* buffer, int gsize, unsigned int *histo)
{
    __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
    if(threadIdx.x < HISTOGRAM_LENGTH)
        histo_private[threadIdx.x] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while(i < gsize) {
        atomicAdd(&(histo_private[buffer[i]]), 1);
        i += stride;
    }
    __syncthreads();

    if(threadIdx.x < HISTOGRAM_LENGTH)
        atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);

}

__global__ void scan(unsigned int *input, float *output, float *output_min, int len, int numPixels) {

    #define p(x) ((1.0 * x) / (1.0 * numPixels))

    __shared__ float T[2 *  BLOCK_SIZE]; 
    // Loading shared mem
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    T[2 * threadIdx.x] = i < len ? p(input[i]) : 0;
    T[2 * threadIdx.x + 1] = i + 1 < len ? p(input[i + 1]) : 0;
    __syncthreads();

    // Reduction step
    int stride = 1;
    while(stride < 2 * BLOCK_SIZE) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < 2 * BLOCK_SIZE && (index - stride) >= 0)
            T[index] += T[index - stride];
        stride *= 2;
    }

    // Post scan step
    stride = BLOCK_SIZE / 2;
    while(stride > 0) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index + stride < 2 * BLOCK_SIZE)
            T[index + stride] += T[index];
        stride /= 2;
    }

    __syncthreads();
    if (i < len)
        output[i] = T[2 * threadIdx.x];
    if (i + 1 < len)
        output[i + 1] = T[2 * threadIdx.x + 1];

    if (threadIdx.x == 0)
        output_min[0] = p(input[0]);

    #undef p
}

__global__  void histo_equalization(unsigned char* input, unsigned char* output, float* CDF, float* CDFmin, int size)
{
    // #define clamp(x,start,end) (min(max(x, start), end))
    // #define correct_color(val) (clamp(255 * (CDF[val] - CDFmin[0]) / (1.0 - CDFmin[0]), 0.0, 255.0 ))
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        output[i] = (unsigned char) min(max(255 * (CDF[(int) input[i]] - CDF[0]) / (1.0 - CDF[0]),0.0),255.0);
        // output[i] = correct_color(input[i]);
    // #undef correct_color
    // #undef clamp
}

__global__ void uchar_2float(unsigned char *input , float* output, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size)
        output[i] = (float) (input[i] / 255.0);
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here

    // Step 1: float to uchar
    int size = imageWidth * imageHeight * imageChannels;
    int gsize = imageWidth * imageHeight; // Grayscale image size
    wbLog(TRACE, "size is ",size, " gray size is ", gsize);
    float* devInputImageData;
    unsigned char* devUcharImage;
    cudaMalloc((void**)&devInputImageData, size * sizeof(float));
    cudaMalloc((void**)&devUcharImage, size * sizeof(unsigned char));
    cudaMemcpy(devInputImageData, hostInputImageData, size * sizeof(float), cudaMemcpyHostToDevice);
    float_2uchar<<<ceil( (1.0 * size) / (1.0 * BLOCK_SIZE) ), BLOCK_SIZE>>>(devInputImageData, devUcharImage, size);

    // -------------------------------
    // PRINT
    // -------------------------------
    // unsigned char* hostUcharImage;
    // hostUcharImage = (unsigned char*) malloc(size * sizeof(unsigned char));
    // cudaMemcpy(hostUcharImage, devUcharImage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // for(int i = size - 20; i < size; i++) {
    //     wbLog(TRACE, "floatin: ", hostInputImageData[i], " char: ", (int) hostUcharImage[i] );
    // }
    // -------------------------------
    // -------------------------------


    // Step 2: RGB to grayscale
    unsigned char* devGrayImage;
    cudaMalloc((void**)&devGrayImage, gsize * sizeof(unsigned char));
    rgb_2gray<<<ceil( (1.0 * gsize) / (1.0 * BLOCK_SIZE) ), BLOCK_SIZE>>>(devUcharImage,devGrayImage,gsize, imageChannels);

    // -------------------------------
    // PRINT
    // -------------------------------
    // unsigned char* hostGrayImage;
    // hostGrayImage = (unsigned char*) malloc(size * sizeof(unsigned char));
    // cudaMemcpy(hostGrayImage, devGrayImage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < 10; i++) {
    //     wbLog(TRACE, "gray image: ", (int) hostGrayImage[i]);
    // }
    // -------------------------------
    // -------------------------------
    
    // Step 3: Histogram of grayImage
    unsigned int* devHisto;
    cudaMalloc((void**)&devHisto, HISTOGRAM_LENGTH * sizeof(int));
    cudaMemset(devHisto, 0, HISTOGRAM_LENGTH * sizeof(int));
    histo_kernel<<<ceil( (1.0 * gsize) / (1.0 * BLOCK_SIZE) ), BLOCK_SIZE>>>(devGrayImage, gsize, devHisto);

    // -------------------------------
    // PRINT
    // -------------------------------
    // int* hostHisto;
    // hostHisto = (int*) malloc(size * sizeof(int));
    // cudaMemcpy(hostHisto, devHisto, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost);
    // int sum = 0;
    // for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
    //     wbLog(TRACE, i , " histo: ", hostHisto[i]);
    //     sum += hostHisto[i];
    // }
    // wbLog(TRACE, "Sum: ", sum);
    // -------------------------------
    // -------------------------------


    // Step 4: scan to compute cdf
    float* devCDF;
    float* devCDFmin;
    cudaMalloc((void**)&devCDF, HISTOGRAM_LENGTH * sizeof(float));
    cudaMalloc((void**)&devCDFmin, 1 * sizeof(float));
    scan<<<1,BLOCK_SIZE>>>(devHisto, devCDF,devCDFmin, HISTOGRAM_LENGTH, gsize );

    // -------------------------------
    // PRINT
    // -------------------------------
    // float* hostCDF;
    // float* hostCDFmin;
    // hostCDF = (float*) malloc(HISTOGRAM_LENGTH * sizeof(float));
    // hostCDFmin = (float*) malloc(1 * sizeof(float));
    // cudaMemcpy(hostCDF, devCDF, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(hostCDFmin, devCDFmin, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = HISTOGRAM_LENGTH - 100; i < HISTOGRAM_LENGTH; i++) {
    //     wbLog(TRACE, i , " cdf: ", hostCDF[i]);
    // }
    // wbLog(TRACE, "CDFmin: ", hostCDFmin[0]);
    // -------------------------------
    // -------------------------------



    // Step 5: apply histogram equalization function
    unsigned char* devUcharImageCorrected;
    cudaMalloc((void**)&devUcharImageCorrected, size * sizeof(unsigned char));
    histo_equalization<<<ceil( (1.0 * size) / (1.0 * BLOCK_SIZE) ),BLOCK_SIZE>>>(devUcharImage,devUcharImageCorrected,devCDF, devCDFmin, size);

    // -------------------------------
    // PRINT
    // -------------------------------
    // unsigned char* hostUcharImageCorrected;
    // hostUcharImageCorrected = (unsigned char*) malloc(size * sizeof(unsigned char));
    // cudaMemcpy(hostUcharImageCorrected, devUcharImageCorrected, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // for(int i = size - 20; i < size; i++) {
    //     wbLog(TRACE, i , " corrected: ", (int) hostUcharImageCorrected[i]);
    // }

    // -------------------------------
    // -------------------------------

    // Step 6: uchar to float
    float* devOutputImageData;
    cudaMalloc((void**)&devOutputImageData, size * sizeof(float));
    uchar_2float<<<ceil( (1.0 * size) / (1.0 * BLOCK_SIZE) ), BLOCK_SIZE>>>(devUcharImageCorrected, devOutputImageData, size);
    cudaMemcpy(hostOutputImageData, devOutputImageData, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // -------------------------------
    // PRINT
    // -------------------------------
    // for(int i = 0; i < 200; i++) {
    //     wbLog(TRACE, i , " output: ", hostOutputImageData[i]);
    // }

    // -------------------------------
    // -------------------------------

    wbExport("outputimg.ppm", outputImage);

  wbSolution(args, outputImage);

  //@@ insert code here
    cudaFree(devInputImageData);
    cudaFree(devUcharImage);
    cudaFree(devGrayImage);
    cudaFree(devHisto);
    cudaFree(devCDF);
    cudaFree(devCDFmin);
    cudaFree(devUcharImageCorrected);
    cudaFree(devOutputImageData);

  return 0;
}
