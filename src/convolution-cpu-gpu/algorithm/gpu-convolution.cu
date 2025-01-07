#include <vector>
#include <cuda_runtime.h>

__global__ void convolutionKernel(int *d_input, int *d_filter, int *d_output,
                                  int inputRows, int inputCols,
                                  int filterRows, int filterCols,
                                  int outputRows, int outputCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < outputRows && col < outputCols)
    {
        int sum = 0;
        
        for (int i = 0; i < filterRows; ++i)
        {
            for (int j = 0; j < filterCols; ++j)
            {
                int inputRow = row + i;
                int inputCol = col + j;
                sum += d_input[inputRow * inputCols + inputCol] * d_filter[i * filterCols + j];
            }
        }
        d_output[row * outputCols + col] = sum;
    }
}

void convolutionGPU(std::vector<std::vector<int>> &input,
                    std::vector<std::vector<int>> &filter,
                    std::vector<std::vector<int>> &output)
{

    int inputRows = input.size();
    int inputCols = input[0].size();
    int filterRows = filter.size();
    int filterCols = filter[0].size();
    int outputRows = output.size();
    int outputCols = output[0].size();

    // Flatten the 2D matrices into 1D arrays for CUDA
    std::vector<int> flatInput(inputRows * inputCols);
    std::vector<int> flatFilter(filterRows * filterCols);
    std::vector<int> flatOutput(outputRows * outputCols, 0);

 #pragma omp parallel for
    for (int i = 0; i < inputRows; ++i)
        std::copy(input[i].begin(), input[i].end(), flatInput.begin() + i * inputCols);
 #pragma omp parallel for
     for (int i = 0; i < filterRows; ++i)
        std::copy(filter[i].begin(), filter[i].end(), flatFilter.begin() + i * filterCols);

    // Allocate device memory
    int *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, flatInput.size() * sizeof(int));
    cudaMalloc(&d_filter, flatFilter.size() * sizeof(int));
    cudaMalloc(&d_output, flatOutput.size() * sizeof(int));

    // Copy data to the device
    cudaMemcpy(d_input, flatInput.data(), flatInput.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, flatFilter.data(), flatFilter.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((outputCols + blockDim.x - 1) / blockDim.x,
                 (outputRows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output,
                                             inputRows, inputCols,
                                             filterRows, filterCols,
                                             outputRows, outputCols);

    // Copy the result back to the host
    cudaMemcpy(flatOutput.data(), d_output, flatOutput.size() * sizeof(int), cudaMemcpyDeviceToHost);

// #pragma parallel for
    for (int i = 0; i < outputRows; ++i)
        std::copy(flatOutput.begin() + i * outputCols, flatOutput.begin() + (i + 1) * outputCols, output[i].begin());

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

}

int detectGPU()
{
    int gpuCount = 0;
    cudaError_t err = cudaGetDeviceCount(&gpuCount);
    return err != cudaSuccess ? 0 : gpuCount;
} 