#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>


void printDeviceProperties(int deviceId = 0) {

    cudaDeviceProp deviceProp;
    cudaError_t error = cudaGetDeviceProperties(&deviceProp, 0);
    
    if (error != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(error) << "\n";
        return;
    }

        // Print device properties in a table format using printf and ANSI colors
    printf("\r\033[96m%-30s\033[91m%d\033[m\n", "Device ID:", deviceId);
    printf("\033[96m%-30s\033[91m%s\033[m\n", "Name:", deviceProp.name);
    printf("\033[96m%-30s\033[91m%lld bytes\033[m\n", "Total global memory:", deviceProp.totalGlobalMem);
    printf("\033[96m%-30s\033[91m%lld bytes\033[m\n", "Shared memory per block:", deviceProp.sharedMemPerBlock);
    printf("\033[96m%-30s\033[91m%d kHz\033[m\n", "Clock rate:", deviceProp.clockRate);
    printf("\033[96m%-30s\033[91m%d\033[m\n", "Warp size:", deviceProp.warpSize);
    printf("\033[96m%-30s\033[91m%d\033[m\n", "Multiprocessor count:", deviceProp.multiProcessorCount);
    printf("\033[96m%-30s\033[91m%d.%d\033[m\n", "CUDA capability:", deviceProp.major, deviceProp.minor);
    printf("\033[96m%-30s\033[91m%d\033[m\n", "Compute capability:", deviceProp.computeMode);
    printf("\033[96m%-30s\033[91m%d\033[m\n", "Max threads per block:", deviceProp.maxThreadsPerBlock);
    printf("\033[96m%-30s\033[91m(%d, %d, %d)\033[m\n", "Max threads dimension:", deviceProp.maxThreadsDim[0],
           deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("\033[96m%-30s\033[91m(%d, %d, %d)\033[m\n", "Max grid size:", deviceProp.maxGridSize[0],
           deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
}

// CUDA kernel for performing convolution
__global__ void convolutionKernel(int *input, int *filter, int *output, int size, int filter_size) {
    int tx = threadIdx.x + blockIdx.x * blockDim.x; // Global row index
    int ty = threadIdx.y + blockIdx.y * blockDim.y; // Global column index

    // Output size calculation
    int output_size = size - filter_size + 1;

    if (tx < output_size && ty < output_size) {
        int sum = 0;
        for (int i = 0; i < filter_size; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                sum += input[(tx + i) * size + (ty + j)] * filter[i * filter_size + j];
            }
        }
        output[tx * output_size + ty] = sum;
    }
}

// Function to initialize a matrix
void initializeMatrix(std::vector<int> &matrix, int size) {
    #pragma omp parallel for
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = rand() % 50;
    }
}

// Function to print a matrix
void printMatrix(const std::vector<int> &matrix, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << matrix[r * cols + c] << "\t";
        }
        std::cout << "\n";
    }
}

void printSummary(int SIZE, double  duration, double durationTransferInit, double durationTransfer, double durationTransfer2) {
    printf("\r============================== MPI CNN Neural Network(Convolution Algorithm) ==============================\n");
    printf("                                                        Developed by Mohsen Gholami - iMohsen02\n");
    printf("                                                        Ferdowsi Uni Of Mashhad(FUM)\n");
    printf("\t\033[96m %-20s \033[91m %-8d \n\033[m", "Matrix Size:", SIZE);
    printf("\t\033[96m %-20s \033[91m %-8d MB\n\033[m", "RAM Usage:", ((SIZE * SIZE) * sizeof(int)) / 1'000'000);
    printf("\n#######################################################################################################\n\n");
    printf("\033[96m %-30s \033[91m %0.0f \033[m %s\n", "Initialize takes", (durationTransferInit), "milliseconds");
    printf("\033[96m %-30s \033[91m %0.0f \033[m %s\n", "Host to Device takes", (durationTransfer ), "milliseconds");
    printf("\033[96m %-30s \033[91m %0.0f \033[m %s\n", "GPU takes", (duration ), "milliseconds");
    printf("\033[96m %-30s \033[91m %0.0f \033[m %s\n", "Device to Host takes", (durationTransfer2 ), "milliseconds");
    printf("\n#######################################################################################################\n\n");
}


int main(int argc, char **argv) {
    system("cls");
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
        return EXIT_FAILURE;
    }

    const int SIZE = std::atoi(argv[1]); // Matrix size
    const int FILTER_SIZE = 2;          // Filter size

    std::vector<int> input(SIZE * SIZE);
    std::vector<int> filter = {1, 0, 0, -1}; // 2x2 filter
    int output_size = SIZE - FILTER_SIZE + 1;
    std::vector<int> output(output_size * output_size, 0);

    // Initialize input matrix
    srand(time(0));
    auto startInit = std::chrono::high_resolution_clock::now();

    printf("\rinitializing arrays ...                              ");
    initializeMatrix(input, SIZE);
    auto endInit = std::chrono::high_resolution_clock::now();
    double durationTransferInit = std::chrono::duration_cast<std::chrono::milliseconds>(endInit - startInit).count();

    // Allocate memory on the device
    printf("\rMallocing arrays on device ...                              ");
    int *d_input, *d_filter, *d_output;
    cudaMalloc(&d_input, SIZE * SIZE * sizeof(int));
    cudaMalloc(&d_filter, FILTER_SIZE * FILTER_SIZE * sizeof(int));
    cudaMalloc(&d_output, output_size * output_size * sizeof(int));

    // Copy data to device
    auto startC = std::chrono::high_resolution_clock::now();

    printf("\rCopying arrays to device ...                              ");
    cudaMemcpy(d_input, input.data(), SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, filter.data(), FILTER_SIZE * FILTER_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    auto endC = std::chrono::high_resolution_clock::now();
    double durationTransfer = std::chrono::duration_cast<std::chrono::milliseconds>(endC - startC).count();


    // Launch the kernel
    dim3 blockDim(16, 16); // Block of 16x16 threads
    dim3 gridDim((output_size + 15) / 16, (output_size + 15) / 16);
    auto start = std::chrono::high_resolution_clock::now();
    printf("\rRunning Kernel ...                              ");
    convolutionKernel<<<gridDim, blockDim>>>(d_input, d_filter, d_output, SIZE, FILTER_SIZE);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    auto startC2 = std::chrono::high_resolution_clock::now();
    printf("\rCoping output to host ...                              ");
    cudaMemcpy(output.data(), d_output, output_size * output_size * sizeof(int), cudaMemcpyDeviceToHost);
    auto endC2 = std::chrono::high_resolution_clock::now();
    double durationTransfer2 = std::chrono::duration_cast<std::chrono::milliseconds>(endC2 - startC2).count();


    // Print results
    // std::cout << "Input Matrix:\n";
    // printMatrix(input, SIZE, SIZE);
    // std::cout << "\nFilter:\n";
    // printMatrix(filter, FILTER_SIZE, FILTER_SIZE);
    // std::cout << "\nOutput Matrix:\n";
    // printMatrix(output, output_size, output_size);

    // Display execution time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printDeviceProperties();
    printSummary(SIZE, duration,durationTransferInit,durationTransfer,durationTransfer2);




    // Free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    return EXIT_SUCCESS;
}
