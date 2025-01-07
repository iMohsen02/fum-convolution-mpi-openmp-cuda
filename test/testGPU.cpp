#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;

    // Get the number of CUDA devices
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No NVIDIA GPU detected on this system." << std::endl;
    } else {
        std::cout << "Number of NVIDIA GPUs detected: " << deviceCount << std::endl;

        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);

            std::cout << "GPU " << i << ": " << deviceProp.name << std::endl;
            std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            std::cout << "  Total global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
            std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
            std::cout << "  Max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        }
    }

    return 0;
}
