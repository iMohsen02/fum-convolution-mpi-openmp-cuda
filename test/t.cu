#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

// CUDA Kernel
__global__ void helloFromCUDA() {
    printf("Hello World from CUDA thread %d, block %d\n", threadIdx.x, blockIdx.x);
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    omp_set_num_threads(3);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // OpenMP Parallel Section
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // Get the thread ID
        int num_threads = omp_get_num_threads(); // Get the total number of threads

        #pragma omp critical
        {
            std::cout << "Hello from rank " << rank 
                      << ", thread " << thread_id 
                      << " out of " << num_threads 
                      << " threads\n";
        }
    }

    // Synchronize MPI Processes
    MPI_Barrier(MPI_COMM_WORLD);

    // CUDA Execution
    if (rank == 0) { // Let's run CUDA only on rank 0 for demonstration
        std::cout << "Rank " << rank << ": Launching CUDA kernel...\n";

        // Launch the CUDA kernel
        helloFromCUDA<<<2, 4>>>(); // 2 blocks, 4 threads per block
        cudaDeviceSynchronize();  // Wait for the kernel to complete
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
