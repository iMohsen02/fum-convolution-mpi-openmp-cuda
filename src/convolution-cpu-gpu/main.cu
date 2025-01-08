#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <omp.h>
#include <array>
#include <string>
#include <sstream>

#include "algorithm/cpu-convolution.cpp"
#include "algorithm/gpu-convolution.cu"
#include "utils/cpuUtils.cpp"
#include "utils/printUtils.cpp"
#include "utils/mpiUtils.cpp"
#include "models/systemModel.cpp"

int main(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    double start_program = MPI_Wtime();
    double start_initializing = MPI_Wtime();

    int rank, processors;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processors);

    int gpuCount = detectGPU();
    int maxThreads = omp_get_max_threads();
    std::string cpu = getCPU();
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::vector<System> systems(processors);

    const int SIZE = std::atoi(argv[1]);

    gatherAllSystemsInfo(systems, rank, deviceProp, cpu, maxThreads);

    int batchPerCore;
    calcBatchPerCore(rank, batchPerCore, SIZE);

    int myBatch = deviceProp.totalGlobalMem > 0 ? deviceProp.totalGlobalMem / (1024 * 1024) / 100 : maxThreads;
    if (myBatch == 0)
        myBatch = maxThreads;
    myBatch *= batchPerCore;

    std::vector<std::vector<int>> inputMatrix(SIZE, std::vector<int>(SIZE));
    std::vector<std::vector<int>> input(myBatch, std::vector<int>(SIZE, 0));
    std::vector<std::vector<int>> filter = {{1, 0}, {0, -1}};
    std::vector<std::vector<int>> output(myBatch - filter.size() + 1, std::vector<int>(SIZE - filter.size() + 1));

    inputMatrix.resize(SIZE, std::vector<int>(SIZE));
    initializeMatrix(inputMatrix, SIZE);

    double end_initializing = MPI_Wtime();

    double start_scatter_data = MPI_Wtime();
    loadBalanceScatter(rank, inputMatrix, SIZE, processors, myBatch, batchPerCore, systems, input);
    double end_scatter_data = MPI_Wtime();

    double start_calculation = MPI_Wtime();
    detectGPU() ? convolutionGPU(input, filter, output) : convolutionCPU(input, filter, output);
    double end_calculation = MPI_Wtime();

    double start_gather_data = MPI_Wtime();
    gatherData(rank, SIZE, output, processors, systems, batchPerCore, filter);
    double end_gather_data = MPI_Wtime();

    double end_program = MPI_Wtime();

    if (rank == MASTER)
    {

        printf("=========================== MPI Neural Network(Convolution Algorithm) ===========================\n");
        printf("                                                        Developed by Mohsen Gholami - iMohsen02\n");
        printf("                                                        Ferdowsi Uni Of Mashhad(FUM)\n");
        printf("                                                        Advance Parallel Programming\n\n");

        printf("\t%-25s| %-20s\n", "      Table Of Info", "        value        ");
        printf("\t---------------------------------------------------\n");

        printf("\t%-25s| %d \n", "- Convolution Size", SIZE);
        printf("\t%-25s| %d \n", "- Systems Available", processors);
        printf("\t%-25s| %s \n", "- Filter", "Edge Detection(2x2)");

        printf("\n\n");

        printf("\t    Available System Information    \n");
        printf("\t--------------------------------------------------------------------------------\n");

        for (const auto &s : systems)
            std::cout << s;
        printf("\n\n");

        printf("\t%-25s| %-10s\n", "      Table Of Tasks", "Time(sec)");
        printf("\t----------------------------------------\n");

        printf("\t%-25s| %6f \n", "- Program", end_program - start_program);
        printf("\t%-25s| %6f \n", "- Initialize Data", end_initializing - start_initializing);
        printf("\t%-25s| %6f \n", "- Scatter Data", end_scatter_data - start_scatter_data);
        printf("\t%-25s| %6f \n", "- Convolution Algorithm", end_calculation - start_calculation);
        printf("\t%-25s| %6f \n", "- Gather Data", end_gather_data - start_gather_data);
    }

    MPI_Finalize();
    return 0;
}
