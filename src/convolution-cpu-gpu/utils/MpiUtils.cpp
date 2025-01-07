#include <vector>
#include <mpi.h>
#include <array>
#include "../models/systemModel.cpp"
#include <cuda_runtime.h>

#define MASTER 0
#define MAX_STRING_LENGTH 255

void gatherInt(std::vector<int> &allGpuCounts, int gpuCount)
{
    MPI_Gather(&gpuCount, 1, MPI_INT, allGpuCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void gatherChars(char sendData[255], std::vector<std::array<char, 255>> &recvData, int rank, int size)
{
    MPI_Allgather(sendData, 255, MPI_CHAR, recvData.data(), 255, MPI_CHAR, MPI_COMM_WORLD);
}

void gatherAllSystemsInfo(std::vector<System> &systems, int rank, cudaDeviceProp deviceProp, std::string cpu, int maxThreads)
{

    std::vector<int> gpu_mem_rec(systems.size());
    gatherInt(gpu_mem_rec, deviceProp.totalGlobalMem / (1024 * 1024));

    std::vector<int> cpu_thread_rec(systems.size());
    gatherInt(cpu_thread_rec, maxThreads);

    char sendData[255];
    snprintf(sendData, sizeof(sendData), "%s", cpu.c_str());

    std::vector<std::array<char, 255>> cpu_rec(systems.size());
    gatherChars(sendData, cpu_rec, rank, systems.size());

    snprintf(sendData, sizeof(sendData), "%s", deviceProp.name);

    std::vector<std::array<char, 255>> gpu_rec(systems.size());
    gatherChars(sendData, gpu_rec, rank, systems.size());

    if (rank == MASTER)
        for (int i = 0; i < systems.size(); i++)
        {
            systems[i].rank = i;
            systems[i].cpu = std::string(cpu_rec[i].data());
            systems[i].gpu = gpu_mem_rec[i] == 0 ? "unknown" : std::string(gpu_rec[i].data());

            systems[i].cpu_max_thread = cpu_thread_rec[i];
            systems[i].gpu_memory = gpu_mem_rec[i];

            System::totalCpuThreads += gpu_mem_rec[i] > 0 ? 0 : cpu_thread_rec[i];
            System::totalGpuThreads += gpu_mem_rec[i] / 100;

            // std::cout << systems[i];
        }
}

void calcBatchPerCore(int rank, int &batchPerCore, const int SIZE)
{
    if (rank == MASTER)
        batchPerCore = SIZE / (System::totalCpuThreads + System::totalGpuThreads);

    MPI_Bcast(&batchPerCore, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
}

void loadBalanceScatter(int rank, std::vector<std::vector<int>> &inputMatrix, const int SIZE, int processors, int myBatch, int bachPerCore, std::vector<System> &systems, std::vector<std::vector<int>> &input)
{
    if (rank == MASTER)
    {
        srand(time(0));

        inputMatrix.resize(SIZE, std::vector<int>(SIZE));
        initializeMatrix(inputMatrix, SIZE);

        int offset = 0;
        for (int r = 0; r < processors; r++)
        {

            int b = bachPerCore * systems[r].getBatch();

            int end = offset + b;
            if (r == MASTER)
                for (int i = offset; i < end; i++)
                    input[i] = inputMatrix[i];
            else
            {
                for (int i = offset; i < offset + b; i++)
                {
                    printf("\rMaster is sending machine #%d data: %.0f %", r, (i - offset) / float(b) * 100);
                    MPI_Send(inputMatrix[i].data(), SIZE, MPI_INT, r, MASTER, MPI_COMM_WORLD);
                }
                printf("\rMaster is sent machine #%d data complete!\n", r);
            }

            offset += b;
        }
    }
    else
    {
        for (int i = 0; i < myBatch; ++i)
        {
            MPI_Recv(input[i].data(), input[i].size(), MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // std::cout << "rank " << rank << " receive: " << myBatch << " data.\n";
    }
}

void gatherData(int rank, const int SIZE, std::vector<std::vector<int>> &output, int processors, std::vector<System> &systems, int batchPerCore, std::vector<std::vector<int>> &filter)
{
    int outIndex = 0;
    if (rank == MASTER)
    {

        std::vector<std::vector<int>> outputMatrix(SIZE, std::vector<int>(SIZE));

        for (int i = 0; i < output.size(); ++i)
            outputMatrix[outIndex++] = output[i];

        for (int p = 1; p < processors; ++p)
        {
            int recLen = systems[p].getBatch() * batchPerCore;
            recLen = recLen - filter.size() + 1;

            for (int i = 0; i < recLen; ++i)
            {
                printf("\rMaster is gathering data from machine #%d: %.0f %", p, (i) / float(recLen) * 100);
                MPI_Recv(outputMatrix[outIndex++].data(), SIZE - filter.size() + 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            printf("\rMaster gathered data from machine #%d complete!\n", p);
        }
        // for (int i = 0; i < outIndex; i++)
        //     printVector(outputMatrix[i]);
    }
    else
    {
        for (int i = 0; i < output.size(); i++)
            MPI_Send(output[i].data(), SIZE - filter.size() + 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
    }
}
