#include <intrin.h>
#include <string>
#include <vector>

std::string getCPU()
{
    int cpuInfo[4];
    __cpuid(cpuInfo, 0x80000002);
    char model[48];
    memcpy(model, &cpuInfo[0], sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000003);
    memcpy(model + 16, &cpuInfo[0], sizeof(cpuInfo));
    __cpuid(cpuInfo, 0x80000004);
    memcpy(model + 32, &cpuInfo[0], sizeof(cpuInfo));
    return std::string(model);
}

void initializeMatrix(std::vector<std::vector<int>> &matrix, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            matrix[i][j] = rand() % 50;
        }
    }
}