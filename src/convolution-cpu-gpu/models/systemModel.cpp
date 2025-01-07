#include <iostream>
#include <string>

#ifndef SYSTEM_MODEL_H
#define SYSTEM_MODEL_H

struct System
{
    int rank{0};
    std::string cpu{"unkown"};
    std::string gpu{"unkown"};

    int cpu_max_thread{0};
    int gpu_memory{0};
    int batch{0};

    double total_time{0};

    double communication_time{0};
    double gpu_transfor_time{0};
    double gpu_time{0};
    double cpu_time{0};

    static int totalCpuThreads;
    static int totalGpuThreads;

    int getBatch()
    {
        return gpu_memory ? gpu_memory / 100 : cpu_max_thread;
    }
};
int System::totalCpuThreads = 0;
int System::totalGpuThreads = 0;

std::ostream &operator<<(std::ostream &os, const System &system)
{
    os
        << "\t-" << " System Information " << "#" << system.rank << "\n\t\t"
        << "- CPU: " << system.cpu << "\n\t\t"
        << "- Threads: " << system.cpu_max_thread << "\n\t\t"
        << "- GPU: " << system.gpu << "\n\t\t"
        << "- GPU Memory: " << system.gpu_memory << " MB\n\t\t"
        << "- Batch: " << (system.gpu_memory ? system.gpu_memory / 100 : system.cpu_max_thread) << "/" << (System::totalCpuThreads + System::totalGpuThreads) << "\n\t\t"
        << "- GPU Use: " << (system.gpu_memory ? "YES" : "NO") << "\n\n";
    return os;
}

#endif