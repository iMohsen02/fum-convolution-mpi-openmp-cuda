#include <iostream>
#include <vector>
#include <string>

void printVector(std::vector<std::string> &vec)
{
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << ", ";
    }
    std::cout << "\b\b]\n";
}
void printVector(std::vector<int> &vec)
{
    std::cout << "[";
    for (int i = 0; i < vec.size(); i++)
    {
        std::cout << vec[i] << ", ";
    }
    std::cout << "\b\b]\n";
}

void printProcInfo(const std::vector<int> &gpus, const std::vector<int> &cpus)
{
    printf("%-10s %-20s %-20s\n", "Rank", "CPU", "GPU");
    for (int s = 0; s < gpus.size(); s++)
        printf("%-10d %-20d %-20d\n", s, cpus[s], gpus[s]);
}

// std::vector<std::string> splitStringByComma(const std::string &str, char split)
// {
//     std::vector<std::string> result;
//     std::stringstream ss(str);
//     std::string token;

//     while (std::getline(ss, token, split))
//         result.push_back(token);

//     return result;
// }