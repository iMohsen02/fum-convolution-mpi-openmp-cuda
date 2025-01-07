#include <vector>

void convolutionCPU(std::vector<std::vector<int>> &input,
                    std::vector<std::vector<int>> &filter,
                    std::vector<std::vector<int>> &output)
{
    // Dimensions of input and filter
    int inputRows = input.size();
    int inputCols = input[0].size();
    int filterRows = filter.size();
    int filterCols = filter[0].size();

    // Dimensions of output
    int outputRows = inputRows - filterRows + 1;
    int outputCols = inputCols - filterCols + 1;

    // Initialize output matrix with zeros
    output.resize(outputRows, std::vector<int>(outputCols, 0));

#pragma omp parallel for
    for (int i = 0; i < outputRows; ++i)
    {
        for (int j = 0; j < outputCols; ++j)
        {
            int sum = 0;

            // Apply the filter to the corresponding region of the input matrix
            for (int x = 0; x < filterRows; ++x)
            {
                for (int y = 0; y < filterCols; ++y)
                {
                    sum += input[i + x][j + y] * filter[x][y];
                }
            }

            output[i][j] = sum; // Store the result
        }
    }
}