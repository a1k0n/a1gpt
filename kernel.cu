#include <iostream>
#include <chrono>

__global__ void arithmeticKernel(int numOperations, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float x = (float)index;
    for(int i = 0; i < numOperations; i++) {
        x = x * x + x;
    }
    output[index] = x;  // Add a side-effect to prevent optimization
}

float* gpuTransferFloats(float *data, int size) {
    float *array;
    cudaMallocManaged(&array, size * sizeof(float));
    cudaMemcpy(array, data, size * sizeof(float), cudaMemcpyHostToDevice);
    return array;
}

void gpuDumpMemoryInfo() {
    size_t free_byte;
    size_t total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
           used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
           total_db / 1024.0 / 1024.0);
}

int benchmark() {
    int numOperations = 10000000;
    int numThreadsList[] = {32, 64, 128, 256, 512, 1024};
    int numBlocks;

    // Allocate output array
    float* output;
    cudaMallocManaged(&output, 6144 * 32 * sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    // Run a warm-up kernel
    arithmeticKernel<<<192, 1024>>>(numOperations, output);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken by warmup kernel with numOperations=" << numOperations << ": " << duration.count() << " microseconds" << std::endl;

    for(int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            numBlocks = 6144 * 32 / numThreadsList[i];  // Fill the GPU
            auto start = std::chrono::high_resolution_clock::now();
            arithmeticKernel<<<numBlocks, numThreadsList[i]>>>(numOperations, output);
            cudaDeviceSynchronize();  // Ensure completion of the kernel
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "Time taken by kernel with " << numBlocks << " blocks and " 
                    << numThreadsList[i] << " threads per block: " 
                    << duration.count() << " microseconds" << std::endl;
        }
    }

    // Deallocate output array
    cudaFree(output);
    return 0;
}
