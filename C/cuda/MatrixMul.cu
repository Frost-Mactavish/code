#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
	
#define block 256
#define M 1000
#define N 2000
#define K 3000

static void CUDA_CHECK(cudaError_t err){
    if(err != cudaSuccess){
        printf("CUDA error %d in %s at line %d\n", err,__FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__ void MatrixMul_GPU(float* a, float* b, float* c, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (col < k && row < m){
        for (int i = 0; i < n; i++)
            sum += a[row * n + i] * b[i * k + col];
        c[row * k + col] = sum;
    }
}
	
void MatrixMul(float* a, float* b, float* c, int m, int n, int k){
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < k; ++j){
            float sum = 0.0;
            for (int h = 0; h < n; ++h)
                sum += a[i * n + h] * b[h * k + j];
            c[i * k + j] = sum;
        }
    }
}
	
int main(){

	float *A, *B, *Result, *A_GPU, *B_GPU, *Result_GPU;
    A = (float*)malloc(M * N * sizeof(float));
    B = (float*)malloc(N * K * sizeof(float));
    Result = (float*)malloc(M * K * sizeof(float));

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            A[i * N + j] = rand() % 1024 + 1;
    for (int i = 0; i < N; ++i)
       for (int j = 0; j < K; ++j)
            B[i * K + j] = rand() % 1024 + 1;
    CUDA_CHECK(cudaMalloc((void**)&A_GPU, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&B_GPU, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&Result_GPU, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(A_GPU, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_GPU, B, N * K * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop_cpu, stop_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start);
    cudaEventQuery(start);

    MatrixMul(A, B, Result, M, N, K);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    int grid_rows = (M + block - 1) / block;
    int grid_cols = (K + block - 1) / block;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block, block);
    MatrixMul_GPU << < dimGrid , dimBlock >> > (A_GPU, B_GPU, Result_GPU, M, N, K);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float CPU_time, GPU_time;
    cudaEventElapsedTime(&CPU_time, start, stop_cpu);
    cudaEventElapsedTime(&GPU_time, stop_cpu, stop_gpu);
    printf("GPU Time = %.2f ms.\n", GPU_time);
    printf("CPU Time = %.2f ms.\n", CPU_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(stop_gpu);

    float *Result_copy = Result;
    CUDA_CHECK(cudaMemcpy(Result, Result_GPU, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    int flag = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < K; ++j)
            if (fabs(Result[i * K + j] - Result_copy[i * K + j]) > (1.0e-10))
                flag = 1;
    if (flag)
        printf("Error, Result Discrepency Detected!\n");

    return 0;
}