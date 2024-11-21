#include "iostream"
#include "vector"
#include "cuda_runtime.h"
#include "cublas_v2.h"
using namespace std;

#define batchSize 96
#define M 197
#define N 768
#define thread 1024

// cuda API error checking
static void CUDA_CHECK(cudaError_t err){
    if(err != cudaSuccess){
        printf("CUDA error %d in %s at line %d\n", err,__FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}
// cuBLAS API error checking
static void CUBLAS_CHECK(cublasStatus_t err){
    if(err != CUBLAS_STATUS_SUCCESS){
        printf("CUBLAS error %d in %s at line %d\n", err, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

vector<vector<float>> tensor(batchSize, vector<float>(M * N));
vector<vector<float>> tensor_copy(batchSize, vector<float>(M * N));

vector<vector<float>> w1(batchSize, vector<float>(768 * 2304));
vector<vector<float>> b1(batchSize, vector<float>(197 * 2304));
vector<vector<float>> w2(batchSize, vector<float>(768 * 3072));
vector<vector<float>> b2(batchSize, vector<float>(197 * 3072));
vector<vector<float>> w3(batchSize, vector<float>(3072 * 768));
vector<vector<float>> b3(batchSize, vector<float>(197 * 768));

vector<vector<float>> Q(batchSize * 8, vector<float>(197 * 96));
vector<vector<float>> K(batchSize * 8, vector<float>(197 * 96));
vector<vector<float>> V(batchSize * 8, vector<float>(197 * 96));
vector<vector<float>> QK(batchSize * 8, vector<float>(197 * 197));


void tensor_init(int m, int n){
    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                tensor[b][j * m + i] = (float)(rand() % 101) / 101;
}

void linear_init(){
    const int m = M;
    const int n = N;
    const int k = 2304;
    const int t = 3072;

    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++)
                w1[b][j * n + i] = (float)(rand() % 101) / 101;
    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < k; j++)
                b1[b][j * m + i] = (float)(rand() % 101) / 101;
    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < t; j++)
                w2[b][j * n + i] = (float)(rand() % 101) / 101;
    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < t; j++)
                b2[b][j * m + i] = (float)(rand() % 101) / 101;
    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < t; i++)
            for (int j = 0; j < n; j++)
                w3[b][j * t + i] = (float)(rand() % 101) / 101;
    for (int b = 0; b < batchSize; b++)
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b3[b][j * m + i] = (float)(rand() % 101) / 101;

}

void qkv_init(int m, int n){
    const int m_batchSize = batchSize * 8;
    for (int b = 0; b < m_batchSize; b++){
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                Q[b][j * m + i] = (float)(rand() % 101) / 101;
                K[b][j * m + i] = (float)(rand() % 101) / 101;
                V[b][j * m + i] = (float)(rand() % 101) / 101;
            }
        }
    }
}

float Att_Linear(int input_dim, int output_dim){
    const int m = M;
    const int n = output_dim;
    const int k = input_dim;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const float alpha = 1.0;
    const float beta = 1.0;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    float** d_A_array = nullptr;
    float** d_B_array = nullptr;
    float** d_C_array = nullptr;

    vector<float*> d_A(batchSize, nullptr);
    vector<float*> d_B(batchSize, nullptr);
    vector<float*> d_C(batchSize, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * tensor[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(float) * w1[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(float) * b1[i].size()));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(float*) * batchSize));

    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], tensor[i].data(), sizeof(float) * tensor[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], w1[i].data(), sizeof(float) * w1[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], b1[i].data(), sizeof(float) * b1[i].size(), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
   
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda, d_B_array, ldb, &beta, d_C_array, ldc, batchSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* step 4: copy data to host */
    for (int i = 0; i < batchSize; i++)
        CUDA_CHECK(cudaMemcpyAsync(b1[i].data(), d_C[i], sizeof(float) * b1[i].size(), cudaMemcpyDeviceToHost, stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return elapsedTime;

}


float MatrixMul_1(){
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = M;
    const int n = M;
    const int k = N / 8;
    const int lda = m;
    const int ldb = n;
    const int ldc = m;
    const int m_batchSize = batchSize * 8;

    const float alpha = 1 / pow(24, -0.5);
    const float beta = 0;

    float** d_A_array = nullptr;
    float** d_B_array = nullptr;
    float** d_C_array = nullptr;

    vector<float*> d_A(m_batchSize, nullptr);
    vector<float*> d_B(m_batchSize, nullptr);
    vector<float*> d_C(m_batchSize, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < m_batchSize; i++){
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * Q[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(float) * K[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(float) * QK[i].size()));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(float*) * batchSize));

    for (int i = 0; i < m_batchSize; i++){
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], Q[i].data(), sizeof(float) * Q[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], K[i].data(), sizeof(float) * K[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], QK[i].data(), sizeof(float) * QK[i].size(), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda, d_B_array, ldb, &beta, d_C_array, ldc, batchSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* step 4: copy data to host */
    for (int i = 0; i < m_batchSize; i++)
        CUDA_CHECK(cudaMemcpyAsync(QK[i].data(), d_C[i], sizeof(float) * QK[i].size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < m_batchSize; i++){
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return elapsedTime;
}

float softmax(){
    const int m = M;
    const int count = batchSize * 8 * m;
    float s[count] = {};
    int t = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int b = 0; b < batchSize * 8; b++){
        for (int i = 0; i < m; i++){
            for (int j = 0; j < m; j++){
                s[t]= s[t]+ exp(QK[b][j * m + i]);
            }
            t++;
        }
    }
    t = 0;
    for (int b = 0; b < batchSize * 8; b++)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                QK[b][j * m + i] = exp(QK[b][j * m + i]) / s[t];
            }
            t++;
        }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

float MatrixMul_2(){
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const int m = M;
    const int n = N / 8;
    const int k = m;
    const int lda = m;
    const int ldb = k; 
    const int ldc = m;
    const int m_batchSize = batchSize * 8;

    const float alpha = 1.0;
    const float beta = 0;

    float** d_A_array = nullptr;
    float** d_B_array = nullptr;
    float** d_C_array = nullptr;

    vector<float*> d_A(m_batchSize, nullptr);
    vector<float*> d_B(m_batchSize, nullptr);
    vector<float*> d_C(m_batchSize, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N; 

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < m_batchSize; i++){
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * QK[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(float) * V[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(float) * Q[i].size()));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(float*) * batchSize));

    for (int i = 0; i < m_batchSize; i++){
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], QK[i].data(), sizeof(float) * QK[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], V[i].data(), sizeof(float) * V[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], Q[i].data(), sizeof(float) * Q[i].size(), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda, d_B_array, ldb, &beta, d_C_array, ldc, batchSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* step 4: copy data to host */
    for (int i = 0; i < m_batchSize; i++)
        CUDA_CHECK(cudaMemcpyAsync(Q[i].data(), d_C[i], sizeof(float) * Q[i].size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < m_batchSize; i++){
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return elapsedTime;
}

float MLP_Linear_1(int input_dim, int output_dim){
    const int m = M;
    const int n = output_dim;
    const int k = input_dim;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const float alpha = 1.0;
    const float beta = 1.0;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    float** d_A_array = nullptr;
    float** d_B_array = nullptr;
    float** d_C_array = nullptr;

    vector<float*> d_A(batchSize, nullptr);
    vector<float*> d_B(batchSize, nullptr);
    vector<float*> d_C(batchSize, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * tensor[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(float) * w2[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(float) * b2[i].size()));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(float*) * batchSize));

    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], tensor[i].data(), sizeof(float) * tensor[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], w2[i].data(), sizeof(float) * w2[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], b2[i].data(), sizeof(float) * b2[i].size(), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda, d_B_array, ldb, &beta, d_C_array, ldc, batchSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* step 4: copy data to host */
    for (int i = 0; i < batchSize; i++)
        CUDA_CHECK(cudaMemcpyAsync(b2[i].data(), d_C[i], sizeof(float) * b2[i].size(), cudaMemcpyDeviceToHost, stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return elapsedTime;
}

__global__ void gelu(float* x, int n){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n)
        x[id] = 0.5 * x[id] * (1 + tanh(sqrt(2 / 3.1415926) + 0.004715 * pow(x[id], 3)));  
}

float GELU() {
    vector<float*> d_A(batchSize, nullptr);

    cudaStream_t stream = NULL;
    const int block = (M * N - 0.5) / thread + 1; 
    float elapsedTime = 0.0;

    /* copy data to device */
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * b2[i].size()));
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], b2[i].data(), sizeof(float) * QK[i].size(), cudaMemcpyHostToDevice, stream));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        gelu << <block, thread >> > (d_A[i], QK[i].size());

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float tmp;
        cudaEventElapsedTime(&tmp, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        elapsedTime += tmp;

        CUDA_CHECK(cudaMemcpyAsync(b2[i].data(), d_A[i], sizeof(float) * b1[i].size(), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    for (int i = 0; i < batchSize; i++)
        CUDA_CHECK(cudaFree(d_A[i]));

    return elapsedTime;
}

float MLP_Linear_2(int input_dim, int output_dim){
    const int m = M;
    const int n = output_dim;
    const int k = input_dim;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;

    const float alpha = 1.0;
    const float beta = 1.0;

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    float** d_A_array = nullptr;
    float** d_B_array = nullptr;
    float** d_C_array = nullptr;

    vector<float*> d_A(batchSize, nullptr);
    vector<float*> d_B(batchSize, nullptr);
    vector<float*> d_C(batchSize, nullptr);

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * b2[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B[i]), sizeof(float) * w3[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C[i]), sizeof(float) * b3[i].size()));
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B_array), sizeof(float*) * batchSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C_array), sizeof(float*) * batchSize));

    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], b2[i].data(), sizeof(float) * b2[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_B[i], w3[i].data(), sizeof(float) * w3[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_C[i], b3[i].data(), sizeof(float) * b3[i].size(), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaMemcpyAsync(d_A_array, d_A.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B_array, d_B.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C_array, d_C.data(), sizeof(float*) * batchSize, cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    /* step 3: compute */
    cublasSgemmBatched(cublasH, transa, transb, m, n, k, &alpha, d_A_array, lda, d_B_array, ldb, &beta, d_C_array, ldc, batchSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* step 4: copy data to host */
    for (int i = 0; i < batchSize; i++)
        CUDA_CHECK(cudaMemcpyAsync(b3[i].data(), d_C[i], sizeof(float) * b3[i].size(), cudaMemcpyDeviceToHost, stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A_array));
    CUDA_CHECK(cudaFree(d_B_array));
    CUDA_CHECK(cudaFree(d_C_array));
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_B[i]));
        CUDA_CHECK(cudaFree(d_C[i]));
    }

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return elapsedTime;
}


__global__ void var(float* x, int n, float avg){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n)
        x[id] = pow(x[id] - avg, 2);
}

__global__ void normalize(float* x, int n, float avg, float S){
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < n)
        x[id] = (x[id] - avg) / sqrt(S + 1e-5);
}

float LayerNorm(){
    vector<float*> d_A(batchSize, nullptr);
    vector<float*> d_A_copy(batchSize, nullptr);
    float sum = 0.0;
    float S = 0.0;
    cudaStream_t stream = NULL;

    cublasHandle_t handle;
    cublasCreate(&handle);

    const int token_size = M * N;
    const int block = (token_size - 0.5) / thread + 1;
    /* copy data to device */
    float elapsedTime = 0.0;
    for (int i = 0; i < batchSize; i++){
        sum = 0.0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A[i]), sizeof(float) * tensor[i].size()));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A_copy[i]), sizeof(float) * tensor[i].size()));
        CUDA_CHECK(cudaMemcpyAsync(d_A[i], tensor[i].data(), sizeof(float) * tensor[i].size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_A_copy[i], tensor[i].data(), sizeof(float) * tensor[i].size(), cudaMemcpyHostToDevice, stream));

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        cublasSasum(handle, token_size, d_A[i], 1, &sum);
        var << <block, thread >> > (d_A_copy[i], token_size, sum / (token_size));
        cublasSasum(handle, token_size, d_A_copy[i], 1, &S);
        normalize << <block, thread >> > (d_A[i], token_size, sum / (token_size), S);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float tmp;
        cudaEventElapsedTime(&tmp, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        elapsedTime += tmp;
        CUDA_CHECK(cudaMemcpyAsync(tensor[i].data(), d_A[i], sizeof(float) * tensor[i].size(), cudaMemcpyDeviceToHost, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    for (int i = 0; i < batchSize; i++){
        CUDA_CHECK(cudaFree(d_A[i]));
        CUDA_CHECK(cudaFree(d_A_copy[i]));
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return elapsedTime;
}

float Attention(){
    float time = 0.0;
    
    time += Att_Linear(N, 2304);
    time += MatrixMul_1();
    time += softmax();
    time += MatrixMul_2();

    return time;
}

float MLP(){
    float time = 0.0;

    time += MLP_Linear_1(N,3072);
    time += GELU();
    time += MLP_Linear_2(3072,N);
    tensor = b3;

    return time;
}

float Residual() {
    const int m = M;
    const int n = N;
    const int vector_size = batchSize * m * n;
    vector<float> A(vector_size);
    vector<float> B(vector_size);

    int t = 0;
    for (int b = 0; b < batchSize; b++){
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                A[t] = tensor[b][j * m + i];
                B[t] = tensor_copy[b][j * m + i];
                t++;
            }
        }
    }

    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const float alpha = 1.0;
    const int incx = 1;
    const int incy = 1;

    float* d_A = nullptr;
    float* d_B = nullptr;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(float) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(float) * B.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice, stream));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    /* step 3: compute */
    CUBLAS_CHECK(cublasSaxpy(cublasH, A.size(), &alpha, d_A, incx, d_B, incy));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(B.data(), d_B, sizeof(float) * B.size(), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));

    t = 0;
    for (int b = 0; b < batchSize; b++){
        for (int i = 0; i < m; i++){
            for (int j = 0; j < n; j++){
                tensor[b][j * m + i] = B[t];
                t++;
            }
        }
    }
    return elapsedTime;
}


int main(){
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    tensor_init(M, N);
    tensor_copy = tensor;
    linear_init();
    qkv_init(M, N / 8);

    for(int i = 0; i < 100; i++)
        LayerNorm();
    
    float GPU_time = 0.0;
    cudaEventRecord(start);
    GPU_time += LayerNorm();
    GPU_time += Attention();
    GPU_time += Residual();
    tensor_copy = tensor;

    GPU_time += LayerNorm();
    GPU_time += MLP();
    GPU_time += Residual();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("BatchSize:%d\n", batchSize);
    printf("Execution time:%.2fms\n", elapsedTime);
    printf("GPU kernel time:%.2fms\n", GPU_time);

    return 0;
}
