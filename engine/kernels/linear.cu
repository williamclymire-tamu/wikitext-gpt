#include "linear.cuh"

static cublasHandle_t g_handle = nullptr;

void cublas_init() {
    if (!g_handle) {
        cublasStatus_t st = cublasCreate(&g_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS init failed: %d\n", st);
            exit(1);
        }
    }
}

void cublas_destroy() {
    if (g_handle) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

// add bias[N] to every row of Y[M, N]
__global__ void bias_add_kernel(float* __restrict__ Y, const float* __restrict__ bias,
                                int M, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int col = idx % N;
    Y[idx] += bias[col];
}

void linear_cuda(
    const float* X, const float* W, const float* bias, float* Y,
    int M, int N, int K)
{
    // Row-major: Y[M,N] = X[M,K] @ W^T[K,N]   where W is stored [N,K]
    //
    // cuBLAS is column-major. Row-major A[r,c] in memory = col-major A'[c,r].
    // So: X' = [K,M], W' = [K,N], Y' = [N,M]
    // We need Y' = W'^T @ X' = [N,K] @ [K,M] = [N,M]  ✓
    //
    // cublasSgemm(op_A, op_B, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    //   computes C[m,n] = alpha * op(A)[m,k] @ op(B)[k,n] + beta * C   (col-major)
    //
    // m=N, n=M, k=K
    // op(A) = W'^T → A = W', opA = CUBLAS_OP_T, lda = K
    // op(B) = X'   → B = X', opB = CUBLAS_OP_N, ldb = K
    // C = Y', ldc = N

    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st = cublasSgemm(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        W, K,    // W'[K,N] col-major, transposed to [N,K]
        X, K,    // X'[K,M] col-major
        &beta,
        Y, N);   // Y'[N,M] col-major = Y[M,N] row-major

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed: %d\n", st);
        exit(1);
    }

    if (bias) {
        int total = M * N;
        int threads = 256;
        bias_add_kernel<<<cdiv(total, threads), threads>>>(Y, bias, M, N);
        CUDA_CHECK(cudaGetLastError());
    }
}
