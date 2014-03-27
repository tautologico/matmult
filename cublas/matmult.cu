// Matrix multiplication benchmark using CUDA+CUBLAS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1200

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

float* gen_mat(int m, int n, float start, float inc)
{
    float *res = malloc(m * n * sizeof(float));
    float acc = start;

    if (res == NULL)
        return NULL;

    for (int j = 0; j <= n; ++j) {
        for (int i = 0; i <= m; ++i) {
            res[IDX2C(i, j, m)] = acc;
            acc += inc;
        }
    }

    return res;
}

void bench_matrix_mul(int n)
{
    float *b, *devPtrB;
    float *a, *devPtrA;
    float *devPtrC;
    float alpha = 1.0f;
    float beta  = 0.0f;

    a = gen_mat(n, n, 0.0, 0.5);
    if (a == NULL) {
        printf("host memory allocation failed");
        return EXIT_FAILURE;
    }

    b = gen_mat(n, n, 4.25, 0.25);
    if (b == NULL) {
        printf("host memory allocation failed");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&devPtrA, n * n * sizeof(*a));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed for A");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&devPtrB, n * n * sizeof(*b));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed for B");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc ((void**)&devPtrC, n * n * sizeof(float));
    if (cudaStat != cudaSuccess) {
        printf("device memory allocation failed for C");
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix (n, n, sizeof(*a), a, n, devPtrA, n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed for matrix A");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    stat = cublasSetMatrix (n, n, sizeof(*b), b, n, devPtrB, n);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("data download failed for matrix B");
        cudaFree (devPtrA);
        cudaFree (devPtrB);
        cudaFree (devPtrC);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    // measure time
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, 
                       &alpha, devPtrA, n, devPtrB, n, &beta, devPtrC, n);
    // measure time
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("Error with cublasSgemm");
    }

    // getmatrix?

    free(a);
    free(b);
    cudaFree (devPtrA);
    cudaFree (devPtrB);
    cudaFree (devPtrC);
}

int main (void)
{
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    int i, j;
    float *b, *devPtrB;
    float *a, *devPtrA;

    if (cudaSetDevice(0) != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device\n");
        return EXIT_FAILURE;
    }

    cudaDeviceSynchronize();
    cudaThreadSynchronize();

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    bench_matrix_mul(1200);

    cublasDestroy(handle);

    return EXIT_SUCCESS;
}
