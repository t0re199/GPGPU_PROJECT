#include <cuda.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#define min(v1, v2) v1 < v2 ? v1 : v2

#define THREADS 512
#define BLOCKS 32
#define TILE_SIZE 1024

#define LEN 1024

void assertfy(const int *buffer, const int len);
void printUpTo(int *buffer, const int len);

void assertfy(const int *buffer, const int len)
{
    int i = 1;

    for (; i < len; i++)
    {
        assert(buffer[i] >= buffer[i - 1]);
    }
}

void printUpTo(int *buffer, const int len)
{
    int i = 0;
    for (; i < len; i++)
    {
        printf("%d\t", buffer[i]);
    }
    printf("\n");
}

__global__ void fillAscending(int *buffer, const int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len)
    {
        buffer[i] = i;
    }
}

__host__ __device__ int coRank(int k, int *A, int m, int *B, int n)
{
    int i = (k < m) ? k : m;
    int j = k - i;
    int i_low = (0 > (k - n)) ? 0 : k - n;
    int j_low = (0 > (k - m)) ? 0 : k - m;
    int delta;
    bool active = true;

    while (active)
    {
        if (i > 0 && j < n && A[i - 1] > B[j])
        {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j - 1] >= A[i])
        {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else
        {
            active = false;
        }
    }

    return i;
}

__host__ __device__ 
int circularCoRank(int k, int *A, int m, int *B, int n, int A_S_start, int B_S_start, int tile_size)
{
    int i = (k < m) ? k : m;
    int j = k - i;
    int i_low = (0 > (k - n)) ? 0 : k - n;
    int j_low = (0 > (k - m)) ? 0 : k - m;
    int delta;
    bool active = true;

    while (active)
    {
        int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size : A_S_start + i;
        int i_m_1_cir = (A_S_start + i - 1 >= tile_size) ? A_S_start + i - 1 - tile_size : A_S_start + i - 1;
        int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size : B_S_start + j;
        int j_m_1_cir = (B_S_start + j - 1 >= tile_size) ? B_S_start + j - 1 - tile_size : B_S_start + j - 1;

        if (i > 0 && j < n && A[i_m_1_cir] > B[j_cir])
        {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j_m_1_cir] >= A[i_cir])
        {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else
        {
            active = false;
        }
    }
    return i;
}


__host__ __device__
void circularSequentialMerge(int *A, int m, int *B, int n, int *C, int A_S_start, int B_S_start, int tile_size)
{
    int i = 0; 
    int j = 0; 
    int k = 0;
    while ((i < m) && (j < n))
    {
        int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size : A_S_start + i;
        int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size : B_S_start + j;
        if (A[i_cir] <= B[j_cir])
        {
            C[k++] = A[i_cir];
            i++;
        }
        else
        {
            C[k++] = B[j_cir];
            j++;
        }
    }
    if (i == m)
    {
        while (j < n)
        {
            int j_cir = (B_S_start + j >= tile_size) ? B_S_start + j - tile_size : B_S_start + j;
            C[k++] = B[j_cir];
            j++;
        }
    }
    else
    {
        while (i < m)
        {
            int i_cir = (A_S_start + i >= tile_size) ? A_S_start + i - tile_size : A_S_start + i;
            C[k++] = A[i_cir];
            i++;
        }
    }
}


__global__ void circularMergeKernel(int *A, int m, int *B, int n, int *C, int tile_size)
{
    extern __shared__ int shareAB[];
    int *A_S = shareAB;
    int *B_S = shareAB + tile_size;
    int C_curr = blockIdx.x * ceil((m + n) / (float)gridDim.x);
    int C_next = min((blockIdx.x + 1) * (int)ceil((m + n) / (float)gridDim.x), m + n);

    if (threadIdx.x == 0)
    {
        A_S[0] = coRank(C_curr, A, m, B, n);
        A_S[1] = coRank(C_next, A, m, B, n);
    }
    __syncthreads();

    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int counter = 0;
    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = ceil((C_length) / (float)tile_size);
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    int A_S_start = 0;
    int B_S_start = 0;
    int A_S_consumed = tile_size;
    int B_S_consumed = tile_size;

    while (counter < total_iteration)
    {
        for (int i = 0; i < A_S_consumed; i += blockDim.x)
        {
            if (i + threadIdx.x < A_length - A_consumed && i + threadIdx.x < A_S_consumed)
                A_S[(A_S_start + i + threadIdx.x) % tile_size] = A[A_curr + A_consumed + i + threadIdx.x];
        }

        for (int i = 0; i < B_S_consumed; i += blockDim.x)
        {
            if (i + threadIdx.x < B_length - B_consumed && i + threadIdx.x < B_S_consumed)
                B_S[(B_S_start + i + threadIdx.x) % tile_size] = B[B_curr + B_consumed + i + threadIdx.x];
        }
        __syncthreads();

        
        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;

        int a_curr = circularCoRank(c_curr,
                                    A_S, min(tile_size, A_length - A_consumed),
                                    B_S, min(tile_size, B_length - B_consumed),
                                    A_S_start, B_S_start, tile_size);
        int b_curr = c_curr - a_curr;
        int a_next = circularCoRank(c_next,
                                    A_S, min(tile_size, A_length - A_consumed),
                                    B_S, min(tile_size, B_length - B_consumed),
                                    A_S_start, B_S_start, tile_size);
        int b_next = c_next - a_next;

        circularSequentialMerge(A_S, a_next - a_curr,
                                B_S, b_next - b_curr,
                                C + C_curr + C_completed + c_curr,
                                A_S_start + a_curr, B_S_start + b_curr, tile_size);
        counter++;
        A_S_consumed = circularCoRank(min(tile_size, C_length - C_completed),
                                      A_S, min(tile_size, A_length - A_consumed),
                                      B_S, min(tile_size, B_length - B_consumed),
                                      A_S_start, B_S_start, tile_size);
                                      
        B_S_consumed = min(tile_size, C_length - C_completed) - A_S_consumed;
        A_consumed += A_S_consumed;
        C_completed += min(tile_size, C_length - C_completed);
        B_consumed = C_completed - A_consumed;

        A_S_start = A_S_start + A_S_consumed;
        if (A_S_start >= tile_size)
            A_S_start = A_S_start - tile_size;
        B_S_start = B_S_start + B_S_consumed;
        if (B_S_start >= tile_size)
            B_S_start = B_S_start - tile_size;

        __syncthreads();
    }
}

int main(int argc, char const *argv[])
{
    int m = 1 << 24,
        n = 1 << 24;

    int *A,
        *B,
        *C;

    cudaMallocManaged((void **)&A, m * sizeof(int), cudaMemAttachGlobal);
    cudaMallocManaged((void **)&B, n * sizeof(int), cudaMemAttachGlobal);

    cudaMallocManaged((void **)&C, (m + n) * sizeof(int), cudaMemAttachGlobal);

    fillAscending<<<ceil((double)m / THREADS), THREADS>>>(A, m);
    fillAscending<<<ceil((double)n / THREADS), THREADS>>>(B, n);

    circularMergeKernel<<<BLOCKS, THREADS, TILE_SIZE * 2 * sizeof(int)>>>(A, m, B, n, C, TILE_SIZE);

    cudaDeviceSynchronize();

    assertfy(C, LEN);
    printUpTo(C, 10);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
