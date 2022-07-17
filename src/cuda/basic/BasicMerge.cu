#include <cuda.h>
#include <stdbool.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>


#define THREADS 512
#define BLOCKS 32

#define min(v1, v2) v1 < v2 ? v1 : v2


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


__global__ 
void fillAscending(int *buffer, const int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len)
    {
        buffer[i] = i;
    }
}

__host__ __device__
int coRank(int k, int* A, int m, int* B, int n)
{
    int i = (k < m) ? k : m;
    int j = k-i;
    int i_low = (0 > (k-n)) ? 0 : k-n; // i_low = max(0, k-n);
    int j_low = (0 > (k-m)) ? 0 : k-m; // j_low = max(0, k-m);
    int delta;
    bool active = true;

    while (active) {
        if (i > 0 && j < n && A[i-1] > B[j]) {
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }
        else if (j > 0 && i < m && B[j-1] >= A[i]) {
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }
        else {
            active = false;
        }
    }

    return i;
}


__device__
void merge(const int *buff1, int m, const int *buff2, int n, int *acc)
{
    int i = 0x0,
        j = 0x0,
        k = 0x0;
    
    while ((i < m) && (j < n))
    {
        if (buff1[i] <= buff2[j])
        {
            acc[k++] = buff1[i++];
        }
        else
        {
            acc[k++] = buff2[j++];
        }
    }
    
    for (; i < m; i++)
    {
        acc[k++] = buff1[i];
    }

    for (; j < n; j++)
    {
        acc[k++] = buff2[j];
    }
}


__global__
void basicMergeKernel(int* A, int m, int* B, int n, int* C)
{
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int k_curr = tid * ceil((m+n)/(float)(blockDim.x*gridDim.x));
    int k_next = min((tid+1) * (int)ceil((m+n)/(float)(blockDim.x*gridDim.x)), m+n);
    int i_curr = coRank(k_curr, A, m, B, n);
    int i_next = coRank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge(A+i_curr, i_next-i_curr, B+j_curr, j_next-j_curr, C+k_curr);
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

    cudaMallocManaged((void **)&C, (n + m) * sizeof(int), cudaMemAttachGlobal);

    fillAscending<<<ceil((double) m / THREADS), THREADS>>>(A, m);
    fillAscending<<<ceil((double) n / THREADS), THREADS>>>(B, n);

    mergeKernel<<<BLOCKS, THREADS>>>(A, m, B, n, C);

    cudaDeviceSynchronize();

    assertfy(C, m + n);
    printUpTo(C, 10);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
