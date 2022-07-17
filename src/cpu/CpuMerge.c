#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>


void assertfy(const int *buffer, const int len);
void printUpTo(int *buffer, const int len);
void fillAscending(int *buffer, const int len);

int co_rank(int k, int * A, int m, int * B, int n);
void merge(const int *A, int m, const int *B, int n, int *C);


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

void fillAscending(int *buffer, const int len)
{
    int i = 0;

    for (; i < len; i++)
    {
        buffer[i] = i;
    }
}


void merge(const int *A, int m, const int *B, int n, int *C)
{
    int i = 0,
        j = 0,
        k = 0;
    
    while ((i < m) && (j < n))
    {
        if (A[i] <= B[j])
        {
            C[k++] = A[i++];
        }
        else
        {
            C[k++] = B[j++];
        }
    }

    for (; j < n; j++)
    {
        C[k++] = B[j];
    }

    for (; i < m; i++)
    {
        C[k++] = A[i];
    }
}


int main(int argc, char const *argv[])
{
    int m = 1 << 24,
        n = 1 << 24;

    int *A,
        *B,
        *C;

    float time;

    A = malloc(m * sizeof(int));
    B = malloc(n * sizeof(int));
    C = malloc((m + n) * sizeof(int));

    fillAscending(A, m);
    fillAscending(B, n);

    time = -clock();
    merge(A, m, B, n, C);
    time += clock();

    time /= (CLOCKS_PER_SEC / 1000);

    printf("[*] Completed, Elapsed Time %1.3f ms.\n", time);

    assertfy(C, m + n);
    printUpTo(C, 10);

    free(A);
    free(B);
    free(C);

    return 0;
}
