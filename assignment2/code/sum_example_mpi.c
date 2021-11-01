//
// Created by Georgy on 10/30/2021.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define N 100000

int main(int argc, char *argv[]){
    double sum, local_sum;
    double *a;
    int i, n=N;

    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = n / size;
    a = (double *) malloc(n);

    for (i=0; i<n; i++){
        a[i] = (rank * n + i) * 0.5;
    }
    local_sum = 0;
    for (i=0; i<n; i++)
        local_sum += a[i];

    free(a);
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (!rank)
        printf("Sum for MPI=%f\n", sum);
    MPI_Finalize();
    return 0;
}
