//
// Created by Georgy on 10/30/2021.
//

#include <stdio.h>
#define N 100000

int main(){
    double sum;
    double a[N];
    int i, n = N;

    for (i=0; i<n; i++){
        a[i] = i*0.5;
    }

    sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (i=0; i<n; i++)
        sum = sum + a[i];
    printf("SUM=%f\n", sum);
}