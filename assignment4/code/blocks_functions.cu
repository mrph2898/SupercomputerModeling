#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "blocks_functions.h"


__device__ double dev_u_2(double x, double y){
    return sqrt(4 + x * y);
}

__device__ double dev_k_3(double x, double y){
    return 4 + x + y;
}

__device__ double dev_q_2(double x, double y){
    double sum = x + y;
    if (sum < 0) {
        return 0;
    } else {
        return sum;
    }
}

__device__ double dev_F(double x, double y){
    return ((pow(x, 3) - x*x*(y - 4) - x*(y*y + 8) +
             y*(y*y + 4*y - 8) + 4*dev_q_2(x, y)*pow((4 + x*y), 2)) /
            (4 * pow((4 + x*y), 1.5)));
}

__device__ double dev_psi_R(double x, double y){
    return (y*(4 + x + y) + 2*(4 + x*y)) / (2*sqrt(4 + x*y));
}

__device__ double dev_psi_L(double x, double y){
    return (-y*(4 + x + y) + 2*(4 + x*y)) / (2*sqrt(4 + x*y));
}


__device__ double dev_psi_T(double x, double y){
    return (x*(4 + x + y)) / (2*sqrt(4 + x*y));
}


__device__ double dev_psi_B(double x, double y){
    return -dev_psi_T(x, y);
}

__global__ void init_w(int n_y, double **w){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    w[tid_x][tid_y] = 0.0;
    return;
}

__global__ void copy_interior_w(int M, int N,
                                double **w,
                                double **w_pr){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    if (tid_x == 0 || tid_y == 0 || tid_x == M + 1 || tid_y == N + 1) {
        w_pr[tid_x][tid_y] = 0.0;
    } else {
        w_pr[tid_x][tid_y] = w[tid_x][tid_y];
    }
    return;
}

__global__ void get_top(int n_x, int n_y,
                        int x_idx, int y_idx,
                        double **w,
                       double *dev_t_send){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    if (tid_y == (y_idx + n_y - 1))
        dev_t_send[i] = w[i+1][n_y];
}

__global__ void get_bottom(int n_x, int n_y,
                           int x_idx, int y_idx,
                           double **w,
                           double *dev_b_send){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    if (tid_y == y_idx)
        dev_b_send[i] = w[i+1][1];
}
__global__ void get_left(int n_x, int n_y,
                         int x_idx, int y_idx,
                         double **w,
                         double *dev_l_send){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int j = tid_y - y_idx;
    if (tid_x == x_idx)
        dev_l_send[j] = w[1][j+1];
}
__global__ void get_right(int n_x, int n_y,
                          int x_idx, int y_idx,
                          double **w,
                          double *dev_r_send){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int j = tid_y - y_idx;
    if (tid_x == (x_idx + n_x - 1))
        dev_r_send[j] = w[n_x][j+1];
}

__global__ void set_top(int n_x, int n_y,
                           int x_idx, int y_idx,
                           double **w,
                           double *dev_t_recv)
                           {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    if (tid_y == (y_idx + n_y - 1))
        w[i][n_y + 1] = dev_t_recv[i - 1];
}

__global__ void set_bottom(int n_x, int n_y,
                              int x_idx, int y_idx,
                              double **w,
                              double *dev_b_recv
){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    if (tid_y == y_idx)
        w[i][0] = dev_b_recv[i-1];
}
__global__ void set_left(int n_x, int n_y,
                            int x_idx, int y_idx,
                            double **w,
                            double *dev_l_recv
){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int j = tid_y - y_idx;
    if (tid_x == x_idx)
        w[0][j] = dev_l_recv[j-1];
}
__global__ void set_right(int n_x, int n_y,
                             int x_idx, int y_idx,
                             double **w,
                             double *dev_r_recv
){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int j = tid_y - y_idx;
    if (tid_x == (x_idx + n_x - 1))
        w[n_x+1][j] = dev_r_recv[j - 1];
}


__global__ void preset_top(int n_x, int n_y,
                        int x_idx, int y_idx,
                        double **w,
                        double h1, double h2
                        ){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    if (tid_y == (y_idx + n_y - 1))
        w[i][n_y + 1] = dev_u_2(A1 + (x_idx + i - 1)*h1,
                                B1 + (y_idx + n_y) * h2);
}

__global__ void preset_bottom(int n_x, int n_y,
                           int x_idx, int y_idx,
                           double **w,
                           double h1, double h2
                           ){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    if (tid_y == y_idx)
        w[i][0] = dev_u_2(A1 + (x_idx + i - 1)*h1,
                                B1 + (y_idx - 1) * h2);
}
__global__ void preset_left(int n_x, int n_y,
                         int x_idx, int y_idx,
                         double **w,
                         double h1, double h2
                         ){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int j = tid_y - y_idx;
    if (tid_x == x_idx)
        w[0][j] = dev_u_2(A1 + (x_idx - 1)*h1,
                          B1 + (y_idx + j - 1) * h2);
}
__global__ void preset_right(int n_x, int n_y,
                          int x_idx, int y_idx,
                          double **w,
                          double h1, double h2
                          ){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int j = tid_y - y_idx;
    if (tid_x == (x_idx + n_x - 1))
        w[n_x+1][j] = dev_u_2(A1 + (x_idx + n_x)*h1,
                              B1 + (y_idx + j - 1) * h2);
}


__global__ void cudaB_right(int M, int N, double **B,
                        int x_idx, int y_idx,
                        double h1, double h2,
                        double x_start, double y_start,
                        int left_border, int right_border,
                        int top_border, int bottom_border){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    int j = tid_y - y_idx;

    B[i][j] = dev_F(x_start + (i - 1) * h1, y_start + (j - 1) * h2);

    if (left_border){
        B[1][j] = (dev_F(x_start, y_start + (j - 1) * h2) +
                dev_psi_L(x_start, y_start + (j - 1) * h2) * 2/h1);
    } else if (right_border){
        B[M][j] = (dev_F(x_start + (M - 1)*h1, y_start + (j - 1) * h2) +
                dev_psi_R(x_start + (M - 1)*h1, y_start + (j - 1) * h2) * 2/h1);
    }
    if (top_border){
        B[i][N] = (dev_F(x_start + (i - 1)*h1, y_start + (N - 1)*h2) +
                dev_psi_T(x_start + (i - 1)*h1, y_start + (N - 1)*h2) * 2/h2);
    } else if (bottom_border){
        B[i][1] = (dev_F(x_start + (i - 1)*h1, y_start) +
                dev_psi_B(x_start + (i - 1)*h1, y_start) * 2/h2);
    }
    if (left_border && top_border){
        B[1][N] = (dev_F(x_start, y_start + (N - 1)*h2) +
        (2/h1 + 2/h2) * (dev_psi_L(x_start, y_start + (N - 1)*h2) +
        dev_psi_T(x_start, y_start + (N - 1)*h2)) / 2);
    } else if (left_border && bottom_border){
        B[1][1] =  (dev_F(x_start, y_start)
        + (2/h1 + 2/h2) * (dev_psi_L(x_start, y_start) + dev_psi_B(x_start, y_start)) / 2);
    } else if (right_border && top_border){
        B[M][N] = (dev_F(x_start + (M - 1)*h1, y_start + (N - 1)*h2) +
        (2/h1 + 2/h2) * (dev_psi_R(x_start + (M - 1)*h1, y_start + (N - 1)*h2) +
        dev_psi_T(x_start + (M - 1)*h1, y_start + (N - 1)*h2)) / 2);
    } else if (right_border && bottom_border){
        B[M][1] = (dev_F(x_start + (M - 1)*h1, y_start) +
        (2/h1 + 2/h2) * (dev_psi_R(x_start + (M - 1)*h1, y_start) +
        dev_psi_B(x_start + (M - 1)*h1, y_start)) / 2);
    }
}

__device__ double dev_aw_x_ij(int N,
                              double **w,
                              double x_start, double y_start,
                              int i, int j,
                              double h1, double h2
                              ){
    return (1/h1) * (dev_k_3(x_start + (i + 0.5 - 1) * h1,y_start + (j - 1) * h2) * (w[i + 1][j] - w[i][j]) / h1
    - dev_k_3(x_start + (i - 0.5 - 1) * h1,y_start + (j - 1) * h2) * (w[i][j] - w[i - 1][j]) / h1);
}

__device__ double dev_aw_ij(int N,
                            double **w,
                            double x_start, double y_start,
                            int i, int j,
                            double h1, double h2
                            ){
    return (dev_k_3(x_start + (i - 0.5 - 1) * h1,y_start + (j - 1) * h2) * (w[i][j] - w[i - 1][j]) / h1);
}

__device__ double dev_bw_y_ij(int N,
               double **w,
               double x_start, double y_start,
               int i, int j,
               double h1, double h2
               ){
    return (1/h2) * (dev_k_3(x_start + (i - 1) * h1,y_start + (j + 0.5 - 1) * h2) * (w[i][j + 1] - w[i][j]) / h2
    - dev_k_3(x_start + (i - 1) * h1,y_start + (j - 0.5 - 1) * h2) * (w[i][j] - w[i][j - 1]) / h2);
}

__device__ double dev_bw_ij(int N,
             double **w,
             double x_start, double y_start,
             int i, int j,
             double h1, double h2
             ){
    return (dev_k_3(x_start + (i - 1) * h1,y_start + (j - 0.5 - 1) * h2) * (w[i][j] - w[i][j-1]) / h2);
}

__global__ void cuda_Aw_mult(int M, int N,
                        int x_idx, int y_idx,
                        double **A, double **w,
                        double h1, double h2,
                        double x_start, double y_start,
                        int left_border, int right_border,
                        int top_border, int bottom_border
                        ) {
    double aw_x, bw_y;
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    int j = tid_y - y_idx;

    if (( i == 0) || i == M+1 || j == 0 || j == N+1){
        A[i][j] = w[i][j];
    } else {
        aw_x = dev_aw_x_ij(N, w, x_start, y_start, i, j, h1, h2);
        bw_y = dev_bw_y_ij(N, w, x_start, y_start, i, j, h1, h2);
        A[i][j] = -aw_x - bw_y + dev_q_2(x_start + (i - 1) * h1,
                                     y_start + (j - 1) * h2) * w[i][j];
    }

    // Left interior border filling
    if (left_border){
        aw_x = dev_aw_ij(N, w, x_start, y_start, 2, j , h1, h2);
        bw_y = dev_bw_y_ij(N, w, x_start, y_start, 1, j, h1, h2);
        A[1][j] = -2*aw_x / h1 - bw_y + (dev_q_2(x_start, y_start + (j - 1) * h2)
                + 2/h1) * w[1][j];
    } else if (right_border){
    // Right interior border
        aw_x = dev_aw_ij(N, w, x_start, y_start, M, j, h1, h2);
        bw_y = dev_bw_y_ij(N, w, x_start, y_start, M, j, h1, h2);
        A[M][j] = 2*aw_x / h1 - bw_y + (dev_q_2(x_start + (M - 1) * h1,
                                            y_start + (j - 1) * h2) + 2/h1) * w[M][j];
    }

    // Top border
    if (top_border){
        aw_x = dev_aw_x_ij(N, w, x_start, y_start, i, N, h1, h2);
        bw_y = dev_bw_ij(N, w, x_start, y_start, i, N, h1, h2);
        A[i][N] = -aw_x + 2*bw_y / h2 + dev_q_2(x_start + (i - 1) * h1,
                                            y_start + (N - 1) * h2) * w[i][N];
    } else if (bottom_border){
    // Bottom border
        aw_x = dev_aw_x_ij(N, w, x_start, y_start, i, 1, h1, h2);
        bw_y = dev_bw_ij(N, w, x_start, y_start, i, 2, h1, h2);
        A[i][1] = -aw_x - 2*bw_y / h2 + dev_q_2(x_start + (i - 1)* h1, y_start) * w[i][1];
    }
    if (left_border && bottom_border){
        aw_x = dev_aw_ij(N, w, x_start, y_start, 2, 1, h1, h2);
        bw_y = dev_bw_ij(N, w, x_start, y_start, 1, 2, h1, h2);
        A[1][1] = -2*aw_x / h1 - 2*bw_y / h2 + (dev_q_2(x_start, y_start) + 2/h1) * w[1][1];
    } else if (left_border && top_border){
        aw_x = dev_aw_ij(N, w, x_start, y_start, 2, N, h1, h2);
        bw_y = dev_bw_ij(N, w, x_start, y_start, 1, N, h1, h2);
        A[1][N] = -2*aw_x / h1 + 2*bw_y / h2 + (dev_q_2(x_start, y_start + (N - 1) * h2) + 2/h1)* w[1][N];
    }
    if (right_border && bottom_border){
        aw_x = dev_aw_ij(N, w, x_start, y_start, M, 1, h1, h2);
        bw_y = dev_bw_ij(N, w, x_start, y_start, M, 2, h1, h2);
        A[M][1] = 2*aw_x / h1 - 2 * bw_y / h2 + (dev_q_2(x_start + (M - 1) * h1, y_start) + 2/h1) * w[M][1];
    } else if (right_border && top_border) {
        aw_x = dev_aw_ij(N, w, x_start, y_start, M, N, h1, h2);
        bw_y = dev_bw_ij(N, w, x_start, y_start, M, N, h1, h2);
        A[M][N] = 2*aw_x / h1 + 2 * bw_y / h2 + (dev_q_2(x_start + (M - 1) * h1,
                                                     y_start + (N - 1) * h2) + 2/h1) * w[M][N];
    }
}

__global__ void calculate_r(int M, int N,
                 int x_idx, int y_idx,
                 double **r,
                 double **Aw,
                 double **B){
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    int j = tid_y - y_idx;

    if(i == 0 || i == M+1 || j == 0 || j == N+1)
        r[i][j] = 0;
    else
        r[i][j] = Aw[i][j] - B[i][j];
}


__device__ double dev_rho_1(int i,
                        int M,
                        int left_border,
                        int right_border){
    if ((left_border && i == 1) || (right_border && i == M))
        return 0.5;
    return 1;
}

__device__ double dev_rho_2(int j,
                        int N,
                        int bottom_border,
                        int top_border){
    if ((bottom_border && j == 1) || (top_border && j == N))
    return   0.5;
    return 1;
}

__global__ void cuda_dot_product(int n_x, int n_y,
                                 int x_idx, int y_idx,
                                 double **U, double **V,
                                 double h1, double h2,
                                 int left_border, int right_border,
                                 int top_border, int bottom_border,
                                 double *partial_product
                                 ){
//    int num_threads_x = (int) sqrt(threadsPerBlock);
//    int num_threads_y = threadsPerBlock / numThreadsX;
    __shared__ double cache[numThreadsX];
//    __shared__ double cache_y[numThreadsY];


    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    int j = tid_y - y_idx;

    int cacheIndex_x = threadIdx.x;
//    int cacheIndex_y = threadIdx.y;
    double temp = 0;
    double rho, r1, r2;

    while (i < n_x) {
        j = tid_y - y_idx;
        while (j < n_y) {
            r1 = dev_rho_1(i, n_x, left_border, right_border);
            r2 = dev_rho_2(j, n_y, bottom_border, top_border);
            rho = r1 * r2;
            double part_dot = (rho * U[i][j] * V[i][j] * h1 * h2);
            temp += part_dot;
            j += blockDim.y * gridDim.y;
        }
        i += blockDim.x * gridDim.x;
    }
    cache[cacheIndex_x] = temp;
    __syncthreads();
    int k = blockDim.x / 2;
    while (k > 0) {
        if (cacheIndex_x < k) {
            cache[cacheIndex_x] += cache[cacheIndex_x + k];
        }
        __syncthreads();
        k = k / 2;
    }
    if (cacheIndex_x == 0) {
        partial_product[blockIdx.x] = cache[0];
    }
    return;
}

__global__ void cuda_w_step(int n_y,
                            int x_idx, int y_idx,
                            double **w,
                            double **r_k,
                            double tau
//                            double *w_next
                            ) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    int i = tid_x - x_idx;
    int j = tid_y - y_idx;
    double r_k_scaled = r_k[i][j] * tau;
    w[i][j] = w[i][j] - r_k_scaled;
    return;
}