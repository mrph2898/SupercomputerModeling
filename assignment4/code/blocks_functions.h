#ifndef BLOCKS_FUNCTIONS_H
#define BLOCKS_FUNCTIONS_H

#define A1 0.0
#define A2 4.0
#define B1 0.0
#define B2 3.0

#define epsilon 1e-5
#define threadsPerBlock 4
#define numThreadsX 2
#define numThreadsY 2
#define EPS_REL  1e-6
#define DOWN_TAG 1000
#define MAX_ITER 100000


__global__ void init_w(int N, double **w);

__global__ void copy_interior_w(int M, int N,
                                double **w,
                                double **w_pr);

__global__ void get_top(int n_x, int n_y,
                        int x_idx, int y_idx,
                        double **w,
                        double *dev_t_send);

__global__ void get_bottom(int n_x, int n_y,
                           int x_idx, int y_idx,
                           double **w,
                           double *dev_b_send);

__global__ void get_left(int n_x, int n_y,
                         int x_idx, int y_idx,
                         double **w,
                         double *dev_l_send);

__global__ void get_right(int n_x, int n_y,
                          int x_idx, int y_idx,
                          double **w,
                          double *dev_r_send);

__global__ void set_top(int n_x, int n_y,
                        int x_idx, int y_idx,
                        double **w,
                        double *dev_t_recv);

__global__ void set_bottom(int n_x, int n_y,
                           int x_idx, int y_idx,
                           double **w,
                           double *dev_b_recv);

__global__ void set_left(int n_x, int n_y,
                         int x_idx, int y_idx,
                         double **w,
                         double *dev_l_recv);

__global__ void set_right(int n_x, int n_y,
                          int x_idx, int y_idx,
                          double **w,
                          double *dev_r_recv);


__global__ void preset_top(int n_x, int n_y,
                        int x_idx, int y_idx,
                        double **w,
                        double h1, double h2);

__global__ void preset_bottom(int n_x, int n_y,
                           int x_idx, int y_idx,
                           double **w,
                           double h1, double h2);

__global__ void preset_left(int n_x, int n_y,
                         int x_idx, int y_idx,
                         double **w,
                         double h1, double h2);

__global__ void preset_right(int n_x, int n_y,
                          int x_idx, int y_idx,
                          double **w,
                          double h1, double h2);


__global__ void cudaB_right(int M, int N, double **B,
                            int x_idx, int y_idx,
                            double h1, double h2,
                            double x_start, double y_start,
                            int left_border, int right_border,
                            int top_border, int bottom_border);

__global__ void cuda_Aw_mult(int M, int N,
                             int x_idx, int y_idx,
                             double **A, double **w,
                             double h1, double h2,
                             double x_start, double y_start,
                             int left_border, int right_border,
                             int top_border, int bottom_border);

__global__ void calculate_r(int M, int N,
                            int x_idx, int y_idx,
                            double **r,
                            double **Aw,
                            double **B);

__global__ void cuda_dot_product(int n_x, int n_y,
                                 int x_idx, int y_idx,
                                 double **U, double **V,
                                 double h1, double h2,
                                 int left_border, int right_border,
                                 int top_border, int bottom_border,
                                 double *partial_product);
__global__ void cuda_w_step(int n_y,
                            int x_idx, int y_idx,
                            double **w,
                            double **r_k,
                            double tau
//                            double *w_next
                            );
#endif //BLOCKS_FUNCTIONS_H
