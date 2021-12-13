from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot solution')
    parser.add_argument('path', help='The path to the root directory')
    args = parser.parse_args()
    x, y = np.meshgrid(np.loadtxt(os.path.join(args.path, 'dim1.csv')),
                       np.loadtxt(os.path.join(args.path, 'dim0.csv')), indexing='ij')
    grid_shape = tuple(np.loadtxt(os.path.join(args.path, 'grid.csv'), dtype=int))
    print('grid={}'.format(grid_shape))
    print([
        [np.loadtxt(os.path.join(args.path, os.path.join(args.path, f'u_{i}_{j}.csv'))).shape for j in range(grid_shape[1])]
        for i in range(grid_shape[0])
    ])
    z = np.concatenate([
        np.concatenate([np.loadtxt(os.path.join(args.path, f'u_{j}_{i}.csv')) for j in range(grid_shape[1])], axis=1)
        for i in range(grid_shape[0])
    ], axis=0).T

    z_true = np.concatenate([
        np.concatenate([np.loadtxt(os.path.join(args.path, f'true.u_{j}_{i}.csv')) for j in range(grid_shape[1])], axis=1)
        for i in range(grid_shape[0])
    ], axis=0).T

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    print(x.shape, y.shape, z.shape)
    print(z_true.shape)
    ax.plot_surface(x, y, z, linewidth=0.2, antialiased=True, color='blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Численное u(x, y)', fontsize=16)

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.plot_surface(x, y, z_true, linewidth=0.2, antialiased=True, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u(x, y)')
    ax.set_title('Истинное u(x, y)', fontsize=16)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(os.path.join(args.path, 'u.png'))

    print('shape={} relative diff = {:.3f}'.format(z.shape, abs(z - z_true).max()/abs(z_true).max()))