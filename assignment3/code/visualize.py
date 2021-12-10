from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

x, y = np.meshgrid(np.loadtxt('dim1.csv'), np.loadtxt('dim0.csv'), indexing='ij')
grid_shape = tuple(np.loadtxt('grid.csv', dtype=int))
print('grid={}'.format(grid_shape))
print([
    [np.loadtxt(f'u_{i}_{j}.csv').shape for j in range(grid_shape[1])]
    for i in range(grid_shape[0])
])
z = np.concatenate([
    np.concatenate([np.loadtxt(f'u_{j}_{i}.csv') for j in range(grid_shape[1])], axis=1)
    for i in range(grid_shape[0])
], axis=0).T

z_true = np.concatenate([
    np.concatenate([np.loadtxt(f'true.u_{j}_{i}.csv') for j in range(grid_shape[1])], axis=1)
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
plt.savefig('u.png')

print('shape={} relative diff = {:.3f}'.format(z.shape, abs(z - z_true).max()/abs(z_true).max()))