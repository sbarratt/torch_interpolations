import torch
import torch_interpolations
import numpy as np
import matplotlib.pyplot as plt

points = [torch.arange(-.5, 2.5, .2) * 1., torch.arange(-.5, 2.5, .2) * 1.]
values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
gi = torch_interpolations.RegularGridInterpolator(points, values)

X, Y = np.meshgrid(np.arange(-.5, 2.5, .02), np.arange(-.5, 2.5, .01))
points_to_interp = [torch.from_numpy(
    X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]
fx = gi(points_to_interp)
print(fx)

fig, axes = plt.subplots(1, 2)

axes[0].imshow(np.sin(X) + 2 * np.cos(Y) + np.sin(5 * X * Y))
axes[0].set_title("True")
axes[1].imshow(fx.numpy().reshape(X.shape))
axes[1].set_title("Interpolated")
plt.show()
