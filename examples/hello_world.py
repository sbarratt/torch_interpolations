import matplotlib.pyplot as plt
import torch_interpolations
import torch

points = [torch.arange(-10, 11, 1) * 1.]
values = torch.sin(points[0])
gi = torch_interpolations.RegularGridInterpolator(points, values)
points_to_interp = [torch.arange(-15, 15, .01) * 1.]
fx = gi(points_to_interp)
plt.plot(points_to_interp[0].numpy(), fx.numpy())
plt.scatter(points[0].numpy(), values.numpy())
plt.show()
