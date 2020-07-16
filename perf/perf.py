import torch
import torch_interpolations
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import time


def time_function(f, n=10):
    times = []
    for _ in range(n):
        tic = time.time()
        f()
        toc = time.time()
        times.append(1000 * (toc - tic))
    return times

points = [torch.arange(-.5, 2.5, .01) * 1., torch.arange(-.5, 2.5, .01) * 1.]
values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
gi = torch_interpolations.RegularGridInterpolator(points, values)

X, Y = np.meshgrid(np.arange(-.5, 2.5, .002), np.arange(-.5, 2.5, .001))
points_to_interp = [torch.from_numpy(
    X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]
rgi = RegularGridInterpolator(
    [p.numpy() for p in points], values.numpy(), bounds_error=False)
input_rgi = np.vstack([X.flatten(), Y.flatten()]).T

points_cuda = [p.cuda() for p in points]
values_cuda = values.cuda()
gi_cuda = torch_interpolations.RegularGridInterpolator(
    points_cuda, values_cuda)
points_to_interp_cuda = [p.cuda() for p in points_to_interp]


def interp_pytorch_cuda():
    torch.cuda.synchronize()
    gi_cuda(points_to_interp_cuda)
    torch.cuda.synchronize()
    return 1.


def interp_pytorch():
    return gi(points_to_interp)


def interp_numpy():
    return rgi(input_rgi)

times_pytorch = time_function(interp_pytorch)
times_pytorch_cuda = time_function(interp_pytorch_cuda)
times_numpy = time_function(interp_numpy)

print("Interpolating %d points on %d by %d grid" %
      (points_to_interp[0].shape[0], values.shape[0], values.shape[1]))
print("PyTorch took %.3f +\\- %.3f ms" %
      (np.mean(times_pytorch), np.std(times_pytorch)))
print("PyTorch Cuda took %.3f +\\- %.3f ms" %
      (np.mean(times_pytorch_cuda), np.std(times_pytorch_cuda)))
print("Scipy took %.3f +\\- %.3f ms" %
      (np.mean(times_numpy), np.std(times_numpy)))
