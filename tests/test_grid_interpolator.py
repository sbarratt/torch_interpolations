import torch
import torch_interpolations
from scipy.interpolate import RegularGridInterpolator
import numpy as np


def test_regular_grid_interpolator():
    points = [torch.arange(-.5, 2.5, .1) * 1., torch.arange(-.5, 2.5, .2) * 1.]
    values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
    gi = torch_interpolations.RegularGridInterpolator(points, values)

    X, Y = np.meshgrid(np.arange(-.5, 2, .1), np.arange(-.5, 2, .1))
    points_to_interp = [torch.from_numpy(
        X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]
    fx = gi(points_to_interp)
    print(fx)
    rgi = RegularGridInterpolator(
        [p.numpy() for p in points], values.numpy(), bounds_error=False)

    np.testing.assert_allclose(
        rgi(np.vstack([X.flatten(), Y.flatten()]).T), fx.numpy(), atol=1e-6)


def test_regular_grid_interpolator_derivative():
    points = [torch.arange(-.5, 2.5, .5) * 1., torch.arange(-.5, 2.5, .5) * 1.]
    values = torch.sin(points[0])[:, None] + 2 * torch.cos(points[1])[None, :] + torch.sin(5 * points[0][:, None] @ points[1][None, :])
    values.requires_grad_(True)

    X, Y = np.meshgrid(np.arange(-.5, 2, .19), np.arange(-.5, 2, .19))
    points_to_interp = [torch.from_numpy(
        X.flatten()).float(), torch.from_numpy(Y.flatten()).float()]

    def f(values):
        return torch_interpolations.RegularGridInterpolator(
            points, values)(points_to_interp)

    torch.autograd.gradcheck(f, (values,), eps=1e-5, atol=1e-1, rtol=1e-1)
