# torch_interpolations

This package implements interpolation routines in [PyTorch](pytorch.org),
making them GPU-capable and differentiable.
The only interpolation routine supported so far is [RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html), from `scipy`.

## Installation

Install the preview of PyTorch here: https://pytorch.org/.
Ensure that you have torch version >= 1.7.0.
Then navigate to the main directory of this repository and run:
```
$ python setup.py install
```

## API
First, you construct a `torch_interpolations.RegularGridInterpolator` object by supplying
it `points` (a list of torch Tensors) and `values` (a torch Tensor):
```
rgi = torch_interpolations.RegularGridInterpolator(points, values)
```
Then, to interpolate a set of points, you run:
```
rgi(points_to_interp)
```
where `points_to_interp` is a list of torch Tensors, each with the same shape.

## Tests
First, install pytest:
```
pip install pytest
```
Then, from the main directory, run:
```
pytest .
```

## Examples
To run the examples, first install the dependencies:
```
pip install scipy matplotlib
```
Then navigate to the `examples` folder.
The examples are the basic example:
```
python basic_example.py
```
and the two-dimensional example:
```
python two_dimensional.py
```

## Performance
We include a performance script in the `perf/` folder. To run it, navigate to the `perf/` folder, and run:
```
python perf.py
```
It compares the `torch` CPU and GPU implementations, and the `scipy` CPU implementation.
The test case involves interpolating 4.5 million points on a 300 x 300 grid.
The timing results on my machine (Intel i9-9980XE and Nvidia GeForce 1080 TI) are:
```
Interpolating 4500000 points on 300 by 300 grid
PyTorch took 146.812 +\- 7.167 ms
PyTorch Cuda took 14.799 +\- 6.816 ms
Scipy took 545.935 +\- 1.089 ms
```
So the torch GPU implementation is 20 times faster than the torch CPU implementation, which itself is twice as fast as the `scipy` implementation.

## License
torch_interpolations carries an Apache 2.0 license.
