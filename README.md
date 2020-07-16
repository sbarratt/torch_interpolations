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

## License
cvxpylayers carries an Apache 2.0 license.