from setuptools import setup, find_packages
import subprocess


setup(
    name="torch_interpolations",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Shane Barratt",
    author_email="stbarratt@gmail.com",
    description="XXX",
    keywords="pytorch torch multilinear interpolation differentiable",
    url="https://www.github.com/sbarratt/torch_interpolations",
    long_description=open("README.md", encoding="utf-8").read(),
)
