from setuptools import setup, find_packages

setup(
    name="geometrize",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.6.0",
        "numpy>=2.2.3",
        "matplotlib>=3.10.1",
        "open3d>=0.19.0",
        "build123d>=0.9.1",
        "numpy-stl>=3.2.0",
    ],
    include_package_data=True,
    author="Roberto Hart-Villamil",
    author_email="rob.hartvillamil@gmail.com",
    description="Geometrize: Automatic geometry parameterization using neural networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/robh96/geometrize",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    keywords=(
        "geometry parameterization",
        "autoencoder",
        "shape representation",
        "point cloud",
        "neural networks",
    ),
)
