from setuptools import setup, find_packages

setup(
    name="geometrize",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "open3d>=0.13.0",
        "build123d>=0.1.0",
        "numpy-stl>=2.16.0",
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
    python_requires=">=3.7",
    keywords=(
        "geometry parameterization",
        "autoencoder",
        "shape representation",
        "point cloud",
        "neural networks",
    ),
)
