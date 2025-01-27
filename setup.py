from setuptools import find_packages, setup

setup(
    name="nucleus-displacement-index",
    version="1.2",
    author="Kolya Lettl",
    author_email="kolya.lettl@uni-leipzig.de",
    description="library for calculating the nucleus displacement index (NusDI)",
    packages=find_packages(),
    install_requires=["numpy", "opencv-python", "shapely"],
)
