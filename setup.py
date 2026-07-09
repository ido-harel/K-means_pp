from setuptools import Extension, setup
 
module = Extension(
    "mykmeanssp",
    sources=["kmeansmodule.c"],
)
 
setup(
    name="mykmeanssp",
    version="1.0",
    description="K-means clustering C extension for HW2 (K-means++).",
    ext_modules=[module],
)
