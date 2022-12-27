import sys

# Available at setup time due to pyproject.toml
import os
from glob import glob
from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

os.environ["CPPFLAGS"] = "-lboost_filesystem"

ext_modules = [
    Pybind11Extension("go",
                      sorted(glob("src/*.cpp")),
                      define_macros=[('VERSION_INFO', __version__)],
                      ),
]

setup(
    name="go",
    version=__version__,
    author="Honglu Fan",
    author_email="honglu2875@gmail.com",
    url="https://github.com/honglu2875/go",
    description="",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
)
