from setuptools import setup, Extension
from Cython.Build import cythonize


MAJOR, MINOR, MICRO = 2, 3, 0
VERSION = f"{MAJOR}.{MINOR}.{MICRO}"
NAME = "BERT4Rec"


ext_modules = [
    Extension(
        name="bert4rec.data.c_mr",
        language="c++",
        sources=["bert4rec/data/c_mr.pyx"],
        extra_compile_args=["-std=c++17", "-pthread", "-O3"]
    )
]


if __name__ == "__main__":
    setup(
        name=NAME,
        version=VERSION,
        description="BERT4Rec for sequential recommendation",
        ext_modules=cythonize(ext_modules),
        packages=["bert4rec", "bert4rec/data", "bert4rec/data/custom_dataset", "bert4rec/model", "bert4rec/trainer", "bert4rec/utils"],
        python_requires=">=3.7"
    )
