from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize('lp_bench_cython/_impl.pyx'),
    packages=['lp_bench_cython'],
    # line_profiler needs the source next to the module to profile it
    package_data={'lp_bench_cython': ['*.pyx']},
)
