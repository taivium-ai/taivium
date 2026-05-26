from setuptools import setup, find_packages

setup(
    name='taivium',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add runtime dependencies here, e.g., 'numpy', 'requests'
    ]
)