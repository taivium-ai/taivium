from setuptools import setup, find_packages

setup(
    name='taivium',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add runtime dependencies here, e.g., 'numpy', 'requests'
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'tox'
        ]
    },
    entry_points={
        'console_scripts': [
            'taivium=taivium.main:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)