from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
]

setup(
    name='EduCDM',
    version='0.0.11',
    extras_require={
        'test': test_deps,
    },
    packages=find_packages(),
    install_requires=[
        "torch",
        "tqdm",
        "numpy>=1.16.5",
        "scikit-learn",
        "pandas",
        "longling>=1.3.23"
    ],  # And any other dependencies for needs
    entry_points={
    },
)
