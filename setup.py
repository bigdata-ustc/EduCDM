from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    # 'pytest-flake8==4.0.1',
    'pytest-flake8<1.1.2',  # When >=1.2.2, incompatable with flake8>=5.0.0. E.g., flake8.options.config.load_config
    'flake8<5.0.0'
]

setup(
    name='EduCDM',
    version='0.0.13',
    extras_require={
        'test': test_deps,
    },
    packages=find_packages(),
    install_requires=[
        "torch", "tqdm", "numpy>=1.16.5", "scikit-learn", "pandas",
        "longling>=1.3.33", "longling<=1.3.36", 'PyBaize>=0.0.7', 'fire', 'networkx>=2.8.5'
    ],  # And any other dependencies for needs
    entry_points={},
)
