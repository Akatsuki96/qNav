
from setuptools import setup, find_packages


setup(
    name="qNav",
    version="1.0.0",
    description="QLearning for Olfactory Navigation",
    python_requires='~=3.10.6',
    setup_requires=['setuptools>=18.0'],
    packages=find_packages(),
    install_requires=['numpy'],
    include_package_data=True,
)