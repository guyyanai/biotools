from setuptools import setup, find_packages

setup(
    name="biotools",               # Replace with your library's name
    version="0.0.10",
    packages=find_packages(),        # Automatically finds modules and sub-packages
    description="Tools for Computational Biology",
    author="Guy Yanai",
    author_email="guy@shay.co.il",
    license="MIT",
    install_requires=[],             # Add any dependencies your library needs
    python_requires=">=3.8",         # Specify the Python versions supported
)
