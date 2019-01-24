import setuptools

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="dwelf",
    version="1.0a1",
    author="Eduardo Nunes",
    author_email="diofanto.nunes@gmail.com",
    license="MIT",
    description="Stellar parameter determination based on Spot Modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/dwelf",
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.11', 'scipy>=0.19.0', 'astropy>=1.3',
                      'matplotlib', 'emcee', 'corner'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
