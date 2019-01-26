from numpy.distutils.core import Extension, setup
import builtins

with open("README.md", 'r') as f:
    long_description = f.read()

extension = Extension(name="_macula", sources=["dwelf/macula.f90"])
builtins.__DWELF_SETUP__ = True
import dwelf
version = dwelf.__version__

setup(
    name="dwelf",
    version=version,
    author="Eduardo Nunes",
    author_email="diofanto.nunes@gmail.com",
    license="MIT",
    description="Stellar parameter determination based on Spot Modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dioph/dwelf",
    packages=["dwelf"],
    package_data={'dwelf':['data/*']},
    include_package_data=True,
    ext_modules=[extension],
    install_requires=["numpy>=1.11", "scipy>=0.19.0", "astropy>=1.3",
                      "matplotlib", "emcee", "tqdm"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ),
)
