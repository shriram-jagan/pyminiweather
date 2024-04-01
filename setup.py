from pathlib import Path

from setuptools import setup


def _get_version() -> str:
    """Read miniweather/VERSION.txt and return its contents."""
    path = Path("pyminiweather").resolve()
    version_file = path / "VERSION.txt"
    return version_file.read_text().strip()


version = _get_version()


with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()


setup(
    name="pyminiweather",
    version=version,
    description=(
        "A miniapp for weather and climate flows based on the c/c++ "
        "version of miniweather by Matt Norman. The miniapp solves "
        "the time-dependent two-dimensional compressible form of "
        "Euler equations on a structured mesh using finite volume "
        "scheme but using cuNumeric"
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Shriram Jagannathan",
    author_email="sjagannathan@nvidia.com",
    packages=['pyminiweather'],
    package_dir={'pyminiweather': 'pyminiweather'},
    entry_points={'console_scripts': ['pyminiweather.py = pyminiweather.__main__:main']},
    include_package_data=True,
    python_requires=">=3.0",
    # install_requires=requirements,
    license="BSD",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.0",
        "Topic :: Software Development",
    ],
)
