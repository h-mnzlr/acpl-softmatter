# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 27.04.2022
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="soft_matter",
    author="Heiko Menzler",
    author_email="heikogeorg.menzler@stud.uni-goettingen.de",
    description="Statisical Physics in soft matter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.10.8",
)
