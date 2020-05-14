from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.16", "scipy>=1.3"]

setup(
    name="pyvest",
    version="0.0.1",
    author="John Buynak",
    author_email="jbuynak94@gmail.com",
    description="Provides valuable quantitative analysis tools",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/JohnBuynak/pyvest",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)