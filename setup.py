from setuptools import setup, find_packages

setup(
    name="benchmark-power-demo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "openai",
    ],
) 