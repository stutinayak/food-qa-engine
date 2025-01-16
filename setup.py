from setuptools import setup, find_packages

setup(
    name="food-qa",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
    ],
) 