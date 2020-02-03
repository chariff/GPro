from setuptools import setup, find_packages

setup(
    name='GPro',
    version='1.0.0',
    url='https://github.com/chariff/GPro',
    packages=find_packages(),
    author='Chariff Alkhassim',
    author_emain='chariff.alkhassim@gmail.com',
    description='Preference Learning with Gaussian Processes',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "pandas >= 0.22.0"
    ],
    license='MIT',
)

