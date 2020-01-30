from setuptools import setup

setup(
    name='GPro',
    version='1.0.0',
    url='https://github.com/chariff/GPro',
    author='Chariff Alkhassim',
    author_emain='chariff.alkhassim@gmail.com',
    description='Preference Learning with Gaussian Processes',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "pandas >= 0.24.0"
    ],
    license='MIT',
)

