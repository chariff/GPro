from setuptools import setup, find_packages

setup(
    name='GPro',
    version='1.0.2',
    url='https://github.com/chariff/GPro',
    packages=find_packages(),
    author='Chariff Alkhassim',
    author_email='chariff.alkhassim@gmail.com',
    description='Preference Learning with Gaussian Processes.',
    long_description='Python implementation of a probabilistic kernel approach '
                     'to preference learning based on Gaussian processes.',
    install_requires=[
        "numpy >= 1.9.0",
        "scipy >= 0.14.0",
        "pandas >= 0.24.0"
    ],
    license='MIT',
)


