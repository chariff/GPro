from setuptools import setup, find_packages


def read_me():
    with open('README.md') as f:
        out = f.read()
    return out


setup(
    name='GPro',
    version='1.0.5',
    url='https://github.com/chariff/GPro',
    packages=find_packages(),
    author='Chariff Alkhassim',
    author_email='chariff.alkhassim@gmail.com',
    description='Preference Learning with Gaussian Processes.',
    # long_description='Python implementation of a probabilistic kernel approach '
    #                  'to preference learning based on Gaussian processes.',
    long_description=read_me(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "numpy >= 1.9.0",
        "scipy > 1.4.0",
        "pandas >= 0.24.0"
    ],
    license='MIT',
)


