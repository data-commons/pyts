from setuptools import setup, find_packages

version_string = '0.1.0'

setup(
    name='pyts',
    description='A library for stats module in python',
    author='data-commons',
    author_email='data-commons-toolchain@googlegroups.com',
    url='https://github.com/data-commons/pyts',
    version=version_string,
    packages=find_packages(),
    include_package_data=True,
    classifiers=[],
    keywords=['stats'],
    install_requires=[
        'numpy >= 1.9.2'
    ],
    test_requires=[
        'nose == 1.3.7'
    ]
)