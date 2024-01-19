from setuptools import setup, find_packages

setup(
    name='mbot_xl320_library',
    version='0.1.0',
    packages=['mbot_xl320_library'],
    package_dir={'': 'src'},
    description='Python library to use XL320 Servo',
    author='Shaw Sun',
    author_email='xssun@umich.edu',
    install_requires=['dynamixel_sdk']
)
