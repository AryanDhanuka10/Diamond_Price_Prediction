# if you want to create your project as package that can be installed and used somewhere else
# then use setup.py always for making any project make a new setup.py for each
from typing import List

from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements


setup(
    name="DiamondPricePrediction",
    version="0.0.1",
    author="Aryan Dhanuka",
    author_email="a9936067905@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages(),
)
# use pip install . insted of python3 setup.py install
