from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    with open(file_path, 'r') as f:
        reqs = f.readlines()
        reqs = [req.replace('\n', '') for req in reqs]

        if HYPHEN_E_DOT in reqs:
            reqs.remove(HYPHEN_E_DOT)

    return reqs

setup(
    name='ML_project_deploy',
    version='0.0.1',
    author='Aritra',
    author_email='aritra323404@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)