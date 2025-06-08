from setuptools import find_packages, setup
from typing import List 

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    retrieve the required packages
    '''
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
            
        return requirements


setup(
    name='Binary_Prediction_of_Smoker_Status',
    version='0.0.1', 
    author='Ajinkya',
    author_email='ajinkya.tamhankar18@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)