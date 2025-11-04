from setuptools import setup, find_packages
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

# load version
with open(os.path.join(current_dir, 'pibs/__version__.py'), 'r') as f:
    exec(f.read(), globals())


# requirements
with open(os.path.join(current_dir,'requirements.txt'), 'r') as f:
    requirements = f.readlines()

# Project description
with open(os.path.join(current_dir,'README.md'), 'r') as f:
  long_description = f.read()


setup(
        name = 'pibs',
        version = __version__,
        url = 'https://github.com/lukasfreter/pibs'
        authors = [
            {name = 'Lukas Freter', email = 'lukas.freter@aalto.fi'},
            {name = 'Piper Fowler-Wright', email = ''},

        packages=find_packages(),

        install_requires= requirements,
        description = 'An implementation of a numerically exact method for solving the dissipative Tavis-Cummings model with local losses',
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        keywords = ['open-quantum-systems', 'tavis-cummings'],
        license = '', #?
        

        )
