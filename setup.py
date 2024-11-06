# setup.py

from setuptools import setup, find_packages

setup(
    name='mplsandbox',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'mplsandbox=mplsandbox.tool:main',
        ],
    },
    install_requires=[
        'docker',
        'flask',
        'guesslang',
        'openai',
        'astpretty',
        'pyflowchart',
        'javalang',
    ],
)
