# setup.py

# from setuptools import setup, find_packages

# setup(
#     name='mplsandbox',
#     version='0.1',
#     packages=find_packages(),
#     entry_points={
#         'console_scripts': [
#             'mplsandbox=mplsandbox.tool:main',
#         ],
#     },
#     install_requires=[
#         'docker',
#         'flask',
#         'guesslang',
#         'openai',
#         'astpretty',
#         'pyflowchart',
#         'javalang',
#     ],
# )


from setuptools import setup, find_packages
import pathlib

try:
    with open(pathlib.Path(__file__).parent / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Multi-Programming Language Sandbox for LLMs"

setup(
    name="mplsandbox",
    version="0.1.0",
    author="Jianxiang Zang",
    author_email="525967361@qq.com",
    description="Multi-Programming Language Sandbox for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ablustrund/MPLSandbox",
    
    python_requires=">=3.7",
    install_requires=[
        'docker',
        'flask',
        'guesslang',
        'openai',
        'astpretty',
        'pyflowchart',
        'javalang',
    ],
    
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,  
    entry_points={
        'console_scripts': [
            'mplsandbox=mplsandbox.tool:main',
        ],
    },
    

    classifiers=[
        "Development Status :: 3 - Alpha",
        
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        
        "License :: OSI Approved :: Apache Software License",
        
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        "Operating System :: OS Independent",
        
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
        "Framework :: Matplotlib",
    ],
    
    
    extras_require={
        "security": ["pyOpenSSL>=23.0"],
    }
)