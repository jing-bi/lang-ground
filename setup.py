from setuptools import setup, find_packages

setup(
    name="langground",  
    version="0.1.0",    
    description="Use natural language to ground relevant things.",  
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Jing Bi, Guangyu Sun", 
    author_email="jbi5@ur.rochester.edu, guangyu@ucf.edu",
    url="https://github.com/jing-bi/lang-ground",  
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'opencv-python',
        'accelerate',
        'pillow',
        'scipy',
        'gradio',
        'tenacity'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  
)