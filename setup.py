from setuptools import setup

setup(
    name='gelslim_shear',
    packages=['gelslim_shear'],
    version="0.0.1",
    url="https://github.com/MMintLab/gelslim_shear",
    author='WilliamvdB',
    author_email='willvdb@umich.edu',
    description="Gelslim tactile sensor shear field approximation via optical flow",
    install_requires=[
        'numpy',
        'torch',
        'scipy',
        'opencv-python',
        'scikit-image',
    ]
)