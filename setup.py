import pathlib
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit('Sorry, Python >=3.8 is required for ViLMedic.')

setup(
    name='vilmedic',
    version='1.3.5',
    description='ViLMedic is a modular framework for multimodal research at the intersection of vision and language in the medical field.',
    author='Jean-Benoit Delbrouck',
    license='MIT',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='medical nlp deep-learning pytorch',
    python_requires='>=3.8',
    setup_requires="Cython",
    install_requires=['appdirs',
                      'omegaconf==2.0.6',
                      'rouge_score',
                      'scikit-learn',
                      'scikit-image',
                      'pydicom',
                      'transformers==4.39.0',
                      'torchvision==0.17.1',
                      'torch==2.2.1',
                      'seaborn',
                      'stanza',
                      'bert-score==0.3.11',
                      'torch-optimizer',
                      'umap-learn',
                      'opencv-python==4.5.4.60',
                      'gdown==4.6.0',
                      'datasets',
                      'radgraph',
                      'f1chexbert',
                      'einops',
                      'monai',
                      'pydantic==1.7.4',
                      ],
    include_package_data=True,
    exclude_package_data={'': ['.git']},
    packages=find_packages(exclude=["bin"]),
    scripts=[str(p) for p in pathlib.Path('bin/scripts').glob('*')],
    zip_safe=False)
