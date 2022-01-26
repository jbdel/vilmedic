import pathlib
from setuptools import setup, find_packages

setup(
    name='vilmedic',
    version='1.0.7',
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
    python_requires='~=3.9',
    install_requires=['Cython', 'appdirs==1.4.4', 'omegaconf==2.0.6', 'torchvision==0.9.1', 'rouge_score',
                      'scikit_image==0.18.2',
                      'scikit-learn==0.24.2', 'pydicom==2.2.0', 'transformers==4.5.1', 'tokenizers==0.10.3',
                      'seaborn==0.11.1', 'gdown==4.2.0',
                      'dalle-pytorch==1.4.2', 'torchxrayvision==0.0.32',
                      'torch==1.8.1', 'pytorch-lightning==1.4.2', 'pytorch-metric-learning==0.9.99',
                      'torch-optimizer==0.1.0', 'umap-learn==0.5.2', 'opencv-python==4.5.4.60',
                      'mauve-text', 'numba==0.54.1', 'torchmetrics==0.5.0', 'numpy==1.20.3'],
    include_package_data=True,
    exclude_package_data={'': ['.git']},
    packages=find_packages(exclude=["bin"]),
    scripts=[str(p) for p in pathlib.Path('bin/scripts').glob('*')],
    zip_safe=False)
