import pathlib
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 8):
    sys.exit('Sorry, Python >=3.8 is required for ViLMedic.')

setup(
    name='vilmedic',
    version='1.2.6',
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
    install_requires=['appdirs==1.4.4',
                      'omegaconf==2.0.6',
                      'torchvision==0.9.1',
                      'rouge_score',
                      'youtokentome==1.0.3',
                      'tokenizers==0.11.6',
                      'scikit_image==0.18.2',
                      'scikit-learn==0.24.2',
                      'pydicom==2.2.0',
                      'transformers==4.17.0',
                      'seaborn==0.11.1',
                      'dalle-pytorch==1.4.2',
                      'torchxrayvision==0.0.32',
                      'stanza==1.3.0',
                      'bert-score==0.3.11',
                      'torch==1.8.1',
                      'pytorch-lightning==1.4.2',
                      'pytorch-metric-learning==0.9.99',
                      'torch-optimizer==0.1.0',
                      'umap-learn==0.5.2',
                      'opencv-python==4.5.4.60',
                      'mauve-text',
                      'numba==0.54.1',
                      'torchmetrics==0.5.0',
                      'numpy==1.20.3',
                      'gdown==4.3.1',
                      'spacy===3.2.3',
                      # radgraph / allennlp dependencies
                      'overrides==3.1.0',
                      'boto3==1.21.13',
                      'jsonpickle==2.1.0',
                      'h5py==3.6.0',
                      'tensorboardX==2.5',
                      ##
                      'psutil==5.9.0',
                      'lightning-bolts==0.5.0',
                      'faiss-gpu',
                      ],
    include_package_data=True,
    exclude_package_data={'': ['.git']},
    packages=find_packages(exclude=["bin"]),
    scripts=[str(p) for p in pathlib.Path(
        'bin/scripts').glob('*')],
    zip_safe=False)
