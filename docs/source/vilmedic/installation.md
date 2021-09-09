# Installation

This page provides basic prerequisites to run ViLMedic.

## Hardware & Software Setup

A machine with at least **1 GPU (>= 8GB)** is required.  

ViLMedic is built on top of important packages:

- [Python](https://www.python.org/downloads/) >= 3.5
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.0 and [cuDNN](https://developer.nvidia.com/cudnn)
- [PyTorch](http://pytorch.org/) == 1.7.1 with CUDA 
- [transformers](https://huggingface.co/transformers/) == 4.5.1

All the packages and minor dependencies can be installed using
```bash
pip install -r requirements.txt
```

ViLMedic structure is as follows:

```
|-- vilmedic
	|-- data
	|-- bin
	|-- config
	|-- vilmedic
	|  |-- datasets
	|  |-- executors
	|  |-- networks
	|  |  |  |-- blocks
	|  |  |  |-- models
	|  |-- scorers
```

All datasets downloaded from this documentation should be placed in the `data` folder.