# Installation
ViLMedic is built on top of important packages:

- [python](https://www.python.org/downloads/) 3.9
- [pytorch](http://pytorch.org/) 1.8.1
- [transformers](https://huggingface.co/transformers/) 4.5.1

First install a python >= 3.6 environment
```bash
conda create --name vilmedic python==3.9 -y
```

If you plan to use the model-zoo only, you can install vilmedic using pip:
```bash
pip install vilmedic
```

If you want to initiate trainings or replicate solutions, install ViLMedic in development mode:
```bash
git clone https://github.com/jbdel/vilmedic
python setup.py develop
```

ViLMedic is structured as follows:

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

All pretrained models and datasets downloaded from this documentation will be placed in the `data` folder.