# Installation
ViLMedic is built on top of important packages:

- [python](https://www.python.org/downloads/) 3.9
- [pytorch](http://pytorch.org/) 1.8.1
- [transformers](https://huggingface.co/transformers/) 4.5.1

Install ViLMedic in development mode:
```bash
conda create --name vilmedic python==3.9 -y
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