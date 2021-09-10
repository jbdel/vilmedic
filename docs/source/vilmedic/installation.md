# Installation
ViLMedic is built on top of important packages:

- [python](https://www.python.org/downloads/) >= 3.5
- [pytorch](http://pytorch.org/) == 1.7.1
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