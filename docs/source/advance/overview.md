
# Overview
<p align="center">
<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/overview.jpg" width="50%"/>
</p>

ViLMedic contains reference implementations of state-of-the-art vision and 
language architectures (referred as "**blocks**" ) and full solutions for multimodal medical 
tasks using one or several blocks (referred as "**models**").

### Blocks and models
Models are full solutions for multimodal medical tasks, usually composed of blocks. <br/>

For example, the task of Radiology Report Generation (RRG) encodes an image and outputs a report. Typically, a 
[baseline solution](https://github.com/jbdel/vilmedic/blob/main/vilmedic/models/rrg/RRG.py) uses two blocks: a [CNN](https://github.com/jbdel/vilmedic/blob/main/vilmedic/blocks/vision/cnn.py) 
and [BioMed-RoBERT HuggingFace transformer as decoder](https://github.com/jbdel/vilmedic/blob/main/vilmedic/blocks/huggingface/encoder_decoder/encoder_decoder_model.py).

Documentation of available solutions are available [here](https://vilmedic.readthedocs.io/en/latest/vilmedic/models.html) and their respective python code in 
`vilmedic/models`.

### Configuration files
All configurations about training, evaluation, models and blocks parameters are stored in YALM file that will be processed with the 
[omegaconf](https://github.com/omry/omegaconf) library. Configuration files are stored in the `config` directory. 

A configuration file usually defines the model, dataset, trainor, validator and ensemblor parameters.

### Trainor

The [Trainor](https://github.com/jbdel/vilmedic/blob/main/vilmedic/executors/trainor.py) has a few jobs.
1. It iterates over the dataset and gives the batch to the model. 
1. The model returns a loss and the trainor calls the optimization algorithm. 
1. After an epoch, the Trainor calls the validator and receives the validation loss and metrics. 
1. According to the current evaluation scores, the Trainor can:
    1. Save the current model
    1. Call a LR scheduler
    1. Early-stop the training

All options are defined through the Trainor settings in the config file.

### Validator
The Validator performs one epoch of the validation set and compute metrics and scores based on the model's output.

The [Validator](https://github.com/jbdel/vilmedic/blob/main/vilmedic/executors/validator.py) is 
either called by the Trainor or the [Ensemblor](https://github.com/jbdel/vilmedic/blob/main/bin/ensemble.py).
The Validator process is the same in both cases except for two differences:
1. The Ensemblor sends a list of models while the Trainor sends the model being trained
1. The Ensemblor can send a list of post-processing methods

### Scorers
Scorers are methods called by the Validator that outputs a score or a metrics based on the model's output.
Available scorers are found in the `vilmedic/blocks/scorers` directory.

