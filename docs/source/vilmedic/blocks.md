# General concepts
ViLMedic contains reference implementations of state-of-the-art vision and 
language architectures (referred as "**blocks**" ) and full solutions for multimodal medical 
tasks using one or several blocks (referred as "**models**").

### Blocks and models
Models are full solutions for multimodal medical tasks, composed of blocks. <br/><br/>

For example, the task of Radiology Report Generation (RRG) encodes an image and outputs a report. Typically, a 
[baseline solution](https://github.com/jbdel/vilmedic/blob/main/vilmedic/networks/models/rrg/RRG.py) uses two blocks: a [CNN](https://github.com/jbdel/vilmedic/blob/main/vilmedic/networks/blocks/vision/cnn.py) 
and [BioMed-RoBERT HuggingFace transformer as decoder](https://github.com/jbdel/vilmedic/blob/main/vilmedic/networks/blocks/huggingface/encoder_decoder/encoder_decoder_model.py).
<br/>
Models and solutions are found in the `vilmedic/networks` directory.

### Configuration files
All configurations about training, evaluation, models and blocks parameters are stored in YALM file that will be processed with the 
[omegaconf](https://github.com/omry/omegaconf) library. <br/>
Configuration files are stored in the `config` directory.

### Scorers
Scorers are methods called during evaluations and outputs a score and can be specified in the configuration files.
For example, the BLEU scorer method takes hypothese and reference sentences as input and outputs the BLEU score.
<br/>
Models and solutions are found in the `vilmedic/scorers` directory.

