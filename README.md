Report sum:

Download mimic-cxr file [here](https://drive.google.com/drive/folders/1So1-aZuEsqAYC3bQLAy1go6H09POntEW?usp=sharing) 
and place it in data/report_sum/mimic-cxr. For each split, you have three aligned files. For eg:

```
train.impression.tok
train.findings.tok
train.image.tok
```
If you want to train a multimodal model download the visual features from [here](https://drive.google.com/drive/folders/1Z3sST5qUfMvm3ndSzEpfotycsO1VXwUh?usp=sharing).
For example, download xrv-conv features:
```
train.mimic-cxr.xrv.layer4.npz
validate.mimic-cxr.xrv.layer4.npz
test.mimic-cxr.xrv.layer4.npz
```
and place it in data/report_sum/features/.

You can train a model using convolutional features using `config/report_sum/conv_multimodal_mimic.yml`.
The training command is:
```
for i in {1..5}; 
do 
    python bin/train.py config/report_sum/conv_multimodal_mimic.yml
done
```
This will train 5 models. The output will be in `conv_multimodal_mimic`

You can then ensemble models using:
```
python bin/ensemblor.py config/report_sum/conv_multimodal_mimic.yml
```
The [ensemblor] part of config will be called. <br/>
ensemblor.mode can be 'best-n' to ensemble n models, or 'all' to take all trained models

You can easily override arguments using `-o` argument. for eg:
```
python bin/ensemblor.py config/report_sum/conv_multimodal_mimic.yml -o ensemblor.mode=best-2 ensemblor.beam_width=12
```
Same goes for training.