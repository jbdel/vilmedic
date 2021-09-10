# Models

Here is the list of all available models in ViLMedic

## Radiology Report Generation

```
for i in {1..6}
do
    python bin/train.py config/RRG/rrg.yml \
                trainor.batch_size=64 \
                validator.batch_size=4 \
                name=rrg 
done

python bin/ensemblor.py config/RRG/rrg.yml ensemblor.batch_size=4 name=rrg ensemblor.beam_width=8 ensemblor.mode=all
```

The model is defined as such in the config file:
```
model:
  proto: RRG
  decoder:
    proto: data/report_sum/huggingface/biomed_roberta_base
  cnn:
    proto: CNN
    backbone: densenet169
    output_layer: features
    dropout_out: 0.0
    permute: batch_first
    visual_embedding_dim: 1664
    freeze: False
```

### Results

| Split  |     BLEU | ROUGE2 | METEOR | CiDER | 
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Mimic-val   | 43.61   |  43.90  | 45.36  | 6      
| Mimic-test    |  82.30  |  81.53  | 82.26  |  8        
