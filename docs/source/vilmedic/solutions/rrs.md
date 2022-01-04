
## Radiology Report Summarization

### Monomodal
```
for i in {1..6}
do
    python bin/train.py config/summarization/biorobert_mono.yml \
        trainor.batch_size=64 \
        validator.batch_size=4 \
        name=sum_mono
done
```
The model is defined as such in the config file:
```
model:
  proto: SumHugMono
  encoder:
    proto: data/report_sum/huggingface/biomed_roberta_base
  decoder:
    proto: data/report_sum/huggingface/biomed_roberta_base
```
### Metrics and scores
One model: 

```
python bin/ensemble.py config/RRG/biorobert_mono.yml \
    ensemblor.batch_size=4 \
    ensemblor.beam_width=8 \
    ensemblor.mode=best-1 \
    name=sum_mono 
```

| Split  |     BLEU | ROUGE2 | METEOR | ROUGEL
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|
| Mimic-val   | 27.55   |  48.10  | 25.27    | 57.16 
| Mimic-test    |  19.58  |  33.63  | 21.02 | 45.16
