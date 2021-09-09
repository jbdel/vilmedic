#### Monomodal

Download mimic-cxr/indiana folders [here](https://drive.google.com/drive/folders/1So1-aZuEsqAYC3bQLAy1go6H09POntEW?usp=sharing) 
and place it in data/report_sum/. For each split, you have three aligned files. For eg:

```
train.impression.tok
train.findings.tok
train.image.tok
val.impression.tok
...
```

To train a roberta model, also download the huggingface folder, and place it in data/report_sum/.

You can train five models using the following command:
```
for i in {1..5}; 
do 
    python bin/train.py config/summarization/rnn_mono.yml \
            model.encoder.hidden_size=1024 \
            model.encoder.input_size=768 \
            model.decoder.hidden_size=1024 \
            model.decoder.input_size=768 \
            trainor.batch_size=64
done
```

for indiana 
```
    python bin/train.py config/summarization/rnn_mono.yml \
            dataset.src.root=data/report_sum/indiana/ \
            dataset.tgt.root=data/report_sum/indiana/ \
            name=rnn_mono_indiana \
            model.encoder.n_vocab=1546 \
            model.decoder.n_vocab=1192 \
            model.encoder.hidden_size=1024 \
            model.encoder.input_size=768 \
            model.decoder.hidden_size=1024 \
            model.decoder.input_size=768 \
            trainor.batch_size=8
```

when using biobert:
```
python bin/train.py config/summarization/biorobert_mono.yml \
            validator.batch_size=4 \
            trainor.batch_size=8 \
            trainor.grad_accu=4 \
            trainor.lr=0.00005 \
            weight_decay=0.00001

python bin/train.py config/summarization/biorobert_mono.yml \
            dataset.src.root=data/report_sum/indiana/ \
            dataset.tgt.root=data/report_sum/indiana/ \
            validator.batch_size=4 \
            trainor.batch_size=8 \
            trainor.grad_accu=1 \
            name=biorobert_mono_indiana \
            trainor.lr=0.00005 \
            weight_decay=0.00001
```

#### Multimodal

You can train a multimodal model by downloading mimic images from [here](https://drive.google.com/file/d/1V3RcFKzfFZgfs0-yXqZaUHET3CktTxiM/view?usp=sharing
) and indiana images from [here](https://drive.google.com/file/d/10Heokxw-22CLSUEluEBsSSgK90efxQll/view?usp=sharing).

The training command are the same as above, just replace <br/>
rnn_mono => rnn_multi.yml<br/>
biorobert_mono => biorobert_multi.yml
