# Radiology Report Generation

## Usage 
```
from vilmedic import AutoModel

model, processor = AutoModel.from_pretrained("rrs/biomed-roberta-baseline-mimic")

batch = processor.inference(src=["the lungs are clear the cardiomediastinal silhouette is within normal limits no acute osseous abnormalities"],
                            tgt=["no acute cardiopulmonary process"])

print(batch.keys())
>> dict_keys(['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask'])

out = model(**batch)
print(out.keys())
>> dict_keys(['loss', 'logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions', 'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions'])
```

## Generate summary

``` 
batch = processor.inference(src=["the lungs are clear the cardiomediastinal silhouette is within normal limits no acute osseous abnormalities"])

batch = {k: v.cuda() for k, v in batch.items()}
hyps = model.enc_dec.generate(**batch,
                              num_beams=8,
                              num_return_sequences=1,
                              max_length=processor.tgt_tokenizer_max_len,
                              )
hyps = [processor.tgt_tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False) for h in hyps]
print(hyps)
>> ['no acute cardiopulmonary process']
```

## Output scoring
``` 
from vilmedic.blocks.scorers.NLG import ROUGEScorer
refs = ["no acute cardiopulmonary process"]
print(ROUGEScorer(rouges=['rougeL']).compute(refs, hyps)[0])
# 1.0
```

## Models
| Name  | Dataset  | Model Card | 
| ------------- |:-------------:|:-------------:|
| rrs/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| rrs/biomed-roberta-baseline-indiana| [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)
