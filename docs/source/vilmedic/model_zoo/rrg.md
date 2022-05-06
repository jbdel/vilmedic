# Radiology Report Generation

## Usage 
```
from vilmedic import AutoModel

model, processor = AutoModel.from_pretrained("rrg/biomed-roberta-baseline-mimic")

batch = processor.inference(seq='no acute cardiopulmonary process .',
                            image='files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg')

out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'logits', 'past_key_values', 'hidden_states', 'attentions', 'cross_attentions'])
```

## Generate report

``` 
from vilmedic import AutoModel
import torch

model, processor = AutoModel.from_pretrained("rrg/biomed-roberta-baseline-mimic")

batch = processor.inference(image=[
    ["files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"],
    ["files/p10/p10000032/s50414267/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg"],
])

batch_size = len(batch["images"])
beam_size = 8
encoder_output, encoder_attention_mask = model.encode(**batch)
expanded_idx = torch.arange(batch_size).view(-1, 1).repeat(1, beam_size).view(-1).cuda()

# Using huggingface generate method
hyps = model.dec.generate(
    input_ids=torch.ones((len(batch["images"]), 1), dtype=torch.long).cuda() * model.dec.config.bos_token_id,
    encoder_hidden_states=encoder_output.index_select(0, expanded_idx),
    encoder_attention_mask=encoder_attention_mask.index_select(0, expanded_idx),
    num_return_sequences=1,
    max_length=processor.tokenizer_max_len,
    num_beams=8,
)
hyps = [processor.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False) for h in hyps]
print(hyps)
# ['no acute cardiopulmonary process .', 'in comparison with study of there is little change and no evidence of acute cardiopulmonary disease . no pneumonia vascular congestion or pleural effusion .']
```
## Output scoring

``` 
from vilmedic.blocks.scorers.NLG import ROUGEScorer
refs = ['no acute cardiopulmonary process .', 'no evidence of acute cardiopulmonary process  .']
print(ROUGEScorer(rouges=['rougeL']).compute(refs, hyps)[0])
# 0.6724137931034483
```

## Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| rrg/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| rrg/biomed-roberta-baseline-indiana| [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)
| rrg/baseline-padchest| [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/) 