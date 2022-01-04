# Model Zoo

The following is a list of pretrained model available in ViLMedic

## Self-supversion
### SimCLR


#### Usage 
```
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/simclr-mimic-64")
batch = processor.inference(image=[files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg'])
out = model(**batch, from_training=False)
print(out.keys())
# dict_keys(['loss', 'visual'])
```
#### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| selfsup/simclr-mimic-16 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-32 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-64 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   

### GLoRIA

#### Usage 
```
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/gloria-chexpert")
batch = processor.inference(seq=['minimal residual atelectasis at the left lung zone'], 
                            image=['CheXpert-v1.0-small/valid/patient64545/study1/view1_frontal.jpg'])
out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'global_features', 'local_features', 'word_embeddings', 'sent_embeddings'])
```
#### Zero-shot classification

``` 
reports = {
    "atelectasis": ['minimal residual atelectasis at the left lung zone',
                    'minimal subsegmental atelectasis at the left lung base',
                    'trace atelectasis at the mid lung zone',
                    'mild bandlike atelectasis at the lung bases',
                    ],
    "cardiomegaly": ["cardiac silhouette size is upper limits of normal",
                     "cardiomegaly which is unchanged",
                     "mildly prominent cardiac silhouette",
                     "portable view of the chest demonstrates stable cardiomegaly",
                     ]}

image = [
    "CheXpert-v1.0-small/valid/patient64545/study1/view1_frontal.jpg",  # atelectasis
    "CheXpert-v1.0-small/valid/patient64560/study1/view1_frontal.jpg",  # atelectasis
    "CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg",  # cardiomegaly
    "CheXpert-v1.0-small/valid/patient64549/study1/view1_frontal.jpg",  # cardiomegaly
]

class_similarities = []
for v in reports.values():
    batch = processor(seq=v, image=image)
    cls_similarity = model.zero_shot_classification(**batch)
    class_similarities.append(cls_similarity)

class_similarities = np.stack(class_similarities, axis=1)
if class_similarities.shape[0] > 1:
    class_similarities = (class_similarities - class_similarities.mean(axis=0)) / (class_similarities.std(axis=0))

print(pd.DataFrame(class_similarities, columns=reports.keys()))

#    atelectasis  cardiomegaly
# 0     1.126602     -0.320315
# 1    -1.598918     -1.494752
# 2     0.059104      0.996650
# 3     0.413202      0.818415
```

#### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| [selfsup/gloria-chexpert](https://github.com/marshuang80/gloria)  | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)   |  [Link]()

### ConVIRT

#### Usage 
```
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/convirt-mimic")
batch = processor.inference(seq=["no acute cardiopulmonary process"],
                            image=["files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"])

out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'loss_l', 'loss_v', 'linguistic', 'visual'])
```

#### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| rrg/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   

## Radiology Report Generation

### Usage 
```
from vilmedic import AutoModel

model, processor = AutoModel.from_pretrained("rrg/biomed-roberta-baseline-mimic")

batch = processor.inference(seq='no acute cardiopulmonary process .',
                            image='files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg')

out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'logits', 'past_key_values', 'hidden_states', 'attentions', 'cross_attentions'])
```

### Generate report

``` 
from vilmedic import AutoModel
import torch

model, processor = AutoModel.from_pretrained("rrg/biomed-roberta-baseline-mimic")

batch = processor.inference(image=[
    "files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
    "files/p10/p10000032/s50414267/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg",
])

batch_size = len(batch["images"])
beam_size = 8
expanded_idx = torch.arange(batch_size).view(-1, 1).repeat(1, beam_size).view(-1).cuda()

hyps = model.dec.generate(
    input_ids=torch.ones((len(batch["images"]), 1), dtype=torch.long).cuda() * model.dec.config.bos_token_id,
    encoder_hidden_states=model.encode(**batch).index_select(0, expanded_idx),
    num_return_sequences=1,
    max_length=120,
    num_beams=8,
)
hyps = [processor.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False) for h in hyps]
print(hyps)
# ['no acute cardiopulmonary process .', 'in comparison with study of there is little change and no evidence of acute cardiopulmonary disease . no pneumonia vascular congestion or pleural effusion .']

```
### Output scoring

``` 
from vilmedic.scorers.NLG import ROUGEScorer
refs = ['no acute cardiopulmonary process .', 'no evidence of acute cardiopulmonary process  .']
print(ROUGEScorer(rouges=['rougeL']).compute(refs, hyps)[0])
# 0.6724137931034483
```

### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| rrg/biomed-roberta-baseline-mimic| [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   


## Medical VQA


### Usage 
```
from vilmedic import AutoModel

model, processor = AutoModel.from_pretrained("mvqa/mvqa-imageclef")
batch = processor.inference(image=["data/images/imageclef-vqa-images-512/synpic253.jpg"])
out = model(**batch, from_training=False)
answer = out["answer"][0]

print(out.keys())
print(processor.labels_map.idx2label[answer.item()])

# dict_keys(['loss', 'output', 'answer', 'attentions'])
# horseshoe kidney
```

### Compute accuracy on custom data

``` 
from vilmedic import AutoModel

model, processor = AutoModel.from_pretrained("mvqa/mvqa-imageclef")
batch = processor.inference(image="data/images/imageclef-vqa-images-512/synpic253.jpg", label="horseshoe kidney")
out = model(**batch, from_training=False)

print(out["answer"].item() == batch["labels"].item())
# True
```

### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| mvqa/mvqa-imageclef| [ImageCLEF-VQAMed](https://www.imageclef.org/2021/medical/vqa)   

