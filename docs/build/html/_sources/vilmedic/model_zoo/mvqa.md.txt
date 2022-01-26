
# Medical VQA


## Usage 
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

## Compute accuracy on custom data

``` 
from vilmedic import AutoModel

model, processor = AutoModel.from_pretrained("mvqa/mvqa-imageclef")
batch = processor.inference(image="data/images/imageclef-vqa-images-512/synpic253.jpg", label="horseshoe kidney")
out = model(**batch, from_training=False)

print(out["answer"].item() == batch["labels"].item())
# True
```

## Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| mvqa/mvqa-imageclef| [ImageCLEF-VQAMed](https://www.imageclef.org/2021/medical/vqa)   

