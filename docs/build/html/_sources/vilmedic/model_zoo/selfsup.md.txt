# Self-supversion


## VAE

### Usage 
```
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/vae-padchest")
batch = processor.inference(image=[
    "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
    "p10/p10000032/s50414267/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg",
])
out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'output'])
```
### Reconstruction
```
from vilmedic import AutoModel
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


model, processor = AutoModel.from_pretrained("selfsup/vae-mimic")

batch = processor.inference(image=[
    "p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg",
    "p10/p10000032/s50414267/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.jpg",
])

codes = model.vae.get_codebook_indices(batch['images'].cuda())
hard_recons = model.vae.decode(codes)
show(make_grid(hard_recons.detach().cpu(), normalize=True, range=(-1, 1)))
```
<img src="https://raw.githubusercontent.com/jbdel/vilmedic/main/docs/source/images/vae_mimic.png" width="150px"/>

### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| selfsup/vae-mimic | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/vae-indiana | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)
| selfsup/vae-padchest | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/) 



## SimCLR

### Usage 
```
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/simclr-mimic-64")
batch = processor.inference(image=[files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg'])
out = model(**batch, from_training=False)
print(out.keys())
# dict_keys(['loss', 'visual'])
```

### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| selfsup/simclr-mimic-16 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-32 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/simclr-mimic-64 | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   

## GLoRIA

### Usage 
```
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/gloria-chexpert")
batch = processor.inference(seq=['minimal residual atelectasis at the left lung zone'], 
                            image=['CheXpert-v1.0-small/valid/patient64545/study1/view1_frontal.jpg'])
out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'global_features', 'local_features', 'word_embeddings', 'sent_embeddings'])
```
### Zero-shot classification

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

### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| [selfsup/gloria-chexpert](https://github.com/marshuang80/gloria)  | [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)   |  [Link]()
| [selfsup/gloria-mimic-48]()  | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)  |  [Link]()

## ConVIRT

### Usage 
```
from vilmedic import AutoModel
model, processor = AutoModel.from_pretrained("selfsup/convirt-mimic")
batch = processor.inference(seq=["no acute cardiopulmonary process"],
                            image=["files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg"])

out = model(**batch)
print(out.keys())
# dict_keys(['loss', 'loss_l', 'loss_v', 'linguistic', 'visual'])
```

### Models
| Name  |   dataset | Model Card | 
| ------------- |:-------------:|:-------------:|
| selfsup/convirt-mimic | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/convirt-mimic-balanced | [mimic-cxr](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)   
| selfsup/convirt-padchest-16 | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   
| selfsup/convirt-padchest-32 | [padchest](https://bimcv.cipf.es/bimcv-projects/padchest/)   
| selfsup/convirt-indiana-16 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   
| selfsup/convirt-indiana-32 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   
| selfsup/convirt-indiana-64 | [indiana](https://www.kaggle.com/raddar/chest-xrays-indiana-university/)   
