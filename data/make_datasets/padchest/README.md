
#### Reports
Download images and reports from https://bimcv.cipf.es/bimcv-projects/padchest/.

Place the csv file `PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv` folder in this 
`data/make_datasets/padchest` folder. 

Your tree should look like this:

``` 
padchest
├── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
├── create_section_files.py
├── make_padchest.py
└── README.md
```

Then run the command:

```
python make_mimic_cxr.py --task [rrg,rrs,selfsup]
```

#### Images
Download images and reports from https://bimcv.cipf.es/bimcv-projects/padchest/

The images must be stored in the root `data/images/padchest-images-512` folder of ViLMedic. 
You are free to resize the images using the following transform:
``` 
transforms.Compose([transforms.Resize(512)])        
```

**Warning: do not use `transforms.Resize(512,512)`**        


Your `data` tree should now look like this:

```
data
├── images
│   └── padchest-images-512
│       ├── 216840111366964013829543166512013338135747880_02-094-102.png
│       ├── 216840111366964013829543166512013338135758176_02-089-127.png
│       ├── ...
├── SELECTED_TASK
│   └── padchest
│       ├── test.report.tok
│       ├── test.image.tok
│       ├── train.report.tok
│       ├── train.image.tok
│       ├── validate.report.tok
│       └── validate.image.tok
```

