
#### Reports
You must a physionet account with permissions to download MIMIC-CXR Database. 

Download `mimic-cxr-reports.zip` from https://physionet.org/content/mimic-cxr/2.0.0/mimic-cxr-reports.zip.
Place the extracted `mimic-cxr-reports` folder in this `data/make_datasets/mimic_cxr` folder. 

Download `mimic-cxr-2.0.0-split.csv.gz` and `mimic-cxr-2.0.0-metadata.csv.gz` from https://physionet.org/content/mimic-cxr-jpg/2.0.0/. Place the 
extracted files in this `data/make_datasets/mimic-cxr` folder.

Your tree should look like this:

``` 
mimic_cxr
├── mimic-cxr-reports
│   └── files
│       ├── p10
│       ├── p11
│       ├── p12
│       ├── p13
│       ├── p14
│       ├── p15
│       ├── p16
│       ├── p17
│       ├── p18
│       └── p19
├── create_section_files.py
├── make_mimic_cxr.py
├── section_parser.py
├── mimic-cxr-2.0.0-split.csv
├── mimic-cxr-2.0.0-metadata.csv
└── README.md
```

Then run the command:

```
python create_section_files.py --no_split --reports_path ./mimic-cxr-reports/files --output_path ./ 
```

Then 

```
python make_mimic_cxr.py --task [rrg,rrs,selfsup]
```

#### Images
You must a physionet account with permissions to download MIMIC-CXR Database. 
Download images from https://physionet.org/content/mimic-cxr-jpg/2.0.0/ 

The downloaded `files` folder must be stored in the root `data/images/mimic-cxr-images-512` folder of ViLMedic. 
You are free to resize the images using the following transform:
``` 
transforms.Compose([transforms.Resize(512)])        
```

**Warning: do not use `transforms.Resize(512,512)`**        


Your `data` tree should now look like this:

```
data
├── images
│   └── mimic-cxr-images-512
│       └── files
│            ├── p10
│            ├── ...
├── SELECTED_TASK
│   └── mimic-cxr
│       ├── test.report.tok
│       ├── test.image.tok
│       ├── train.report.tok
│       ├── train.image.tok
│       ├── validate.report.tok
│       └── validate.image.tok
```

#### Labels

For any report file, you can compute the labels using the following script:
```
python get_chexbert_label.py {PATH_TO_REPORT_FILE}
```
Example : 
```
python get_chexbert_label.py /home/user/vilmedic/data/SELFSUP/mimic-cxr/train.report.tok
```
The ouput of the script will be `/home/user/vilmedic/data/SELFSUP/mimic-cxr/train.label.tok` with all the labels.


