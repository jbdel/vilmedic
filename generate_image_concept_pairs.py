import copy 
import os
import numpy as np
from tqdm import tqdm 

DATA_DIR = '/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/'
DST_DATA_DIR = os.path.join(DATA_DIR, 'concepts_augmented')
N_AUGMENTATIONS = 10

os.makedirs(DST_DATA_DIR, exist_ok=True)

def load_file(path):
    """Default loading function, which loads nth sentence at line n.
    """
    with open(path, 'r') as f:
        content = f.read().strip()
    return [s for s in content.split('\n')]


for split in ['validate']:
    for key in ['concat_concepts']:
        print(split, key)

        concept_path = os.path.join(DATA_DIR, 'concepts', f'{split}.{key}.tok')
        concepts_list = load_file(concept_path)
        concepts_list = [l.strip().split(',') for l in concepts_list]

        image_path = os.path.join(DATA_DIR, 'findings', f'{split}.image.tok')
        image_list = load_file(image_path)


        new_concepts_list = []
        new_image_list = []
        print(len(image_list), len(new_image_list))
        for i, concepts in tqdm(enumerate(concepts_list), total=len(concepts_list)):
            for seed in range(N_AUGMENTATIONS):
                concepts_copy = copy.deepcopy(concepts)
                image_copy = copy.deepcopy(image_list[i])
                
                np.random.seed(seed)
                np.random.shuffle(concepts_copy)

                new_concepts_list.append(concepts_copy)
                new_image_list.append(image_copy)

        print(len(image_list), len(new_image_list))

        dst_concept_path = os.path.join(DST_DATA_DIR, f'{split}.{key}.tok')
        with open(dst_concept_path, 'w') as f:
            for line in new_concepts_list:
                line_to_write = ','.join(line)
                f.write(f"{line_to_write}\n")

        dst_image_path = os.path.join(DST_DATA_DIR, f'{split}.image.tok')
        with open(dst_image_path, 'w') as f:
            for line in new_image_list:
                # line_to_write = ','.join(line[key])
                f.write(f"{line}\n")

