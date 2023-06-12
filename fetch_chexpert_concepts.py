import copy 
import os
import numpy as np
from tqdm import tqdm 

DATA_DIR = '/home/cvanuden/git-repos/conceptGPT/data/rrg/chexpert'
CHEXPERT_CONCEPTS = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'effusion']

def load_file(path):
    """Default loading function, which loads nth sentence at line n.
    """
    with open(path, 'r') as f:
        content = f.readlines()
    return [s.strip("\n") for s in content]


for split in ['test']:
    for key in ['all_concepts']:
        print(split, key)

        concept_path = os.path.join(DATA_DIR, f'{split}.{key}.tok')
        concepts_list = load_file(concept_path)
        concepts_list = [l.strip('\n').split(',') for l in concepts_list]

        new_concepts_list = []
        new_concepts_set = set()
        num_concepts = []
        for concepts_for_image in concepts_list:
            new_concepts_for_image = [c for c in concepts_for_image if any([chexpert_concept in c for chexpert_concept in CHEXPERT_CONCEPTS])]
            new_concepts_list.append(new_concepts_for_image)

            # bookkeeping
            num_concepts.append(len(new_concepts_for_image))
            for concept in new_concepts_for_image:
                new_concepts_set.add(concept)

        print(len(new_concepts_set), list(new_concepts_set)[:10])
        print(np.mean(num_concepts), np.min(num_concepts), np.max(num_concepts), np.std(num_concepts))
        print(f"{len([c for c in num_concepts if c == 0])}/{len(num_concepts)} have no concepts")

        # # filter to have at least 4 concepts
        # indices = [i for i in range(len(new_concepts_list)) if len(new_concepts_list[i]) >= 4]

        # print(len(indices))

        # indices = indices[:100]

        dst_concept_path = os.path.join(DATA_DIR, f'{split}.report_radgraph_concepts.tok')
        print(dst_concept_path)
        with open(dst_concept_path, 'w') as f:
            for i, line in enumerate(new_concepts_list):
                line_to_write = ','.join(line)
                line_to_write = line_to_write if len(line_to_write) > 0 else "no finding"
                f.write(f"{line_to_write}\n")