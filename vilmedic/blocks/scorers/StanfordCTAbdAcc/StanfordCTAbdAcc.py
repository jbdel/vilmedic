import torch.nn as nn

labels_lists = {
    "radiologist_labels_from_n_grams": {'focal splenic lesions', 'pericholecystic fluid', 'dilated loops of small',
                                        'spleen is enlarged , measuring',
                                        'for biliary duct dilatation', 'gallstones without ct', 'spleen is enlarged',
                                        'gallbladder wall thickening',
                                        'lesion in the pancreatic', 'evidence of cholecystitis', 'surgically absent',
                                        'ct evidence of acute cholecystitis',
                                        'ct evidence of cholecystitis', 'thickening or pericholecystic fluid',
                                        'upper limits of normal',
                                        'prostate is mildly enlarged', 'acute cholecystitis',
                                        'thickening or pericholecystic',
                                        'pancreatic ductal dilatation', 'the prostate is mildly', 'wall thickening',
                                        'findings of cholecystitis',
                                        'bowel obstruction', 'the spleen is enlarged', 'gallstones',
                                        'the upper limits of normal',
                                        'evidence of acute cholecystitis', 'the spleen is enlarged measuring',
                                        'pancreatic ductal dilation',
                                        'gallstones no extrahepatic biliary dilatation',
                                        'wall thickening or pericholecystic',
                                        'calcification is demonstrated the pancreatic', 'of cholecystitis',
                                        'wall thickening or pericholecystic fluid',
                                        'the spleen is enlarged ,', 'lymphadenopathy by size criteria'},
    "chatgpt_labels_from_n_grams": {
        'gallbladder',
        'gallbladder wall',
        'cholecystitis',
        'pericholecystic fluid',
        'ct findings',
        'gallstones',
        'biliary duct dilatation',
        'lymphadenopathy',
        'prostate',
        'uterus',
        'small bowel',
        'bowel obstruction',
        'seminal vesicles',
        'right lower quadrant',
        'spleen',
        'pancreas',
        'pancreatic duct',
        'pancreatic ductal dilatation',
        'pancreatic head',
        'pancreatic tail',
        'pancreatic body',
        'no focal lesions',
        'no obstruction',
        'normal size',
        'normal appearance',
        'mildly enlarged',
        'enlarged',
        'absent',
        'evidence of',
        'no evidence of',
        'dilated',
        'not dilated'
    }
}


class StanfordCTAbdAcc(nn.Module):
    def __init__(self, **kwargs):
        super(StanfordCTAbdAcc, self).__init__()

    def forward(self, hyps, refs):

        assert len(refs) == len(hyps), "refs and hyps should have the same length"

        scores = {}

        for labels_name, labels_list in labels_lists.items():
            # Keep track of accuracy per sentence
            accuracy_per_sentence = []
            # Iterate through each pair of sentences
            for ref, hyp in zip(refs, hyps):
                # Find the substrings in the ref sentence
                substrings_in_ref = {label for label in labels_list if label in ref}

                # Count how many of these substrings are also in the hyp sentence
                matching_substrings = sum(1 for label in substrings_in_ref if label in hyp)

                # Calculate accuracy for this sentence
                if substrings_in_ref:
                    sentence_accuracy = matching_substrings / len(substrings_in_ref)

                    # Append to the list of accuracies
                    accuracy_per_sentence.append(sentence_accuracy)

            # Calculate the average accuracy over all sentences
            average_accuracy = sum(accuracy_per_sentence) / len(accuracy_per_sentence) if accuracy_per_sentence else 0

            scores[labels_name] = average_accuracy

        scores["averaged"] = sum([v for v in scores.values()]) / len(list(scores.values()))
        return [scores]


if __name__ == '__main__':
    import random
    import json

    refs = [
        'two dominant rim calcified gallstones , measuring up to 2 . 9 cm , are seen within the lumen of the gallbladder . the gallbladder appears small and contracted , accentuating wall thickening .',
        'prominent mesenteric lymph nodes are noted , measuring up to 1 . 0 cm ( 4 : 192 ).',
        'multiple scattered subcentimeter mesenteric lymph nodes are seen .',
        'punctate calcific densities now seen within the gallbladder ( 3 - 161 ) likely reflecting tiny gallstones , without ct evidence of acute cholecystitis .',
        'distended with layering sludge .', 'layering gallstones without ct evidence of acute cholecystitis .',
        'contracted , with evidence of submucosal edema likely reflecting increased portal venous pressures',
        'distended . no wall thickening or pericholecystic fat stranding .',
        'gallstones . gallbladder is dilated . no findings suggestive of acute cholecystitis .',
        'there are numerous low density lymph nodes within the upper abdomen , primarily surrounding the splenic vein , measuring up to 9 mm . increased number of retroperitoneal nodes , the largest an aortocaval node measures 1 . 0 cm ( 4 : 135 ).',
        'the gallbladder is distended , however , there is no evidence of cholelithiasis or wall thickening to suggest cholecystitis .',
        'contracted . cholecystostomy tube has been removed .',
        'large calcified gallstones at the dependent portion of the gallbladder are noted .',
        'biliary excretion of contrast is visualized in the gallbladder . no gallstones .',
        'air - filled gallstones . otherwise normal appearance of the gallbladder .', 'no radiopaque stones .',
        'mild gallbladder wall thickening which may be related to peritoneal carcinomatosis .',
        'the gallbladder slightly distended measuring up to 4 . 6 cm in maximal diameter , likely related to fasting state . no gallbladder wall thickening , pericholecystic fluid , or radiopaque stones identified .',
        'vicarious excretion of contrast .']

    hyps = random.sample(refs, len(refs))
    scores = {"stanford_ct": StanfordCTAbdAcc()(refs=refs, hyps=hyps)[0]}
    print(json.dumps(scores, indent=4))
