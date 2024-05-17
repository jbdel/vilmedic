import os
import re
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

pair_to_reward_dict = dict()


class GREENModel(nn.Module):
    def __init__(
            self,
            model_id_or_path,
            do_sample=False,
            batch_size=4,
            return_0_if_no_green_score=True,
    ):
        super().__init__()
        self.do_sample = do_sample
        self.batch_size = batch_size
        self.return_0_if_no_green_score = return_0_if_no_green_score
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            trust_remote_code=True,
            device_map={"": "cuda:{}".format(torch.cuda.current_device())},
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_id_or_path,
            add_eos_token=True,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.chat_template = "{% for message in messages %}\n{% if message['from'] == 'human' %}\n{{ '<|user|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'system' %}\n{{ '<|system|>\n' + message['value'] + eos_token }}\n{% elif message['from'] == 'gpt' %}\n{{ '<|assistant|>\n'  + message['value'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        self.categories = [
            "Clinically Significant Errors",
            "Clinically Insignificant Errors",
            "Matched Findings",
        ]

        self.sub_categories = [
            "(a) False report of a finding in the candidate",
            "(b) Missing a finding present in the reference",
            "(c) Misidentification of a finding's anatomic location/position",
            "(d) Misassessment of the severity of a finding",
            "(e) Mentioning a comparison that isn't in the reference",
            "(f) Omitting a comparison detailing a change from a prior study",
        ]

    def make_prompt(self, text1, text2):
        prompt = f"Objective: Evaluate the accuracy of a candidate radiology report in comparison to a reference radiology report composed by expert radiologists.\n\n    Process Overview: You will be presented with:\n\n    1. The criteria for making a judgment.\n    2. The reference radiology report.\n    3. The candidate radiology report.\n    4. The desired format for your assessment.\n\n    1. Criteria for Judgment:\n\n    For each candidate report, determine:\n\n    The count of clinically significant errors.\n    The count of clinically insignificant errors.\n\n    Errors can fall into one of these categories:\n\n    a) False report of a finding in the candidate.\n    b) Missing a finding present in the reference.\n    c) Misidentification of a finding's anatomic location/position.\n    d) Misassessment of the severity of a finding.\n    e) Mentioning a comparison that isn't in the reference.\n    f) Omitting a comparison detailing a change from a prior study.\n    Note: Concentrate on the clinical findings rather than the report's writing style. Evaluate only the findings that appear in both reports.\n\n    2. Reference Report:\n    {text1}\n\n    3. Candidate Report:\n    {text2}\n\n    4. Reporting Your Assessment:\n\n    Follow this specific format for your output, even if no errors are found:\n    ```\n    [Explanation]:\n    <Explanation>\n\n    [Clinically Significant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Clinically Insignificant Errors]:\n    (a) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n    ....\n    (f) <Error Type>: <The number of errors>. <Error 1>; <Error 2>; ...; <Error n>\n\n    [Matched Findings]:\n    <The number of matched findings>. <Finding 1>; <Finding 2>; ...; <Finding n>\n    ```\n"
        return prompt

    def tokenize_batch_as_chat(self, batch):
        batch = [
            self.tokenizer.apply_chat_template(
                i, tokenize=False, add_generation_prompt=True
            )
            for i in batch
        ]

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )

        return batch

    def get_response(self, input_ids, attention_mask):

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=self.do_sample,
            max_length=2048,
            top_p=None,
        )

        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_list = []
        for i in responses:
            if "<|assistant|>" in i:
                i = i.split("<|assistant|>")[-1]
            i = i.replace("</s>", "").replace("<unk>", "")
            response_list.append(i)

        return response_list

    def parse_error_counts(self, text, category, for_reward=False):

        if category not in self.categories:
            raise ValueError(
                f"Category {category} is not a valid category. Please choose from {self.categories}."
            )

        # Pattern to match integers within the category, stopping at the next category or end of text
        pattern = rf"\[{category}\]:\s*(.*?)(?:\n\s*\n|\Z)"
        category_text = re.search(pattern, text, re.DOTALL)

        # Initialize the counts
        sum_counts = 0
        sub_counts = [0 for i in range(6)]

        # If the category is not found, return 0 or None
        if not category_text:
            if self.return_0_if_no_green_score:
                return sum_counts, sub_counts
            else:
                # we need to know whether the category is empty or not, otherwise we overesitmate the reward
                return None, [None for i in range(6)]
        # If the category is found, but the category is empty, return 0
        if category_text.group(1).startswith("No"):
            return sum_counts, sub_counts

        if category == "Matched Findings":
            counts = re.findall(r"^\b\d+\b(?=\.)", category_text.group(1))
            if len(counts) > 0:
                sum_counts = int(counts[0])
            return sum_counts, sub_counts
        # Possible fine-grained error categories for categories Significant and Insignificant Clinical Errors
        else:  # "Clinically Significant Errors" or "Clinically Insignificant Errors"
            # Split each string at the first space and keep only the first part
            sub_categories = [s.split(" ", 1)[0] + " " for s in self.sub_categories]
            # Find all sub_categories in the matched text
            matches = sorted(re.findall(r"\([a-f]\) .*", category_text.group(1)))

            # this is for the gpt-4 template which assigns a number to the subcategories not letters
            if len(matches) == 0:
                matches = sorted(re.findall(r"\([1-6]\) .*", category_text.group(1)))
                sub_categories = [
                    f"({i})" + " " for i in range(1, len(self.sub_categories) + 1)
                ]

            for position, sub_category in enumerate(sub_categories):
                # need to loop over all matches, because the sub_categories are not always in the same order
                for match in range(len(matches)):
                    if matches[match].startswith(sub_category):
                        # If the sub_category is found, insert the count to sub_counts at the ordered position
                        count = re.findall(r"(?<=: )\b\d+\b(?=\.)", matches[match])
                        if len(count) > 0:
                            # take the first number after the colon
                            sub_counts[position] = int(count[0])
            return sum(sub_counts), sub_counts

    def compute_green(self, response):
        # significant clinical errors, we want to look at each error type
        sig_present, sig_errors = self.parse_error_counts(response, self.categories[0])
        # matched findings, we want to look at the sum of all errors
        matched_findings, _ = self.parse_error_counts(response, self.categories[2])

        # set the prior study (sub_categories: (e) Mentioning a comparison that isn't in the reference, (f) Omitting a comparison detailing a change from a prior study) errors to 0
        # sig_errors[-2:] = 0, 0

        if matched_findings == 0:
            return 0

        if (
                sig_present is None or matched_findings is None
        ):  # when the template does not include the key "Clinically Significant Errors"
            return None

        return matched_findings / (matched_findings + sum(sig_errors))

    def forward(self, input_ids, attention_mask):
        print(len(input_ids))
        reward_model_responses = self.get_response(input_ids, attention_mask)

        greens = [self.compute_green(response) for response in reward_model_responses]
        greens = [green for green in greens if green is not None]

        return torch.tensor(greens, dtype=torch.float).cuda()


class GREEN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = GREENModel(**kwargs).cuda()
        self.model.eval()

        print("Using {} GPUs!".format(torch.cuda.device_count()))
        self.model = torch.nn.DataParallel(self.model)

    def forward(self, refs, hyps):
        assert len(refs) == len(hyps)

        with torch.no_grad():
            pairs_to_process = []
            indices_to_process = []
            final_scores = torch.zeros(len(refs))

            for i, (ref, hyp) in enumerate(zip(refs, hyps)):
                if (ref, hyp) in pair_to_reward_dict:
                    final_scores[i] = pair_to_reward_dict[(ref, hyp)]
                else:
                    pairs_to_process.append((ref, hyp))
                    indices_to_process.append(i)

            if pairs_to_process:
                batch = [self.model.module.make_prompt(ref, hyp) for ref, hyp in pairs_to_process]
                batch = [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}] for prompt in batch]
                batch = self.model.module.tokenize_batch_as_chat(batch)

                greens_tensor = self.model(batch['input_ids'], batch['attention_mask'])

                if len(greens_tensor) == len(pairs_to_process):
                    for i, (ref, hyp) in enumerate(pairs_to_process):
                        score = greens_tensor[i]
                        pair_to_reward_dict[(ref, hyp)] = score
                        final_scores[indices_to_process[i]] = score
                else:
                    print("An inconsistency was detected in processing pairs.")

            # Compute mean_green over the entire set of final_scores
            mean_green = final_scores.mean()

            return mean_green, final_scores


if __name__ == '__main__':
    import time

    model = GREEN(
        model_id_or_path="StanfordAIMI/GREEN",
        do_sample=False,  # should be always False
        batch_size=32,
        return_0_if_no_green_score=True,
    )
    x = time.time()

    refs = [
        "Interstitial opacities without changes.",
        "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
        "Bibasilar atelectasis. Otherwise, no acute intrathoracic process.",
        "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
        "Interval resolution of previously seen mild pulmonary edema with trace bilateral pleural effusions.",
        "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
        "Bilateral pleural effusions, large on the right and small on the left. No definite focal consolidation identified, although evaluation is limited secondary to these effusions.",
        "1. Mild left basal atelectasis. Otherwise unremarkable. 2. No definite displaced rib fracture though if there is continued concern dedicated rib series may be performed to further assess.",
    ]
    hyps = [
        "Interstitial opacities at bases without changes.",
        "Interval resolution of previously seen mild pulmonary edema with trace bilateral pleural effusions.",
        "Bibasilar atelectasis. Otherwise, no acute intrathoracic process.",
        "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
        "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
        "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
        "In comparison with the study of ___, the increased opacification at the right base has essentially cleared with better inspiration. Cardiac silhouette remains at the upper limits of normal in size and there is again tortuosity of the aorta without vascular congestion or pleural effusion. Biapical changes, especially on the right, are stable.",
        "1. Mild left basal atelectasis. Otherwise unremarkable.",
    ]

    mean_green, greens = model(refs=refs, hyps=hyps)
    x = time.time() - x
    print(x)
    print("Mean for the given examples is: ", mean_green)
    print("Array for the given examples is: ", greens)
