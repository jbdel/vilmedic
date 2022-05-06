import torch
from vilmedic.blocks.scorers.scores import REWARD_COMPLIANT
import inspect
import torch.nn.functional as F
import json


def scst_loss(input,
              seq,
              reward_sampling,
              reward_greedy,
              scores_weights,
              pad_token_id):
    N, L = input.shape[:2]
    # HuggingFace TopKLogitsWarper (if top_k > 0) puts -float('inf') for non top_k (or if we use bad_words_ids)
    # Padding can then have logits -float('inf') though we have to pad because hyp generation is finished
    # -float('inf') * 0 masking results in NaN so that doesnt mitigate the issue. need to do:
    input[input == -float("Inf")] = 0.
    #####
    delta_reward = [torch.tensor(rs).cuda() - torch.tensor(rg).cuda() for rs, rg in zip(reward_sampling, reward_greedy)]
    reward = delta_reward[0]

    # reward

    def other(input, reward, mask):
        input = input.squeeze(-1)
        input = input * mask
        input = input / torch.sum(mask)

        loss = [scores_weights[i] * (-input * r.view(N, 1).expand_as(input)) for i, r in enumerate(reward)]
        loss = sum([torch.sum(l) for l in loss])
        print("here", (loss))

    other(input, delta_reward, (seq > pad_token_id).float())
    reward = reward.view(N, 1, 1).expand_as(input)
    input = -input * reward

    # masking
    mask = (seq > pad_token_id).float()
    output = input.view(-1) * mask.view(-1)

    # mean
    output = torch.sum(output) / torch.sum(mask)
    print(output)
    troll
    #
    # for rs, rg in zip(reward_sampling, reward_greedy):
    #     delta_reward = torch.tensor(reward_sampling).cuda() - torch.tensor(reward_greedy).cuda()

    return output, delta_rewatd


class SCST:
    def __init__(self, decoder, dl, scores, scores_args=None, scores_weights=None, top_k=None):

        self.tokenizer = dl.dataset.tokenizer
        self.decoder = decoder
        self.top_k = top_k
        self.bos_token_id = self.decoder.config.bos_token_id
        self.eos_token_id = self.decoder.config.eos_token_id
        self.pad_token_id = self.decoder.config.pad_token_id
        self.max_length = dl.dataset.tokenizer_max_len
        self.scores = scores
        self.scores_args = scores_args
        self.scores_weights = scores_weights

        if not isinstance(scores, list):
            scores = [scores]

        assert all([score in REWARD_COMPLIANT for score in scores]), "{} not in {}".format(scores,
                                                                                           REWARD_COMPLIANT.keys())
        # Scores weights
        if len(scores) > 1:
            assert scores_weights is not None, "You need to mention scores_weights"
            assert isinstance(scores_weights, list), "scores_weights must be a list"
            assert len(scores_weights) == len(scores), "You need to mention as much scores_weights as scores"
        else:
            self.scores_weights = [1.0]

        # Scores args
        if scores_args is not None:
            if not isinstance(scores_args, list):
                scores_args = [scores_args]
            assert len(scores_args) == len(scores), \
                "You need to mention as much scores_args as scores (i.e. [{arg1:value}, {}, {arg1:value}])"
        else:
            self.scores_args = [None] * len(scores)

        self.scorers = []
        self.scorers_index = []
        for index, score in enumerate(scores):
            scorer, scorer_index = REWARD_COMPLIANT[score]
            if self.scores_args[index] is not None:
                scorer = scorer(**self.scores_args[index])
            self.scorers.append(scorer)
            self.scorers_index.append(scorer_index)

    def forward_greedy(self, input_ids, encoder_hidden_states, encoder_attention_mask):
        assert not torch.is_grad_enabled(), "Please add torch.no_grad() decorator"
        batch_size = input_ids.shape[0]

        out = self.decoder.generate(
            input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
            max_length=self.max_length,
            num_beams=1,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            encoder_hidden_states=encoder_hidden_states.detach(),
            encoder_attention_mask=encoder_attention_mask.detach(),
            forced_eos_token_id=True,
            use_cache=True,
        )
        greedy_input_ids = out.sequences
        reward_greedy, hyp_list, ref_list = self.get_reward(greedy_input_ids.detach().data, input_ids)
        return reward_greedy, hyp_list, ref_list

    def forward_sampling(self, input_ids, encoder_hidden_states, encoder_attention_mask, reward_greedy):
        assert torch.is_grad_enabled()
        batch_size = input_ids.shape[0]

        out = inspect.unwrap(self.decoder.generate)(  # inspect.unwrap removes the torch.no_grad() decorator
            self=self.decoder,
            input_ids=torch.ones((batch_size, 1), dtype=torch.long).cuda() * self.bos_token_id,
            max_length=self.max_length,
            num_beams=1,
            num_return_sequences=1,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            bad_words_ids=[[self.pad_token_id], [self.bos_token_id]],
            top_k=self.top_k,
            forced_eos_token_id=True,
            output_scores=True,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
        )
        sampled_ids = out.sequences[:, 1:].contiguous()
        logits = torch.stack(out.scores, dim=1)
        logits = F.log_softmax(logits, dim=-1)
        sampled_logits = logits.gather(2, sampled_ids.unsqueeze(-1))

        reward_sampling, hyp_list, _ = self.get_reward(sampled_ids.data, input_ids)
        loss, delta_reward = scst_loss(sampled_logits,
                                       sampled_ids.data,
                                       reward_sampling,
                                       reward_greedy,
                                       self.scores_weights,
                                       self.pad_token_id)
        return loss, delta_reward, reward_sampling, hyp_list

    def get_reward(self, rollout_input_ids, input_ids):
        hyp_list = []
        ref_list = []
        for h, r in zip(rollout_input_ids, input_ids):
            hyp_list.append(self.tokenizer.decode(h, skip_special_tokens=True, clean_up_tokenization_spaces=False))
            ref_list.append(self.tokenizer.decode(r, skip_special_tokens=True, clean_up_tokenization_spaces=False))

        reward = [scorer(ref_list, hyp_list)[scorer_index] for scorer, scorer_index in
                  zip(self.scorers, self.scorers_index)]
        return reward, hyp_list, ref_list

    def __repr__(self):
        s = "SCST\n"
        s += json.dumps({
            'Scores': str(self.scores),
            'scores_args': str(self.scores_args),
            'scores_weights': str(self.scores_weights),
            'Generate': {'top_k': self.top_k},
        }, indent=4)
        return s
