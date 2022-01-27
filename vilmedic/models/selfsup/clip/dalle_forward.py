import torch.nn.functional as F
import torch
from einops import rearrange

def prob_mask_like(shape, prob, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_empty(t):
    return t.nelement() == 0

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def forward(
        self,
        text,
        image=None,
        return_loss=False,
        null_cond_prob=0.,
        cache=None,
):
    assert text.shape[
               -1] == self.text_seq_len, f'the length {text.shape[-1]} of the text tokens you passed in does not have the correct length ({self.text_seq_len})'
    batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len

    # randomly remove text condition with <null_cond_prob> probability

    if null_cond_prob > 0:
        null_mask = prob_mask_like((batch,), null_cond_prob, device=device)
        text *= rearrange(~null_mask, 'b -> b 1')

    # make sure padding in text tokens get unique padding token id

    text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
    text = torch.where(text == 0, text_range, text)

    # add <bos>

    text = F.pad(text, (1, 0), value=0)

    tokens = self.text_emb(text)
    tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))

    seq_len = tokens.shape[1]

    if exists(image) and not is_empty(image):
        is_raw_image = len(image.shape) == 4

        if is_raw_image:
            image_size = self.vae.image_size
            assert tuple(image.shape[1:]) == (
            3, image_size, image_size), f'invalid image of dimensions {image.shape} passed in during training'

            image = self.vae.get_codebook_indices(image)

        image_len = image.shape[1]
        image_emb = self.image_emb(image)

        image_emb += self.image_pos_emb(image_emb)

        tokens = torch.cat((tokens, image_emb), dim=1)

        seq_len += image_len

    # when training, if the length exceeds the total text + image length
    # remove the last token, since it needs not to be trained

    if tokens.shape[1] > total_seq_len:
        seq_len -= 1
        tokens = tokens[:, :-1]

    if self.stable:
        alpha = 0.1
        tokens = tokens * alpha + tokens.detach() * (1 - alpha)

    if exists(cache) and cache.get('offset'):
        tokens = tokens[:, -1:]
    out = self.transformer(tokens, cache=cache)

    if self.stable:
        out = self.norm_by_max(out)

    logits = self.to_logits(out)

    # mask logits to make sure text predicts text (except last token), and image predicts image

    logits_mask = self.logits_mask[:, :seq_len]
    if exists(cache) and cache.get('offset'):
        logits_mask = logits_mask[:, -1:]
    max_neg_value = -torch.finfo(logits.dtype).max
    logits.masked_fill_(logits_mask, max_neg_value)

    if exists(cache):
        cache['offset'] = cache.get('offset', 0) + logits.shape[1]

    if not return_loss:
        return logits, text, image

    assert exists(image), 'when training, image must be supplied'

    offsetted_image = image + self.num_text_tokens
    labels = torch.cat((text[:, 1:], offsetted_image), dim=1)

    logits = rearrange(logits, 'b n c -> b c n')

    loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])
    loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])

    loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)
    return loss