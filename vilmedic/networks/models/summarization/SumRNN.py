import torch.nn as nn
import torch
import os
import numpy as np
from vilmedic.networks.blocks.rnn.textencoder import TextEncoder
from vilmedic.networks.blocks.rnn.decoder import ConditionalDecoder
from vilmedic.networks.blocks.rnn.visualdecoder import VisualConditionalDecoder
from vilmedic.networks.blocks.vision import *
from vilmedic.networks.blocks.rnn.evaluation import beam_search
from vilmedic.networks.models.utils import get_n_params


class SumRNN(nn.Module):
    def __init__(self, encoder, decoder, cnn=None, **kwargs):
        super().__init__()
        self.kwargs = kwargs

        encoder_func = encoder.proto
        decoder_func = decoder.proto

        self.enc = eval(encoder_func)(**encoder)
        self.dec = eval(decoder_func)(**decoder, encoder_size=self.enc.ctx_size)

        # Evaluation
        self.eval_func = beam_search

        self.reset_parameters()

        self.cnn = None
        if cnn is not None:
            cnn_func = cnn.pop('proto')
            self.visual_projection = nn.Linear(cnn.pop("visual_embedding_dim"), self.dec.hidden_size)
            self.cnn = eval(cnn_func)(**cnn)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)

        if 'src_emb' in self.kwargs:
            self.set_embeddings(self.kwargs['src_emb'], self.enc.emb)
        if 'tgt_emb' in self.kwargs:
            self.set_embeddings(self.kwargs['tgt_emb'], self.dec.emb)

        if hasattr(self, 'enc') and hasattr(self.enc, 'emb'):
            # Reset padding embedding to 0
            with torch.no_grad():
                self.enc.emb.weight.data[0].fill_(0)

    def encode(self, input_ids, feats=None, images=None, **kwargs):
        # RNN model is batch_first = False
        input_ids = input_ids.permute(1, 0)

        if feats is not None:
            feats = feats.cuda().permute(1, 0, 2)  # RNN takes (n, bs, feat)
            return {'enc': self.enc(input_ids, feats), 'feats': (feats, None)}
        elif images is not None:
            with torch.no_grad():
                feats = self.cnn(images.cuda())
            feats = self.visual_projection(feats)
            feats = feats.permute(1, 0, 2)  # RNN takes (n, bs, feat)
            return {'enc': self.enc(input_ids), 'feats': (feats, None)}
        else:
            return {'enc': self.enc(input_ids)}

    def decode(self, enc_outputs, decoder_input_ids):
        # RNN model is batch_first = False
        decoder_input_ids = decoder_input_ids.permute(1, 0)
        result = self.dec(enc_outputs, decoder_input_ids)
        result['n_items'] = torch.nonzero(decoder_input_ids[1:]).shape[0]
        return result

    def forward(self, input_ids, decoder_input_ids, **kwargs):
        input_ids = input_ids.cuda()
        decoder_input_ids = decoder_input_ids.cuda()

        enc_outputs = self.encode(input_ids, **kwargs)
        result = self.decode(enc_outputs, decoder_input_ids)

        result['loss'] = result['loss'] / result['n_items']

        return result

    def set_embeddings(self, path, obj):
        filename = os.path.basename(path)
        embs = np.load(path, allow_pickle=True)
        assert len(embs) == obj.weight.size(0)
        success = 0
        with torch.no_grad():
            for i, emb in enumerate(embs):
                if emb is None:
                    continue
                obj.weight.data[i] = torch.from_numpy(emb)
                success += 1
        print(filename, success, '/', len(embs), 'words loaded')

    def __repr__(self):
        s = super().__repr__() + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
