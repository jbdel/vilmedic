# -*- coding: utf-8 -*-
import torch.nn.functional as F
from .decoder import ConditionalDecoder
from .layers.attention import Attention
from .layers.hierarchical_attention import HierarchicalAttention


def get_rnn_hidden_state(h):
    """Returns h_t transparently regardless of RNN type."""
    return h if not isinstance(h, tuple) else h[0]


class VisualConditionalDecoder(ConditionalDecoder):

    def __init__(self, visual_ctx_size, **kwargs):
        super().__init__(**kwargs)

        self.visual_ctx_size = visual_ctx_size

        # Create attention layer
        self.visual_att = Attention(self.visual_ctx_size, self.hidden_size,
                                    transform_ctx=self.transform_ctx,
                                    mlp_bias=self.mlp_bias,
                                    att_type=self.att_type,
                                    att_activ=self.att_activ,
                                    att_bottleneck=self.att_bottleneck,
                                    temp=self.att_temp)

        self.fusion = HierarchicalAttention([self.hidden_size, self.hidden_size],
                                            self.hidden_size, self.hidden_size)

    def f_next(self, ctx_dict, y, h):
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        self.txt_alpha_t, txt_z_t = self.att(
            h1.unsqueeze(0), *ctx_dict['enc'])

        self.img_alpha_t, img_z_t = self.visual_att(
            h1.unsqueeze(0), *ctx_dict['feats'])

        self.h_att, z_t = self.fusion([txt_z_t, img_z_t], h1.unsqueeze(0))

        h2_c2 = self.dec1(z_t, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        logit = self.hid2out(h2)

        if self.dropout_out > 0:
            logit = self.do_out(logit)

        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        return log_p, self._rnn_pack_states(h2_c2)
