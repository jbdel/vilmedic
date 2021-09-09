from vilmedic.networks.blocks.others.mcan import MCA_ED
from vilmedic.networks.blocks.others.mcan import LayerNorm
from vilmedic.networks.blocks.others.mcan.make_mask import make_mask
from vilmedic.networks.blocks.others.mcan import AttFlat

from vilmedic.networks.blocks.classifier.evaluation import evaluation
from vilmedic.networks.blocks.classifier.losses import get_loss
import torch.nn as nn


class SIM_MCAN(nn.Module):
    def __init__(self, visual, adapter, linguistic, answer_size, loss, **kwargs):
        super(SIM_MCAN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=linguistic.TOKEN_SIZE,
            embedding_dim=linguistic.WORD_EMBED_SIZE
        )

        visual_func = visual.pop('proto')
        self.cnn = eval(visual_func)(**visual)

        self.adapter = nn.Sequential(
            nn.Linear(adapter.pop('input_size'), adapter.pop('output_size')),
            # torch.nn.LayerNorm(linguistic.hidden_size, eps=linguistic.layer_norm_eps)
        )

        # # Loading the GloVe embedding weights
        # if __C.USE_GLOVE:
        #     self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=linguistic.WORD_EMBED_SIZE,
            hidden_size=linguistic.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.backbone = MCA_ED(linguistic)

        # Flatten to vector
        self.attflat_img = AttFlat(linguistic)
        self.attflat_lang = AttFlat(linguistic)

        # Classification layers
        self.proj_norm = LayerNorm(linguistic.FLAT_OUT_SIZE)
        self.proj = nn.Linear(linguistic.FLAT_OUT_SIZE, answer_size)

        self.loss_func = get_loss(loss.pop('proto'), **loss).cuda()
        # Evaluation
        self.eval_func = evaluation

    def forward(self, images, input_ids, attention_mask, labels, **kwargs):
        # Pre-process Language Feature
        input_ids = input_ids.cuda()
        images = images.cuda()

        img_feat = self.cnn(images)

        img_feat = self.adapter(img_feat)
        lang_feat_mask = make_mask(input_ids.unsqueeze(2))
        lang_feat = self.embedding(input_ids)
        lang_feat, _ = self.lstm(lang_feat)

        img_feat_mask = make_mask(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        output = self.proj(proj_feat)

        loss = self.loss_func(output, labels.cuda(), **kwargs)
        
        return {'loss': loss, 'output': output}
