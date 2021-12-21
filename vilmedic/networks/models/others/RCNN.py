import torch.nn as nn
from vilmedic.networks.models.utils import get_n_params
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from .coco.coco_eval import CocoEvaluator
from .coco.coco_utils import get_coco_api_from_dataset
from tqdm import tqdm
import numpy as np



def evaluation(models, config, dl, **kwargs):
    losses = []
    pbar = tqdm(dl, total=len(dl))
    iou_types = ["bbox"]
    coco = get_coco_api_from_dataset(dl.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for batch in pbar:
        results = [model(**batch) for model in models]
        results = results[0]
        res = {target["image_id"].item(): output for target, output in zip(batch["targets"], results)}
        coco_evaluator.update(res)
        break

    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    troll
    return {'loss': loss}

class RCNN(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        backbone = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=5)
        self.model = FasterRCNN(backbone,
                                num_classes=14)
        # Evaluation
        self.eval_func = evaluation
        # self.eval_func = beam_search

    def forward(self, images, targets, **kwargs):
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        outputs = self.model(images.cuda(), targets=targets)

        if not self.training:
            return [{k: v.cpu() for k, v in t.items()} for t in outputs]
        else:
            return {'loss': sum(loss for loss in outputs.values())}

    def __repr__(self):
        s = "RCNN\n"
        s += str(self.model) + '\n'
        s += "{}\n".format(get_n_params(self))
        return s
