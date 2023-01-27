from ..common.optim import SGD as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier
from ..common.data.coco import dataloader
from ..common.models.mask_rcnn_fpn import model
from ..common.train import train

from detectron2.config import LazyCall as L
from detectron2.modeling.backbone.convnext import ConvNeXt


# Replace default ResNet with RegNetX-4GF from the DDS paper. Config source:
# https://github.com/facebookresearch/pycls/blob/2c152a6e5d913e898cca4f0a758f41e6b976714d/configs/dds_baselines/regnetx/RegNetX-4.0GF_dds_8gpu.yaml#L4-L9  # noqa
model.backbone.bottom_up = ConvNeXt(
    in_chans=3,
    depths=[3, 3, 9, 3],
    dims=[96, 192, 384, 768],
    drop_path_rate=0.4,
    layer_scale_init_value=1.0,
    out_indices=[0, 1, 2, 3],
    stem=4
)
model.pixel_std = [57.375, 57.120, 58.395]

optimizer.weight_decay = 5e-5
train.init_checkpoint = None
# RegNets benefit from enabling cudnn benchmark mode
train.cudnn_benchmark = True
