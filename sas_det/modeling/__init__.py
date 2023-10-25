# Copyright (c) NEC Laboratories America, Inc.

from .backbone import (
    build_clip_language_encoder,
    get_clip_tokenzier,
    get_clip_image_transform,
)

from .meta_arch import clip_rcnn as _

from .roi_heads import (
    CLIPRes5ROIHeads,
    FastRCNNOutputLayers,
)
