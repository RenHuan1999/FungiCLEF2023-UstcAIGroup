from timm.models import create_model  
from .MetaFG import *
from .MetaFG_meta import *
from .convnext import *
from .convnext_dynamic_mlp import *
from .beit_vit import *
from .beit_vit_dyn import *

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'MetaFG':
        model = create_model(
                config.MODEL.NAME,
                pretrained=False,
                num_classes=config.MODEL.NUM_CLASSES,
                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                img_size=config.DATA.IMG_SIZE,
                only_last_cls=config.MODEL.ONLY_LAST_CLS,
                extra_token_num=config.MODEL.EXTRA_TOKEN_NUM,
                meta_dims=config.MODEL.META_DIMS,
                use_arcface=config.TRAIN.USE_ARCFACE,
                never_mask=config.TRAIN.NEVER_MASK,
        )
    elif model_type == "convnext":
        model = create_model(
                config.MODEL.NAME,
                pretrained=False,
                num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == "convnext_dyn":
        model = create_model(
                config.MODEL.NAME,
                config=config,
                pretrained=False,
                num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == "beit_vit":
        model = create_model(
                config.MODEL.NAME,
                pretrained=False,
                num_classes=config.MODEL.NUM_CLASSES,
        )
    elif model_type == "beit_vit_dyn":
        model = create_model(
                config.MODEL.NAME,
                config=config,
                pretrained=False,
                num_classes=config.MODEL.NUM_CLASSES,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
