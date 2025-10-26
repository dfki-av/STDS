import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .latte import Latte_models
from .latte_img import LatteIMG_models
from .latte_t2v import LatteT2V
# from .latte_inference import Latte_inference_models

from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    if 'LatteIMG' in args.model:
        return LatteIMG_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                fusion_based_condition = args.fusion_based_condition,
                conditioning_type = args.conditioning_type,
                in_channels = args.in_channels,
                months_required = args.months_required,
                stochastic_depth_drop_rate = args.stochastic_depth_drop_rate
            )
    elif 'LatteT2V' in args.model:
        return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)
    elif 'Latte' in args.model:
        return Latte_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                fusion_based_condition = args.fusion_based_condition,
                conditioning_type = args.conditioning_type,
                in_channels = args.in_channels,
                months_required = args.months_required,
                stochastic_depth_drop_rate = args.stochastic_depth_drop_rate,
                attention_drop_rate = args.attention_drop_rate,
                projection_drop_rate = args.projection_drop_rate
            )
    else:
        raise '{} Model Not Supported!'.format(args.model)
    
def get_inference_models(args):
    if 'LatteIMG' in args.model:
        return LatteIMG_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                fusion_based_condition = args.fusion_based_condition,
                conditioning_type = args.conditioning_type,
                stochastic_depth_drop_rate = args.stochastic_depth_drop_rate,
                attention_drop_rate = args.attention_drop_rate,
                projection_drop_rate = args.projection_drop_rate
            )
    elif 'LatteT2V' in args.model:
        return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)
    elif 'Latte' in args.model:
        return Latte_models[args.model](
                input_size=args.latent_size,
                num_classes=args.num_classes,
                num_frames=args.num_frames,
                learn_sigma=args.learn_sigma,
                extras=args.extras,
                fusion_based_condition = args.fusion_based_condition,
                conditioning_type = args.conditioning_type,
                in_channels = args.in_channels,
                months_required = args.months_required,
                stochastic_depth_drop_rate = args.stochastic_depth_drop_rate,
                attention_drop_rate = args.attention_drop_rate,
                projection_drop_rate = args.projection_drop_rate
            )
    else:
        raise '{} Model Not Supported!'.format(args.model)
