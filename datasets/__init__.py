from .sky_datasets import Sky
from torchvision import transforms
from .taichi_datasets import Taichi
from datasets import video_transforms
from .ucf101_datasets import UCF101
from .ffs_datasets import FaceForensics
from .ffs_image_datasets import FaceForensicsImages
from .sky_image_datasets import SkyImages
from .ucf101_image_datasets import UCF101Images
from .taichi_image_datasets import TaichiImages
from .satellite_image_dataset import SatelliteImages
from .satellite_datasets import Satellite
from .satellite_datasets_deforestation import Satellite_deforestation
from .satellite_datasets_with_changed_year_embed import Satellite_alt_year
from .satellite_datasets_only_image_cond import Satellite_only_image
from .satellite_datasets_cyprus_coastal_ndwi import Satellite_Cyprus_Coastal_NDWI
def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 'ffs':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensics(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ffs_img':
        transform_ffs = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return FaceForensicsImages(args, transform=transform_ffs, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return UCF101(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'ucf101_img':
        transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return UCF101Images(args, transform=transform_ucf101, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return Taichi(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'taichi_img':
        transform_taichi = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return TaichiImages(args, transform=transform_taichi, temporal_sample=temporal_sample)
    elif args.dataset == 'sky':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Sky(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == 'sky_img':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return SkyImages(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == 'sat_img':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return SatelliteImages(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == 'sat':  
        transform_sky = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Satellite(args, transform=transform_sky, temporal_sample=temporal_sample)
    elif args.dataset == "deforestation":
        transform_deforestation = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Satellite_deforestation(args, transform=transform_deforestation, temporal_sample=temporal_sample)
    elif args.dataset == "sat_alt_year":
        transform_deforestation = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Satellite_alt_year(args, transform=transform_deforestation, temporal_sample=temporal_sample)
    elif args.dataset == "sat_only_img":
        transform_deforestation = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Satellite_only_image(args, transform=transform_deforestation, temporal_sample=temporal_sample)
    
    elif args.dataset == "cyprus_coastal_ndwi":
        transform_deforestation = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        return Satellite_Cyprus_Coastal_NDWI(args, transform=transform_deforestation, temporal_sample=temporal_sample)
    
    else:
        raise NotImplementedError(args.dataset)