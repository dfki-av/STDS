import os
import torch
import random
import torch.utils.data as data

import numpy as np
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Satellite_alt_year(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):
        # 
        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        self.data_all = self.load_video_frames(self.data_path)

    def __getitem__(self, index):

        vframes = self.data_all[index]
        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.target_video_len, dtype=int) # start, stop, num=50

        select_video_frames = vframes[frame_indice[0]: frame_indice[-1]+1: self.frame_interval] 

        video_frames = []
        frame_years = []
        frame_months = []
        delta_t_years = [torch.tensor(0)]
        frame_metadata = {
            "frame_years":[],
            "delta_t_years": [],
            "frame_months":[]
        }

        for path in select_video_frames:
            # video_frame = torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0)
            video_frame = torch.as_tensor(np.array(Image.open(path).convert('RGB'), dtype=np.uint8, copy=True)).unsqueeze(0)
            # print("video frame", video_frame.shape)

            video_frames.append(video_frame)
            # frame_name_splits = path.split("/")[-1].split(".") # use this for HQ dataset
            frame_name_splits = path.split("/")[-1].split("_") # use this for .jpg dataset

            frame_year = frame_name_splits[0]
            frame_month = frame_name_splits[1] # for sentinel-2 dataset with months data
            # frame_month = 0 # for hq dataset with only years data
            # print("frame name splits", frame_name_splits)
            # print("frame year", frame_year)
            year_normalized = self.normalize_year(frame_year)
            month = int(frame_month)
            # print("month:", month)
            # encoded_month = self.cyclic_encode_month(month)
            # month = int(frame_month)
            # print("frame year", frame_year, "normalized year", year_normalized, year_normalized.shape)
            # frame_years.append(year_normalized)
            # frame_metadata["frame_years"].append(year_normalized)
            # frame_metadata["frame_months"].append(torch.tensor(month))
            frame_years.append(year_normalized)
            frame_months.append(torch.tensor(month))
            # print("Frame months in get item:", frame_metadata)
        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        # print("video_clip_shape", video_clip.shape)
        video_clip = self.transform(video_clip)
        # Use the next line if everything needs to be put toether into a single tensor and
        # the entire tensor is used to as a condition for the video. Make sure to perform
        #  .unsqueeze() before appending the tensors into the list
        delta_t_years.extend(self.calculate_delta_t(frame_years))
        delta_t_years = self.normalize_delta_t(delta_t_years)
        frame_metadata = {
            "frame_years": torch.stack(frame_years).unsqueeze(-1),
            "frame_months": torch.stack(frame_months).unsqueeze(-1),
            "delta_t_years": torch.stack(delta_t_years).unsqueeze(-1)
        }
        # frame_years = torch.cat(frame_years)
        # print("Select Video frames:", select_video_frames)
        return {'video': video_clip, 'video_name': 1, "frame_metadata": frame_metadata}

    def __len__(self):
        return self.video_num
    
    def normalize_year(self, year, base_year = 1997, max_year = 2024):
        # print("year", year, "base_year:", base_year, "max year:", max_year, "neum:", int(year) - base_year, "denom:", max_year - base_year)
        year_norm = (int(year) - base_year)/ (max_year - base_year)
        # (year - min_year) / (max_year - min_year) 
        # print("year after norm:",torch.tensor(year_norm))
        # self.unnormalize_year_norm(year_norm=year_norm)
        return torch.tensor(year_norm)
    
    def normalize_month(self, month, scale = 1000):
            month = int(month) / 12 * scale
            return torch.tensor(month)

    def unnormalize_year(self, year, base_year=1997, max_year=2024):
        year = year * (max_year - base_year) + base_year
        return year
        # year = year / scale * (2100 - base_year) + (base_year if is_print else 0)
        # print("unnoormalized year", year)
        # return torch.tensor([year])

    def calculate_delta_t(self, time_unit):
        delta_t = [self.unnormalize_year(time_unit[i].item()) - self.unnormalize_year(time_unit[i-1].item()) for i in range(1, len(time_unit))]
        return delta_t
    
    def normalize_delta_t(self, delta_t):
        min_value , max_value = min(delta_t), max(delta_t)
        normalized_delta_t = [torch.tensor((delta_t_value - min_value)/(max_value - min_value)) for delta_t_value in delta_t]
        return normalized_delta_t
    
    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        print("Latte Deforestation dataset", dataroot)
        
        for _, meta in enumerate(frame_list):
            root = meta[0]
            # print("meta", meta)
            try:
                # frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1])) # Use this for the hq dataset
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[0]) if item.split(".")[1] == "png" else 0) # Use this for the sentinel 2 /landsat dataset 
                # frames.pop(0) if len
                if(len(frames) and frames[0].split(".")[1] == "json"):
                    frames.pop(0)
            except Exception as e:
                print("Exception",e )
                # print(meta[0]) # root
                # print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            if len(frames) > max(0, self.target_video_len * self.frame_interval): # need all > (16 * frame-interval) videos
            # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                data_all.append(frames)
        self.video_num = len(data_all)
        # 
        return data_all
    

if __name__ == '__main__':

    import argparse
    import torchvision
    import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=4)
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/sky_timelapse/sky_train/")
    config = parser.parse_args()


    target_video_len = config.num_frames

    temporal_sample = video_transforms.TemporalRandomCrop(target_video_len * config.frame_interval)
    trans = transforms.Compose([
        video_transforms.ToTensorVideo(),
        # video_transforms.CenterCropVideo(256),
        video_transforms.CenterCropResizeVideo(256),
        # video_transforms.RandomHorizontalFlipVideo(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    taichi_dataset = Sky(config, transform=trans, temporal_sample=temporal_sample)
    print(len(taichi_dataset))
    taichi_dataloader = data.DataLoader(dataset=taichi_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, video_data in enumerate(taichi_dataloader):
        print(video_data['video'].shape)
        
        # print(video_data.dtype)
        # for i in range(target_video_len):
        #     save_image(video_data[0][i], os.path.join('./test_data', '%04d.png' % i), normalize=True, value_range=(-1, 1))

        # video_ = ((video_data[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
        # torchvision.io.write_video('./test_data' + 'test.mp4', video_, fps=8)
        # exit()