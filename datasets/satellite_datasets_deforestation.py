import os
import torch
import random
import torch.utils.data as data

import numpy as np
from PIL import Image
from skimage import feature, color, filters, exposure, io
import json
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class Satellite_deforestation(data.Dataset):
    def __init__(self, configs,transform, temporal_sample=None, train=True):
        # 
        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        self.data_all = self.load_video_frames(self.data_path)
        self.use_canny_edge_images = configs.use_canny_edge_images
        self.use_histogram_segmentations = configs.use_histogram_segmentations
        self.use_CLIP_image_embeddings = configs.use_CLIP_image_embeddings
        self.future_to_past = configs.future_to_past
    def __getitem__(self, index):

        # vframes = self.data_all[index]
        vframes = self.data_all[index]["frames"]
        metadata_file_name = self.data_all[index]["metadata"]

        total_frames = len(vframes)

        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, num=self.target_video_len, dtype=int) # start, stop, num=50

        select_video_frames = vframes[frame_indice[0]: frame_indice[-1]+1: self.frame_interval] 
        # select_video_metadata = metadata
        video_frames = []
        edge_image = []
        edge_image_tensors = []
        segmented_images = []
        segmented_image_tensors = []
        frame_metadata = {
            "frame_years":[],
            "frame_months":[],
            # "latitude": [],
            # "longitude": [],
            # "country": [],
            # "ndvi": [],
            "images": []
        }
        # 


        with open(metadata_file_name, 'r') as file:
            metadata_content = json.load(file)
        metadata_content["year_wise_metadata"] = metadata_content["year_wise_metadata"][frame_indice[0]: frame_indice[-1]+1: self.frame_interval]
        if(self.future_to_past):
            select_video_frames = select_video_frames[::-1]
        print("Select video frames", select_video_frames)
        # country_code = get_country_code(metadata_content["country"])
        for i, path in enumerate(select_video_frames):

            numeric_metadata = (metadata_content["year_wise_metadata"][i]["year"], metadata_content["x_init"], metadata_content["y_init"], metadata_content["year_wise_metadata"][i]["ndvi"])
            lat_norm, lon_norm, year_norm, ndvi_norm = self.metadata_normalize(numeric_metadata)
            frame_metadata["frame_years"].append(year_norm)
            frame_metadata["frame_months"].append(torch.tensor(6))
            frame_img = np.array(Image.open(path).convert('RGB'), dtype=np.uint8, copy=True)
            video_frame = torch.as_tensor(frame_img).unsqueeze(0)
            video_frames.append(video_frame)
            if(self.use_canny_edge_images):
                
                gray_frame = np.array(Image.open(path).convert("L"))
                edge_image = feature.canny(gray_frame, sigma=0.0, low_threshold=0.1, high_threshold=0.25)
                edge_tensor = torch.as_tensor(np.array(edge_image, dtype=np.uint8, copy=True)).unsqueeze(0).unsqueeze(-1).permute(0,3,1,2)
                edge_tensor = edge_tensor.repeat(1,3,1,1)
                edge_image_tensors.append(edge_tensor)
            elif(self.use_histogram_segmentations):
                gray_frame = np.array(Image.open(path).convert("L"))
                hist, bins = exposure.histogram(gray_frame)
                otsu_threshold = filters.threshold_otsu(gray_frame)  # Compute optimal threshold
                segmented_image = gray_frame > otsu_threshold  # Apply the threshold
                # segmented_images.append(segmented_image)
                # save_path = "Your path here"
                # io.imsave(f"{save_path}/segmented_img_{i}.png", segmented_image)
                segmented_image_tensor = torch.as_tensor(np.array(segmented_image, dtype=np.uint8, copy=True)).unsqueeze(0).unsqueeze(-1).permute(0,3,1,2)
                segmented_image_tensor = segmented_image_tensor.repeat(1,3,1,1)
                segmented_image_tensors.append(segmented_image_tensor)
                # print(random_frame.shape)
        if(self.future_to_past):
            frame_metadata["frame_years"] = frame_metadata["frame_years"][::-1]
            frame_metadata["frame_months"] = frame_metadata["frame_months"][::-1]
            print("Reversing the yaers:", frame_metadata["frame_years"] )

        video_clip = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2)
        frame_metadata = {
            "frame_years": torch.stack(frame_metadata["frame_years"]).unsqueeze(-1),
            "frame_months": torch.stack(frame_metadata["frame_months"]).unsqueeze(-1),
        }
        return {'video': video_clip, 'video_name': 1, "frame_metadata": frame_metadata}

    def __len__(self):
        return self.video_num
    
    def metadata_normalize(self, metadata, base_lon=180, base_lat=90, base_year=1997, max_ndvi = 1,min_ndvi = -1, scale=1000):
        year, lat, lon, ndvi = metadata
        ndvi = 0 if ndvi == None else ndvi

        lon = lon / (180 + base_lon) * scale
        lat = lat / (90 + base_lat) * scale
        year = year / ((2100 - base_year) * scale)
        ndvi = (ndvi - min_ndvi)/ (max_ndvi - min_ndvi) * scale
        return (torch.tensor(lat), torch.tensor(lon), torch.tensor(year), torch.tensor(ndvi) )

    def normalize_year(self, year, base_year = 1984, scale = 1000):
        # print("year", year)
        year =  int(year)/(2100 - base_year) * scale
        print("Normalized year:", year)
        # print("year tensors shape",torch.tensor(year))
        # self.unnormalize_year(year=year)
        return torch.tensor(year)
    
    def normalize_month(self, month, scale = 1000):
            month = int(month) / 12 * scale
            return torch.tensor(month)

    def unnormalize_year(self, year, base_year=1984, scale=1000,
                            is_print=False):

        year = year / scale * (2100 - base_year) + (base_year if is_print else 0)    
    def load_video_frames(self, dataroot):
        data_all = []
        frame_list = os.walk(dataroot)
        for _, meta in enumerate(frame_list):
            root = meta[0]
            try:
                # frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[-1])) # Use this for the hq dataset
                frames = sorted(meta[2], key=lambda item: int(item.split('.')[0].split('_')[0]) if item.split(".")[1] == "png" else 0) # Use this for the sentinel 2 dataset 
                metadata_files = list(filter(lambda x: ".json" in x, frames))
                metadata_file_name = metadata_files[0] if len(metadata_files) else ''
                metadata_file_path = os.path.join(root, metadata_file_name) if metadata_file_name else ''
                # frames.pop(0) if len
                if(len(frames) and frames[0].split(".")[1] == "json"):
                    metadata = frames.pop(0)
            except Exception as e:
                print("Exception",e )
                # print(meta[0]) # root
                # print(meta[2]) # files
            frames = [os.path.join(root, item) for item in frames if is_image_file(item)]
            if len(frames) > max(0, self.target_video_len * self.frame_interval): # need all > (16 * frame-interval) videos
            # if len(frames) >= max(0, self.target_video_len): # need all > 16 frames videos
                frames_and_metadata = {
                    "frames": frames,
                    "metadata": metadata_file_path
                }
                data_all.append(frames_and_metadata)
        self.video_num = len(data_all)
        return data_all
    

if __name__ == '__main__':

    import argparse
    import torchvision
    # import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--data-path", type=str, default="/Users/prathapnkashyap/Development/RPTU_CS/Fourth_semester_summer_2024/Thesis_work/Time_Invariant_Image_Generation/Data/processed_data/Landsat_8/deforestation/downsampled_256x256/red")
    config = parser.parse_args()

    deforestation_dataset = Satellite_deforestation(config)
    print(len(deforestation_dataset))
    taichi_dataloader = data.DataLoader(dataset=deforestation_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i, video_data in enumerate(taichi_dataloader):
        print(video_data['video'].shape)