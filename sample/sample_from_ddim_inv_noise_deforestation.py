# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys
from pathlib import Path
try:
    import utils

    from diffusion import create_diffusion
    from utils import find_model, get_sample_meta_data_template_dict, create_overlayed_difference_map, calculate_hist_segmentation
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from utils import find_model, get_sample_meta_data_template_dict, create_overlayed_difference_map, calculate_hist_segmentation
# from Latte.utils import calculate_hist_segmentation
import torch
import argparse
import torchvision
from torchvision import transforms
from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import CLIPImageEmbedder, TextEmbedder
import imageio
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from einops import repeat
import pandas as pd
from skimage import feature, exposure, filters
from skimage.metrics import structural_similarity as ssim
import json
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(args):
    # 
    image = Image.open(args.image_path).convert("RGB", dtype=np.uint8, copy=True)
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {args.image_path}")
    h = w = args.image_size
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def load_image_from_list(args, image_path):
    # 
    image = Image.open(image_path).convert("RGB")
    x, y = image.size
    print(f"loaded input image of size ({x}, {y}) from {image_path}")
    h = w = args.image_size
    image = transforms.CenterCrop(min(x,y))(image)
    image = image.resize((w, h), resample=Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.
    # return image

def convert_tensors_to_video(samples, vae, is_ddim_inv_noise = False, file_suffix = '', args = None):
    b, f, c, h, w = samples.shape
    samples = rearrange(samples, 'b f c h w -> (b f) c h w')
    samples = vae.decode(samples / 0.18215).sample
    # samples = vae.decode(samples).sample

    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
    # Save and display images:

    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)

    video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    num_videos = len(list(Path(args.save_video_path,).glob("**/*.mp4")))

    video_name = f'ddim_inversion_noise_{num_videos}_{file_suffix}.mp4' if is_ddim_inv_noise else f"sample_{num_videos}_{file_suffix}__month_{args.start_month}.mp4"
    
    video_save_path = os.path.join(args.save_video_path, video_name)
    print(video_save_path)
    imageio.mimwrite(video_save_path, video_, fps=1, quality=9)
    print('save path {}'.format(args.save_video_path))
    return video_save_path

def save_each_frame(latents, ip_images_path_list, vae, args, ip_images_folder_id, orig_years, start):
    
    save_folder = f"{args.frame_wise_save_path}/{ip_images_folder_id}"
    num_folders = len(list(Path(save_folder).glob("**/")))
    corresponding_ip_images = ip_images_path_list[:len(latents)]
    file_names = [filepath.name for filepath in corresponding_ip_images]
    file_names = [filename.replace(filename[:4], str(orig_years[i])) for i, filename in enumerate(file_names)]
    save_folder = f"{save_folder}/{num_folders}-{str(orig_years[0])}"
    os.makedirs(save_folder, exist_ok=True)
    for i in range(args.num_ip_images):
        # sampled_frames = [latents[i][:,k,:,:,:] for k in range(args.num_frames)]
        op_frame = vae.decode(latents[i]/ 0.18215).sample
        save_path = Path(save_folder).joinpath(file_names[i])
        print(save_path)
        save_image(clamp_img(op_frame), save_path, nrow=1, normlize = True, value_range=(-1, 1))


def save_image_decoded(latents,ip_images, vae, args):
    # 
    ip_op_tensors = []
    if(args.future_to_past):
        ip_images = ip_images[::-1]
    for i in range(len(ip_images)):
        ip_op_tensors.append(ip_images[i])
        gen_img = vae.decode(latents[i]/ 0.18215).sample
        ip_op_tensors.append(gen_img)
        orig_img = ip_images[i][0].permute(1,2,0).cpu().numpy()
        gen_img_np = gen_img[0].permute(1,2,0).cpu().numpy()
        # gray scale
        # gray_ip = np.mean(orig_img[0], axis=0)
        # gray_op = np.mean(gen_img_np[0], axis=0)
        
        
        ssim_diff_overlayed_img = create_overlayed_difference_map(orig_img, gen_img_np)
        ssim_diff_overlayed_img = torch.from_numpy(ssim_diff_overlayed_img).permute(2,0,1).unsqueeze(0).to(device)
        ip_op_tensors.append(ssim_diff_overlayed_img)
    ip_op = torch.cat(ip_op_tensors)

    # ip_op = torch.cat([ip_images, decoded_latents], dim = 0)
    num_images = len(list(Path(args.decoded_latents_save_path).glob("**/*.*")))
    print("num images", num_images)
    os.makedirs(args.decoded_latents_save_path, exist_ok= True )
    save_file_name = "ip_1997_op_" + str(args.start_year) + "_" + args.file_suffix + "_"+ str(num_images)+ '_.png'
    diff_file_name = "ip_1997_op_" + str(args.start_year) + "_" + args.file_suffix + "_"+ str(num_images)+ 'diff_.png'

    save_path = Path.joinpath(Path(args.decoded_latents_save_path), save_file_name)
    diff_save_path = Path.joinpath(Path(args.decoded_latents_save_path), diff_file_name)

    ip_op = clamp_img(ip_op)
    print("IP Images:", save_path)
    # plt.imsave(diff_save_path, cmap='hot')  
    save_image(ip_op, save_path, nrow=3, normlize = True, value_range=(-1, 1))
    return save_path

def clamp_img(img):
    img = torch.clamp(img,min=-1,max=1)
    return 0.5*(img+1)

def load_model_checkpoint(model, ckpt):
    ckpt_path = ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict = False)
    return model

def perform_ddim_inversion(diffusion, model, init_image, y, using_cfg, args):
    sample_fn = model.forward
    model_kwargs = dict(y=y, use_fp16= False)
    
    inversion, progression = diffusion.ddim_sample_loop(
        sample_fn,
        init_image.shape, init_image,  # shape   # Noise (image latent)
        clip_denoised=False, 
        model_kwargs = model_kwargs, # these 3 params are unnamed parameters 
        progress=True, 
        device=device, 
        inversion = True,    )
    return inversion, progression

def normalize_year(year, base_year = 2000, scale = 1000, max_year = 2050):
    base_year = 1975
    upper_year = 2020
    
    mid_year = (base_year + upper_year) / 2    
    half_range = (max_year - base_year) / 2    
    normalized = (year - mid_year) / half_range * scale

    return torch.tensor(normalized).to(device)

    ## Alternate way of normalizing the year data
    # year =  (int(year)- base_year)/(max_year - base_year) * scale
    # return torch.tensor(year).to(device)

def unnormalize_year(self, year, base_year=1984, scale=1000,
                        is_print=False):

        year = year / scale * (2100 - base_year) + (base_year if is_print else 0)
# Assuming that we want the next num_frames predictions and also we want 1 year
# difference between each image.
def convert_years_to_tensors(start_year, num_frame):
    
    years = [start_year  +  i for i in range(num_frame)]
    print("Years:", years)
    normalized_year_tensors = [normalize_year(year) for year in years]
    normalized_year_tensors = torch.stack(normalized_year_tensors).unsqueeze(-1)
    # frame_years = {"frame_years": normalized_year_tensors}
    return normalized_year_tensors
    # return frame_years

def convert_orig_years_to_tensors(years):
    # years = [start_year - i for i in range(num_frame)]
    normalized_year_tensors = [normalize_year(year) for year in years]
    normalized_year_tensors = torch.stack(normalized_year_tensors).unsqueeze(-1)
    # frame_years = {"frame_years": normalized_year_tensors}
    return normalized_year_tensors

def encode_months_to_tensors(start_month, num_frames):
    months = [torch.tensor((start_month)%12 + 0).to(device, dtype = torch.float32) for i in range(1,num_frames+1)]
    months = torch.stack(months).unsqueeze(-1)
    # encoded_months = [cyclic_encode_month(month=month) for month in months]
    return (months)
def cyclic_encode_month(month):
    encoded_months = {
        "month_sin": np.sin(2*np.pi*month/12), # iF you pass the normalized month, divide by 12 * scale (scale: 1000)
        "month_cos": np.cos(2*np.pi*month/12)
    }
    return torch.tensor([encoded_months["month_sin"], encoded_months["month_cos"]]).to(device, dtype = torch.float32)

def add_random_noise_to_ddim_noise(ddim_noise, alpha, num_frames_to_corrupt, start_frame):
    for i in range(start_frame, num_frames_to_corrupt):
        random_tensors = torch.rand_like(ddim_noise[:,i,:,:,:])
        ddim_noise[:,i,:,:,:] = ddim_noise[:,i,:,:,:] * alpha + (1- alpha) * random_tensors
    return ddim_noise

def get_noisy_frames(b,f,c,h,w):
    random_tensors = torch.randn(b, f, c, h, w, dtype=torch.float16, device=device) # b f c h w
    return random_tensors

def add_args_meta_data_in_sample_metadata(sample_metadata_dict, args):
    # for key, value 
    sample_metadata_dict.update({
        "num_frames": args.num_frames,
        "ckpt_path": args.ckpt,
        "ip_image_path": args.ip_image_folder,
        "num_ip_images": args.num_ip_images,
        "start_month": args.start_month,
        "start_year": args.start_year,
        "cfg_value": args.cfg_scale,
    })
    return sample_metadata_dict

def extract_metadata(metadata_dict, args):
    frame_metadata = {
            "frame_years":[],
            # "latitude": [],
            # "longitude": [],
            # "ndvi": [],
            # "images": []
        }
    for frame_no in metadata_dict["year_wise_metadata"]:
        numeric_metadata = (frame_no["year"], metadata_dict["x_init"], metadata_dict["y_init"], frame_no["ndvi"])
        year_norm = metadata_normalize(numeric_metadata)
        frame_metadata["frame_years"].append(year_norm)

    frame_metadata = {
            "frame_years": torch.stack(frame_metadata["frame_years"][-1 * args.num_frames:]).unsqueeze(-1).unsqueeze(0).to(device, dtype = torch.float32),
            # "latitude": torch.stack(frame_metadata["latitude"][-1 * args.num_frames:]).unsqueeze(-1).unsqueeze(0).to(device, dtype = torch.float32),
            # "longitude": torch.stack(frame_metadata["longitude"][-1 * args.num_frames:]).unsqueeze(-1).unsqueeze(0).to(device, dtype = torch.float32),
            # "ndvi": torch.stack(frame_metadata["ndvi"][-1 * args.num_frames:]).unsqueeze(-1).unsqueeze(0).to(device, dtype = torch.float32),
            # "images": guidance_images,
            # # "images": edge_image_tensors
        }
    return frame_metadata

def metadata_normalize(metadata, base_lon=180, base_lat=90, base_year=1997, max_ndvi = 1,min_ndvi = -1, scale=1000):
    year, lat, lon, ndvi = metadata
    # ndvi = 0 if ndvi == None else ndvi

    # lon = lon / (180 + base_lon) * scale
    # lat = lat / (90 + base_lat) * scale
    year = year / (2100 - base_year) * scale
    # ndvi = (ndvi - min_ndvi)/ (max_ndvi - min_ndvi) * scale
    return (torch.tensor(year))
            #  torch.tensor(lat), torch.tensor(lon), torch.tensor(ndvi) )

def get_edge_image(image_path):
    gray = np.array(Image.open(image_path).convert("L"))
    edge_image = feature.canny(gray, sigma=0.0, low_threshold=0.1, high_threshold=0.25)
    edge_tensor = torch.as_tensor(np.array(edge_image, dtype=np.uint8, copy=True)).unsqueeze(0).unsqueeze(-1).permute(0,3,1,2)
    edge_tensor = edge_tensor.repeat(1,3,1,1)
    return (edge_tensor)

def get_histogram_segmentaed_images(image_path):
    gray_frame = np.array(Image.open(image_path).convert("L"))

    hist, bins = exposure.histogram(gray_frame)
    otsu_threshold = filters.threshold_otsu(gray_frame)  # Compute optimal threshold
    segmented_image = gray_frame > otsu_threshold  # Apply the threshold
    # segmented_images.append(segmented_image)
    # save_path = "your image path"
    # io.imsave(f"{save_path}/segmented_img_{i}.png", segmented_image)
    segmented_image_tensor = torch.as_tensor(np.array(segmented_image, dtype=np.uint8, copy=True)).unsqueeze(0).unsqueeze(-1).permute(0,3,1,2)
    segmented_image_tensor = segmented_image_tensor.repeat(1,3,1,1)
    # segmented_image_tensors.append(segmented_image_tensor)
    return segmented_image_tensor

def get_year_range(ip_images_path_list):
    init_year = ip_images_path_list[0].as_posix().split("/")[-1].split("_")[0]
    final_year = ip_images_path_list[-1].as_posix().split("/")[-1].split("_")[0]
    range = int(final_year) - int (init_year)
    return range
def calculate_structural_diffs(ip_images_path_list, args):
    structure_diff_list = [torch.tensor(0).to(device)]
    structure_diff_value = []
    num_comparisons = args.num_frames - 1
    structure_control = 20
    for i in range(num_comparisons):
        gray1 = np.array(Image.open(ip_images_path_list[i]).convert("L"))
        gray2 = np.array(Image.open(ip_images_path_list[i+1]).convert("L"))
        segmented_image_1 = calculate_hist_segmentation(gray1)
        segmented_image_2 = calculate_hist_segmentation(gray2)
        ssim_value, _ = ssim(segmented_image_1, segmented_image_2, full=True, data_range=gray1.max() - gray1.min())
        structural_diff = (1- ssim_value) * structure_control
        scale = 1000

        structure_diff_value.append(structural_diff * scale)
        diff_normalized = torch.tensor(structural_diff * scale).to(device)
        structure_diff_list.append(diff_normalized)
    
    year_range = get_year_range(ip_images_path_list)
    structural_diff_mean = sum(structure_diff_value)/year_range    
    structural_diff_mean_list = [torch.tensor(0).to(device)]+[torch.tensor(structural_diff_mean).to(device) for _ in range(0,5)]
    structural_diff_mean_list = torch.stack(structural_diff_mean_list).unsqueeze(-1)
    # structure_diff_list = torch.stack(structure_diff_list).unsqueeze(-1)
    # return structure_diff_list
    print("structural_diff_mean_list:", structural_diff_mean_list)
    return structural_diff_mean_list


def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    # device = "cpu"
    sample_metadata_dict = get_sample_meta_data_template_dict()
    add_args_meta_data_in_sample_metadata(sample_metadata_dict, args)

    print(sample_metadata_dict)
    if args.ckpt is None:
        assert args.model == "Latte-XL/2", "Only Latte-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)
    # 
    # a pre-trained model or load a custom Latte checkpoint from train.py:
    if(args.use_single_frame_model):         
        model = load_model_checkpoint(model=model, ckpt=args.single_frame_ckpt)
    else:
        model = load_model_checkpoint(model=model, ckpt=args.ckpt)
        
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    year_tensors = convert_years_to_tensors(start_year=args.start_year, num_frame=args.num_frames)
    month_tensors = encode_months_to_tensors(start_month = args.start_month, num_frames = args.num_frames)
# single image
    # init_image = load_image(args)
    # init_image = init_image.to(device)
    # 
    # init_latent = vae.encode(init_image).latent_dist.sample().mul_(0.18215)
    # z = init_latent
    # z = z.unsqueeze(1) # to introduce Another dimension (Frames) in the image tensor
    # z = z.repeat(1,args.num_frames,1,1,1) # Repeat the same frame 16 times to match the dimensions of the model
    # random_tensors = torch.randn(z.shape[0],args.num_frames - 1, z.shape[2], z.shape[3], z.shape[4], device = device) # introduce 15 random tensors apart from the unsqueezed dimension 
    # z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
    # z = torch.cat((z, random_tensors), dim = 1) 

# multi images
    # print("ip image folder", args.ip_image_folder)
    ip_images_path_list = list(Path(args.ip_image_folder).glob("**/*.png"))
    # 
    ip_images_path_list = sorted(ip_images_path_list, key=lambda item: int(item.name.split('.')[0].split('_')[0]) if item.name.split(".")[1] == "png" else 0)[-7:-1]
    num_ip_images_to_use = args.num_ip_images
    
    image_tensors = [load_image_from_list(args, ip_images_path_list[i]).to(device, dtype = torch.float32) for i in range(0, num_ip_images_to_use) if ".json" not in ip_images_path_list[i].as_posix()]
    if(args.use_structural_diff):
            structure_diff_list = calculate_structural_diffs(ip_images_path_list=ip_images_path_list[:args.num_frames], args = args)
    if(args.use_guidance_images):
        if(args.use_canny_edge_images):
            guidance_image_tensors = [get_edge_image(ip_images_path_list[i]).to(device, dtype = torch.float32) for i in range(0, num_ip_images_to_use) if ".json" not in ip_images_path_list[i].as_posix()]
        if(args.use_histogram_segmentations):
            guidance_image_tensors = [get_histogram_segmentaed_images(ip_images_path_list[i]).to(device, dtype = torch.float32) for i in range(0, num_ip_images_to_use) if ".json" not in ip_images_path_list[i].as_posix()]
    init_images_latents = [vae.encode(ip_image_tensor).latent_dist.sample().mul_(0.18215) for ip_image_tensor in image_tensors]
    if(args.future_to_past):
        init_images_latents = init_images_latents[::-1]
    z = torch.cat(init_images_latents).to(device)
    z = z.unsqueeze(0) # introduce batch dimension

    if(num_ip_images_to_use < args.num_frames):
        num_random_frames = args.num_frames - num_ip_images_to_use
        random_tensors = get_noisy_frames(z.shape[0], num_random_frames , z.shape[2], z.shape[3], z.shape[4])
        z = torch.cat((z, random_tensors), dim = 1)
    ## DDIM Inversion
    # unsqueeze here to get a batch size of 1
    if(args.extras == 3):
        
        frame_metadata = {"frame_years": year_tensors.unsqueeze(0), "frame_months": month_tensors.unsqueeze(0)}
    if(args.use_guidance_images):
        frame_metadata = {"frame_years": year_tensors.unsqueeze(0), "frame_months": month_tensors.unsqueeze(0), "images": torch.stack(guidance_image_tensors, dim = 1)}
        g = frame_metadata["images"]
        g = rearrange(g, 'b f c h w -> (b f) c h w').contiguous()
        g = vae.encode(g).latent_dist.sample().mul_(0.18215)
        g = rearrange(g, '(b f) c h w -> b f c h w', b=z.shape[0]).contiguous()
        frame_metadata["images"] = g
        frame_metadata["images"] = torch.cat((frame_metadata["images"], random_tensors), dim = 1)

    elif(args.extras == 4 and args.use_structural_diff):

                frame_metadata = {"frame_years": year_tensors.unsqueeze(0),
                                  "structure_diff": structure_diff_list.unsqueeze(0),
                                  "frame_months": month_tensors.unsqueeze(0),
                                }
    elif(args.extras == 5):
        if(args.use_guidance_images):
            
            if(args.use_CLIP_image_embeddings):
                frame_metadata = {"frame_years": year_tensors.unsqueeze(0), "frame_months": month_tensors.unsqueeze(0), "images": image_tensors[0]}
                
                image_embedder = CLIPImageEmbedder()
                guidance_image = frame_metadata["images"]
                image_embeddings = image_embedder(guidance_image)
                frame_metadata["image_embedding"] = image_embeddings.to(device, dtype = torch.float32)
                del frame_metadata["images"]

    
    ddim_inversion_noise, progression = perform_ddim_inversion(diffusion, model, z, frame_metadata, using_cfg, args=args)
    progression_index = -5
    # # Incase we want to use a different noise than the final ddim inverted noise
    noise_to_use = ddim_inversion_noise # progression[progression_index]["sample"] #ddim_inversion_noise #   progression[progression_index]["sample"]
    file_suffix = args.file_suffix
    noised_frames = noise_to_use[:,:num_ip_images_to_use, :,:,:]
    last_ddim_img_noise = noise_to_use[:,-2:-1, :,:,:]
    
    z = torch.cat((noised_frames, last_ddim_img_noise), dim = 1)
    # noised_frames = noise_to_use[:,:2, :,:,:]

    # 
    repeat_frames = 0
    if(repeat_frames):
        repeated_ddim_inv_noise = repeat(noised_frames, "n f c h w -> n (f k) c h w", k =repeat_frames)
        sample_metadata_dict.update({
            "repeated_after_inversion": True,
            "num_repeated_frames": repeat_frames
        })
    num_random_frames = args.num_frames - num_ip_images_to_use
    sample_metadata_dict.update({
        "num_noisy_frames": num_random_frames
    })
    random_tensors = get_noisy_frames(z.shape[0], num_random_frames , z.shape[2], z.shape[3], z.shape[4])
    z = torch.cat((noised_frames, random_tensors), dim = 1) 
    alpha = 1
    if(alpha < 1):
        num_frames_to_corrupt = 1
        start_frame = -2
        corrupted_frames = f"{start_frame} to {start_frame+ num_frames_to_corrupt-1}"
        sample_metadata_dict.update({
            "alpha_levelof_added_noise": alpha,
            "corrupted_frames": corrupted_frames
        })
        z = add_random_noise_to_ddim_noise(ddim_noise=z,
                                            alpha=alpha,
                                            num_frames_to_corrupt=num_frames_to_corrupt,
                                            start_frame=start_frame)
    if(using_cfg):
            
        # 
        z = repeat(z, "b f c h w -> (b k) f c h w", k = 2)
        frame_metadata["frame_months"] = repeat(frame_metadata["frame_months"], "b f m -> (b k) f m", k = 2)
        frame_metadata["frame_years"] = repeat(frame_metadata["frame_years"], "b f y -> (b k) f y", k = 2)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16) 
        model.to(dtype=torch.float16)
    if args.use_fp16:
        z = z.to(dtype = torch.float16, device = device)
    else:
        z = z.to(device = device)

    if using_cfg:
        
        if(args.extras == 3):
            model_kwargs = dict(y=frame_metadata, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
        sample_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=None, use_fp16=args.use_fp16)

        if(args.extras == 3 or args.extras == 4 or args.extras == 5):
            model_kwargs = dict(y=frame_metadata, use_fp16=args.use_fp16)
        # elif(args.extras == 4):
        #     model_kwargs = dict(y=frame_metadata, use_fp16=args.use_fp16)
        sample_fn = model.forward
    # Sample images:
    
    if args.sample_method == 'ddim':
        samples, ddim_sample_progression = diffusion.ddim_sample_loop(
        # model                 # shape   # Noise # these 3 params are unnamed parameters 
        sample_fn, z.shape, 
        z,
        clip_denoised=False, 
        model_kwargs=model_kwargs, 
        progress=True, 
        device=device,
        inversion = False
    )
        progression_frames = []
        for sample_progress in ddim_sample_progression:
            first_frame = sample_progress["sample"][:,0,:,:,:]
            progression_frames.append(vae.decode(first_frame/0.18215).sample)
        
        progression_frames = torch.cat(progression_frames)
        os.makedirs(args.sample_progression_save_path, exist_ok=True)
        num_images = len(list(Path(args.sample_progression_save_path).glob("**/*.*")))
        save_file_name = "sample_progression_" + str(num_images) + '_.png'

        save_path = Path.joinpath(Path(args.sample_progression_save_path), save_file_name)
        progression_frames_op = clamp_img(progression_frames)
        print("saving sample progressions at:", save_path)

        save_image(progression_frames_op, save_path, nrow=5, normlize = True, value_range=(-1, 1))



    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

    #     ####-------- Cyclic Inversion Code: This code is used to use the images generated in the previous DDIM inversion frame as input and process them to generate
    #  more number of predictions---------#######

    if(args.cyclic_sample_required):
        initial_sample_frames = samples
        num_frames_recycled = 4
        start_sample_frame = args.num_frames - num_frames_recycled
        last_frames = initial_sample_frames[:,start_sample_frame: ,:,:,:]
        random_tensors = get_noisy_frames(1, args.num_frames - num_frames_recycled, 4, latent_size, latent_size)
        z = torch.cat((last_frames, random_tensors), dim = 1) 
        # z = last_frames
        #ddim_inversion_noise, progression = perform_ddim_inversion(diffusion=diffusion, model=model, init_image=z, y= frame_metadata, using_cfg = using_cfg, args = args)
        noise_to_use = z #  ddim_inversion_noise # progression[18]["sample"]
        noised_frames = noise_to_use[:,:num_frames_recycled, :,:,:]
        random_tensors = torch.randn(1, args.num_frames - num_frames_recycled, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
        z = torch.cat((noised_frames, random_tensors), dim = 1) 
        sample_metadata_dict.update({
            "is_cyclic_ddim_performed": True,
            "num_frames_reused_for_cyclic_inv": num_frames_recycled,
        })
        if(using_cfg):
            
        # 
            z = repeat(z, "b f c h w -> (b k) f c h w", k = 2)
            frame_metadata["frame_months"] = repeat(frame_metadata["frame_months"], "b f m -> (b k) f m", k = 2)
            frame_metadata["frame_years"] = repeat(frame_metadata["frame_years"], "b f y -> (b k) f y", k = 2)

        re_samples, ddim_sample_progression = diffusion.ddim_sample_loop(
            # model                 # shape   # Noise # these 3 params are unnamed parameters 
            sample_fn, ddim_inversion_noise.shape, 
            # ddim_inversion_noise, 
            z,
            clip_denoised=False, 
            model_kwargs=model_kwargs, 
            progress=True, 
            device=device,
            inversion = False
        )
    
        samples = torch.cat((initial_sample_frames, re_samples), dim= 1) 
    print(samples.shape)
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)
    
    ip_images_folder_id = "Test_fName"
    future_years = [args.start_year + i for i in range(args.num_frames)]
    start = args.start_year
    sampled_frames = [samples[:,i,:,:,:] for i in range(args.num_frames)]
    save_each_frame(sampled_frames, ip_images_path_list, vae, args, ip_images_folder_id, future_years, start)

    save_image_decoded(sampled_frames, image_tensors, vae, args)
    video_save_path = convert_tensors_to_video(samples, vae,is_ddim_inv_noise = False, file_suffix=file_suffix, args = args)
    sample_metadata_dict.update({
        "final_sample_path": video_save_path
    })
    try:
        existing_metadata = pd.read_csv(args.metadata_save_path)
        metadata_index  = len(existing_metadata)
    except:
        print("Metadata file doesn't exist")
        metadata_index = 0
    metadata_dataframe = pd.DataFrame(sample_metadata_dict, index=[metadata_index])
    metadata_dataframe = pd.concat([existing_metadata, metadata_dataframe])
    metadata_dataframe.to_csv(args.metadata_save_path, index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/home/kashyap/Master_thesis/Latte/configs/satellite_images/sat_img_sample.yaml")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    print("args", args.config)
    main(omega_conf)
