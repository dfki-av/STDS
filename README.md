## STDS: Spatio-Temporal Diffusion model for Satellite Imagery

This is the implementation of [STDS](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13816/1381614/Spatiotemporal-diffusion-model-for-satellite-imagery/10.1117/12.3073127.full) 

This repository is a fork from a transformer-based video diffusion model called latte. 


## Setup

First, download and set up the repo:

```bash
git clone https://github.com/dfki-av/STDS
cd STDS
```

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate stds
```


## Sampling 

You can sample from our **pre-trained Latte models** with [`sample.py`](sample/sample_from_ddim_inv_noise_deforestation.py). 
To use the STDS model optimally, it is recommended to train the model with your own dataset.

This model has been trained by using dataset collected from [Google Earth Engine](https://developers.google.com/earth-engine/datasets/)
 Weights for our pre-trained Latte model can be found [here](https://huggingface.co/maxin-cn/Latte).  The script has various arguments to adjust sampling steps, change the classifier-free guidance scale, etc.


If you would like to measure the quantitative metrics of your generated results, please refer to [here](docs/datasets_evaluation.md).

## Training

We provide a training script for Latte in [`train.py`](train.py). The structure of the datasets can be found [here](docs/datasets_evaluation.md). This script can be used to train class-conditional and unconditional
Latte models. To launch Latte (256x256) training with `N` GPUs on the FaceForensics dataset 
:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --config ./configs/ffs/ffs_train.yaml
```

or If you have a cluster that uses slurm, you can also train Latte's model using the following scripts:

 ```bash
sbatch slurm_scripts/ffs.slurm
```

We also provide the video-image joint training scripts [`train_with_img.py`](train_with_img.py). Similar to [`train.py`](train.py) scripts, these scripts can be also used to train class-conditional and unconditional
Latte models.
If you are familiar with `PyTorch Lightning`, you can also use the training script [`train_pl.py`](train_pl.py) and [`train_with_img_pl.py`](train_with_img_pl.py) provided by [@zhang.haojie](https://github.com/zhang-haojie),

```bash
python train_pl.py --config ./configs/ffs/ffs_train.yaml
```

or

```bash
python train_with_img_pl.py --config ./configs/ffs/ffs_img_train.yaml
```

This script automatically detects available GPUs and uses distributed training.

## Contact Us
**Prathap Kashyap**: [prathapnkashyap@gmail.com]

## Citation
If you find this work useful for your research, please consider citing it.
```bibtex
@inproceedings{kashyap2025spatiotemporal,
  title={Spatiotemporal diffusion model for satellite imagery},
  author={Kashyap, Prathap Nagaraj and Javanmardi, Alireza and Jaiswal, Pragati and Reis, Gerd and Pagani, Alain and Stricker, Didier},
  booktitle={Eleventh International Conference on Remote Sensing and Geoinformation of the Environment (RSCy2025)},
  volume={13816},
  pages={376--384},
  year={2025},
  organization={SPIE}
}
```


## Acknowledgments
STDS has been greatly inspired by the following amazing works and teams: 
[Latte](https://github.com/Vchitect/Latte)
[DiT](https://github.com/facebookresearch/DiT) and [PixArt-α](https://github.com/PixArt-alpha/PixArt-alpha), we thank all the contributors for open-sourcing.


## License
The code and model weights are licensed under [LICENSE](LICENSE).
