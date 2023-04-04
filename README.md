# DyLiN: Making Light Field Networks Dynamic
[![arXiv](https://img.shields.io/badge/arXiv-2303.14243-red.svg)](https://arxiv.org/abs/2303.14243)
[![Website](https://img.shields.io/badge/website-up-yellow.svg)](https://dylin2023.github.io/)


This repository is for the new neral light field (NeLF) method introduced in the following ECCV'22 paper:
> **[DyLiN: Making Light Field Networks Dynamic](https://dylin2023.github.io/)** \
> [Heng Yu](https://heng14.github.io/) <sup>1</sup>, [Joel Julin](https://joeljulin.github.io/) <sup>1</sup>, [Zoltán Á Milacski](https://scholar.google.com/citations?user=rSqodggAAAAJ&hl=es) <sup>1</sup>, [Koichiro Niinuma](https://scholar.google.com/citations?user=AFaeUrYAAAAJ&hl=en) <sup>2</sup>, and [László A. Jeni](https://www.laszlojeni.com/) <sup>1</sup> \
> <sup>1</sup> Carnegie Mellon University <sup>2</sup> Fujitsu Research of America 

### [Project](https://dylin2023.github.io/) | [ArXiv](https://arxiv.org/abs/2303.14243)


<!-- **[TL;DR]** We present R2L, a deep (88-layer) residual MLP network that can represent the neural *light* field (NeLF) of complex synthetic and real-world scenes. It is featured by compact representation size (~20MB storage size), faster rendering speed (~30x speedup than NeRF), significantly improved visual quality (1.4dB boost than NeRF), with no whistles and bells (no special data structure or parallelism required). -->

<!-- <div align="center">
    <a><img src="figs/frontpage.png"  width="700" ></a>
</div>
 -->

## Setup
The codebase is based on [R2L](https://github.com/snap-research/R2L).

### Environment
We use the same environment as it. We test tested it using Python 3.9.

### Data
For synthetic scenes, we use data from [D-NeRF](https://github.com/albertpumarola/D-NeRF). \
For real scenes, we use data from [HyperNeRF](https://github.com/google/hypernerf).

<!-- ### 3. Quick start: test our trained models
- Download models:
```
sh scripts/download_R2L_models.sh
```

- Run
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --cache_ignore data --trial.ON --trial.body_arch resmlp --pretrained_ckpt R2L_Blender_Models/lego.tar --render_only --render_test --testskip 1 --screen --project Test__R2L_W256D88__blender_lego
```   -->
 
## Train DyLiN models
There are several steps in DyLiN training which is similar as [R2L](https://github.com/snap-research/R2L).

(1) Train a teacher NeRF model. \
(2) Use *pretrained* teacher NeRF model to generate synthetic data. \
(3) Train DyLiN network on the synthetic data -- this step can make our DyLiN model perform *comparably* to the teacher NeRF model. \
(4) Finetune the DyLiN model using the *real* data -- this step will further boost the performance and make our DyLiN model *outperform* the teacher NeRF model.

The detailed step-by-step training pipeline is as follows.

#### Step 1. 
For synthetic scenes, please refer to [D-NeRF](https://github.com/albertpumarola/D-NeRF). \
For real scenes, please refer to [HyperNeRF](https://github.com/google/hypernerf).


<!-- Train a NeRF model:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name nerf --config configs/lego.txt --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --project NeRF__blender_lego
```

You can also download the teachers we trained to continue first:
```bash
sh scripts/download_NeRF_models.sh
```

To test the download teachers, you can use
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name nerf --config configs/lego.txt --pretrained_ckpt NeRF_Blender_Models/lego.tar --render_only --render_test --testskip 1 --screen --project Test__NeRF__blender_lego
``` -->


#### Step 2. 

Use the pretrained NeRF model to generate synthetic data (saved in `.npy` format). For details about the  synthetic data, please refer to [R2L](https://github.com/snap-research/R2L). \
For synthetic scenes:
```bash
python tools/gen_pseudo_data_dnerf.py
```
For real scenes:
```bash
python tools/gen_pseudo_data_hypernerf.py
```

<!-- ```bash
CUDA_VISIBLE_DEVICES=0 python utils/create_data.py --create_data rand --config configs/lego.txt --teacher_ckpt Experiments/NeRF__blender_lego*/weights/200000.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_pseudo_images10k --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --project NeRF__blender_lego__create_pseudo
``` -->

<!-- If you are using the downloaded teachers, please use this snippet:
```bash
CUDA_VISIBLE_DEVICES=0 python utils/create_data.py --create_data rand --config configs/lego.txt --teacher_ckpt NeRF_Blender_Models/lego.tar --n_pose_kd 10000 --datadir_kd data/nerf_synthetic/lego:data/nerf_synthetic/lego_pseudo_images10k --screen --cache_ignore data,__pycache__,torchsearchsorted,imgs --project NeRF__blender_lego__create_pseudo
``` -->

<!-- The pseudo data will be saved in `data/nerf_synthetic/lego_pseudo_images10k`. Every 4096 rays are saved in one .npy file. For 10k images (400x400 resoltuion), there will be 309600 .npy files. On our RTX 2080Ti GPU, rendering 1 image with NeRF takes around 8.5s, so 10k images would take around 24hrs. **If you want to try our method quicker, you may download the lego data we synthesized** (500 images, 2.8GB) and go to Step 3:
```bash
sh scripts/download_lego_pseudo_images500.sh
```
The data will be extracted under `data/nerf_synthetic/lego_pseudo_images500`. Using only 500 pseudo images for training would lead to degraded quality, but based on our ablation study (see Fig. 6 in our paper), it works farily good. -->


#### Step 3.
Train R2L model on the synthetic data. \
For synthetic scenes (xxx is the set name, e.g. lego, standup and jumpingjacks):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/xxx_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/xxx_pseudo_images10k_dnerf --n_pose_video 20,1,1 --N_iters 800000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --cache_ignore data,__pycache__,torchsearchsorted,imgs --screen --project R2L__blender_xxx
```
For real scenes :
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/xxx_pseudo_images10k_hypernerf --n_pose_video 20,1,1 --N_iters 800000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --cache_ignore data,__pycache__,torchsearchsorted,imgs --screen --project R2L__blender_xxx --hyperdata
```


<!-- If you are using the downloaded `lego_pseudo_images500` data, please use this snippet:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/lego_pseudo_images500 --n_pose_video 20,1,1 --N_iters 1200000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --cache_ignore data,__pycache__,torchsearchsorted,imgs --screen --project R2L__blender_lego
``` -->

#### Step 4. 
Convert original real data (images) to our `.npy` format. \
(1) For synthetic scenes (xxx is the set name, e.g. lego, standup and jumpingjacks):
```bash
python tools/convert_original_data_dnerf.py --splits train --datadir data/xxx
```
(2) For real scenes: 
```bash
python tools/convert_original_data_hypernerf.py
```

<!-- * For blender data:
```bash
python utils/convert_original_data_to_rays_blender.py --splits train --datadir data/nerf_synthetic/lego
```
The converted data will be saved in `data/nerf_synthetic/lego_real_train`.

* For llff data:
```bash
python utils/convert_original_data_to_rays_llff.py --splits train --datadir data/nerf_llff_data/flower
```
The converted data will be saved in `data/nerf_llff_data/room_real_train`. -->


Finetune the R2L model using real data. (for real scenes, add flag "--hyperdata")
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --datadir_kd data/nerf_synthetic/lego_real_train --n_pose_video 20,1,1 --N_iters 810000 --N_rand 20 --data_mode rays --hard_ratio 0.2 --hard_mul 20 --use_residual --cache_ignore data,__pycache__,torchsearchsorted,imgs --screen --trial.ON --trial.body_arch resmlp --num_worker 8 --warmup_lr 0.0001,200 --save_intermediate_models --pretrained_ckpt Experiments/R2L__blender_lego_SERVER*/weights/ckpt.tar --resume --project R2L__blender_lego__ft
```
<!-- Note, this step is pretty fast and prone to overfitting, so do not finetune it too much. We simply set the finetuning steps based on our validation. -->

## Test DyLiN models
(for real scenes, add flag "--hyperdata")
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name R2L --config configs/lego_noview.txt --n_sample_per_ray 16 --netwidth 256 --netdepth 88 --use_residual --cache_ignore data --trial.ON --trial.body_arch resmlp --pretrained_ckpt Experiments/R2L_XXX/weights/ckpt.tar --render_only_fix_pose --testskip 1 --screen --project Test_XXX
```

<!-- ## Results
The quantitative and qualitative comparison are shown below. See more results and videos on our [webpage](https://snap-research.github.io/R2L/).
<div align="center">
    <a><img src="figs/blender_psnr_comparison.png"  width="700" ></a><br>
    <a><img src="figs/blender_visual_comparison.png"  width="700"></a>
</div> -->


<!-- ## Acknowledgments
In this code we refer to the following implementations: [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [smilelogging](https://github.com/MingSun-Tse/smilelogging). Great thanks to them! We especially thank [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch). Our code is largely built upon their wonderful implementation. We also greatly thank the anounymous ECCV'22 reviewers for the constructive comments to help us improve the paper. -->

## Reference

If our work or code helps you, please consider citing our paper. Thank you!
```BibTeX
@article{yu2023dylin,
  title={DyLiN: Making Light Field Networks Dynamic},
  author={Yu, Heng and Julin, Joel and Milacski, Zoltan A and Niinuma, Koichiro and Jeni, Laszlo A},
  journal={arXiv preprint arXiv:2303.14243},
  year={2023}
}
```



