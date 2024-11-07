# CUDA_VISIBLE_DEVICES=1 ns-train nerfacto --data ../data/DJI_nerf  \
# --max-num-iterations 100000 \
# --steps-per-eval-all-images 100000  \
# --machine.num-devices 1 \
# --vis viewer+wandb \
# --pipeline.datamanager.train-num-rays-per-batch 4096 \
# --load-dir outputs/DJI_nerf/nerfacto/2024-05-12_113321/nerfstudio_models


# CUDA_VISIBLE_DEVICES=1 ns-train splatfacto \
# --data ../data/DJI_nerf_all/nerf_data  \
# --max-num-iterations 100000 \
# --steps-per-eval-all-images 100000  \
# --machine.num-devices 1 \
# --vis viewer+wandb \
# --pipeline.datamanager.train-num-rays-per-batch 4096 \

CUDA_VISIBLE_DEVICES=1 ns-train splatfacto --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False \
--data ../data/DJI_nerf_all/nerf_data  \
--max-num-iterations 100000 \
--vis viewer+wandb \

# CUDA_VISIBLE_DEVICES=1 ns-train splatfacto \--data ../data/DJI_nerf/  \\n--machine.num-devices 1 \\n--vis viewer+wandb \\n

# ns-export gaussian-splat --load-config outputs/DJI_nerf/splatfacto/2024-05-13_130220/config.yml --output-dir exports/splat/


# ns-train nerfacto --data data/meetingroom 
# --machine.device-type xpu

# CUDA_VISIBLE_DEVICES=3 ns-train nerfacto --data data/meetingroom  \
# --pipeline.model.max-res 19912 \
# --pipeline.model.log2-hashmap-size 22  \
# --pipeline.model.far-plane 100  \
# --pipeline.datamanager.train-num-rays-per-batch 12800 \
# --max-num-iterations 100000  \
# --steps-per-eval-all-images 100000 \
# --machine.num-devices 1

# ns-render camera-path --load-config outputs/images-jpeg-2k/nerfacto/2024-03-18_153301/config.yml --camera-path-filename data/nerfstudio/desolation/camera_paths/outer_inner_spiral.json  --output-path renders/desolation.mp4

# ns-render camera-path --load-config outputs/images-jpeg-2k/nerfacto/2024-03-18_153301/config.yml --camera-path-filename /mnt/sh_flex_storage/home/xiangyiz/project/nerfstudio/data/eyefultower/apartment/images-jpeg-2k/camera_paths/2024-03-25-14-28-14.json --output-path renders/images-jpeg-2k/2024-03-25-14-28-14.mp4