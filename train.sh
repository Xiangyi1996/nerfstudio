# Train on xpu
ns-train nerfacto --data data/nerfstudio/desolation --machine.device-type xpu

# Train on cpu
ns-train nerfacto --data data/meetingroom --machine.device-type cpu

# Train on nv card
CUDA_VISIBLE_DEVICES=0 ns-train nerfacto --data data/meetingroom  \
--max-num-iterations 100000 \
--steps-per-eval-all-images 100000  \
--machine.num-devices 1 \
--vis viewer+wandb \
--pipeline.datamanager.train-num-rays-per-batch 4096

# Train on nv card with high resolution
CUDA_VISIBLE_DEVICES=3 ns-train nerfacto --data data/meetingroom  \
--pipeline.model.max-res 19912 \
--pipeline.model.log2-hashmap-size 22  \
--pipeline.model.far-plane 100  \
--pipeline.datamanager.train-num-rays-per-batch 12800 \
--max-num-iterations 100000  \
--steps-per-eval-all-images 100000 \
--machine.num-devices 1

# render the final results as a video
ns-render camera-path --load-config outputs/xxx/config.yml --camera-path-filename data/xxx/camera_paths/outer_inner_spiral.json  --output-path renders/xxx.mp4

# ns-render camera-path --load-config outputs/images-jpeg-2k/nerfacto/2024-03-18_153301/config.yml --camera-path-filename /mnt/sh_flex_storage/home/xiangyiz/project/nerfstudio/data/eyefultower/apartment/images-jpeg-2k/camera_paths/2024-03-25-14-28-14.json --output-path renders/images-jpeg-2k/2024-03-25-14-28-14.mp4