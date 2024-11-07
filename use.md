
# Train
## train from scratch
ns-train nerfacto --data data/nerfstudio/poster --load-dir {outputs/.../nerfstudio_models}

## link to remote 
ssh -L 7007:127.0.0.1:7007 username@training-host-ip

## multi-gpu train from scratch
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ns-train nerfacto --vis viewer --machine.num-devices 6 --pipeline.datamanager.train-num-rays-per-batch 4096 --data data/nerfstudio/bww_entrance

## Resume from existing checkpoint
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ns-train nerfacto --data data data/nerfstudio/bww_entrance --load-dir 

# CUDA dependent
## tinycudann



# Todo
- (Done) run the viewer on high quality

- (WIP) Check the cuda dependent


## Mixed precision: torch.cpu.amp + ipex
- torch.cuda.amp.grad_scaler import GradScaler nerfstudio/engine/trainer.py line 34
- from torch.cuda.amp import custom_bwd, custom_fwd nerfstudio/field_components/activations.py line 25

## torch.xpu.empty_cache()
- torch.cuda.empty_cache()

## torch.cuda.max_memory_allocated()
torch.cuda.max_memory_allocated() nerfstudio/field_components/mlp.py line282

## nerfacc.accumulate_along_rays
Accumulate volumetric values along the ray.




- (WIP) Check the blur 
- (WIP) Check the VR google available
- (WIP) Use the custom dataset try & IOS



## Nerfacto model

Camera pose refinement: optimize and refine the poses.

Piecewise Sampler: Piecewise sampler to produce the initial set of samples of the scene.

Proposal Sampler: consolidates the sample locations to the regions of the scene that contribute most to the final render (typically the first surface intersection)

Density Field: represent a coarse density representation of the scene to guide sampling. Combining a hash encoding decreasing the encoding dictionary size and number of feature levels with a small fused MLP (from tiny-cuda-nn) 


## Blender
allows compositing multiple NeRF objects into a NeRF scene. 




# Todo 3.18
- (Done) try a high res dataset like eyefultower to test if the data resolution limit the final rederning quality
- (Done) report PSNR and other metrics
 
    "psnr": 22.85936737060547,
    "psnr_std": 2.5529844760894775,
    "ssim": 0.8166646957397461,
    "ssim_std": 0.05813973769545555,
    "lpips": 0.3669483959674835,
    "lpips_std": 0.06379594653844833,
    "num_rays_per_sec": 1234773.25,
    "num_rays_per_sec_std": 129683.828125,
    "fps": 0.4407285451889038,
    "fps_std": 0.0462881438434124



# Eye
## Summary
This PR adds preliminary support for the EyefulTower dataset. Specifically, the following main functionality:

Ability to download part or all of the dataset via the ns-download-data script, with selectable dataset (of 11) and resolution (of 4).
Automatic generation of nerfstudio-compatible transforms.json upon download, allowing each dataset to be loaded into memory
Usage
You can download a single EyefulTower dataset and resolution with basic download command. This will grab the riverview scene at 2k JPEG resolution.

$ ns-download-data eyefultower
You can use the flags to download more of the dataset, or just download it all.

$ ns-download-data eyefultower --capture-name apartment # Download the apartment scene with the default 2k JPEG resolution
$ ns-download-data eyefultower --capture-name riverview table kitchen --resolution-name jpeg_2k exr_2k # Download multiple datasets and resolutions at the same time
$ ns-download-data eyefultower --resolution-name all --capture-name all # Download everything
Data is downloaded using the aws s3 sync command as suggested in the EyefulTower documentation, which means redundant data will not be redownloaded. This makes it easy to start small with the dataset (e.g. running only ns-download-data eyefultower) and eventually working your way through the commands until you've downloaded everything.

## Training
I've tested training on 3 scenes (riverview, office_view2, office1b), all using this command and parameters with the dataset swapped out:

ns-train nerfacto --data .\data\eyefultower\{dataset}\images-jpeg-2k\ \
    --pipeline.model.max-res 19912 \
    --pipeline.model.log2-hashmap-size 22 \
    --pipeline.model.far-plane 100 \
    --pipeline.datamanager.train-num-rays-per-batch 12800 \
    --max-num-iterations 100000 \
    --steps-per-eval-all-images 100000
All 3 datasets seemed to finish training and offer good results. See results section below.

Comments / questions / future work
Metashape: Originally, I tried using the ns-process-data script to convert from the cameras.xml files provided with Eyeful into something that nerfstudio understands. This doesn't work as-is though, since the Eyeful datasets use rig constraints, which the metashape processing in process_data.py doesn't understand. I made some progress towards implementing this and generated updated cameras.xml files for each of the downscaled datasets (which the original EyefulTower doesn't provide), but hit an issue when generating the transforms (since the exact tree isn't documented by metashape), and decided to just convert the existing cameras.json like I did. Adding support for rigs would be a great future addition.

EXR: I've generated the transforms.json file for the EXR images, but I haven't tried using them with any of the nerfstudio pipelines as I don't believe there's currently HDR training support. Assuming that's correct, I'm hoping that offering this dataset integration is a good first step towards adding HDR support in nerfstudio.

Fisheye: 5 of the 11 datasets from EyefulTower are taken with the V1 rig, which uses fisheye cameras. I've marked them as OPENCV_FISHEYE in the transforms.json, but I haven't tested those datasets as I'm not sure how good the fisheye support is. I see that some support was just merged in this PR a few hours ago. Perhaps fisheye support for EyefulTower can be a future PR?

Larger datasets: 6 of the 11 datasets from EyefulTower are taken with the V2 rig, and should in theory work fine with this code on a sufficiently powerful machine. On my workstation, though, I ran out of RAM when trying to load one of the larger scenes (apartment), so I stuck to the 3 smaller V2 scenes (riverview, office_view2, office1b). I believe the default dataloader tries to load the entire dataset into memory at once. Are there other dataloaders that can load the data in batches, rather than loading it all at once?

Automatic downscaling: The ns-process script automatically generates downsampled images. The method I took, directly generating the transforms.json, does not do that. This seems to work fine, at least for the smaller scenes as described above. I'm not sure exactly where the downsampled images are used, but I did notice that there's a line in the training output which says:

Auto image downscale factor of 1                                                 nerfstudio_dataparser.py:349
Does this indicate that there's automatic downscaling that could be applied at runtime, to perhaps scale down the larger datasets (e.g. apartment) into something smaller, e.g. 1k or 512, such that the entire (downscaled) dataset could fit into memory? This question assumes there's no batched dataloader we could use instead.

Splits: The EyefulTower dataset provides a splits.json file indicating how data should be partitioned into a train and test split. Currently, I don't use this file, as I'm not sure what format nerfstudio expects the splits in, or if there's support at all. If someone could point me in the right direction, I can try to add support. Right now, all the cameras, both train and test, are dumped into the transforms.json.

Subsampled data: In a similar vein to the splits file, as a way to try to reduce the amount of data that's loaded, I also generate a transforms_300.json and transforms_half.json, which (as you might've guessed) contain 300 images and half the dataset, respectively. I couldn't find a flag that would let me use these transforms_*.json files rather than the original transforms.json file in the generated datasets, but I'd love to know if one exists.

