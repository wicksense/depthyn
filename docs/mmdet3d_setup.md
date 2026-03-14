# MMDetection3D Setup

Depthyn keeps the ML backend out of the main project environment on purpose.
MMDetection3D is treated as an optional detector host, not as the foundation of
the repo.

If you want to use MMDetection3D during Stage 1, the cleanest flow is:
- export replay frames with `depthyn prepare-ml-replay`
- run MMDetection3D externally on those frames
- import the normalized detections back through the `precomputed` detector path

That means you can bring up MMDetection3D once, point Depthyn at that Python,
and keep the replay/viewer tooling lightweight.

## Suggested Environment

These are the commands I expect you to run manually on this VM when we are
ready for the real ML backend:

```bash
/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/bin/mamba create -y -n depthyn-mmdet3d python=3.10 pip
source /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/etc/profile.d/conda.sh
conda activate depthyn-mmdet3d
/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/bin/mamba install -y -n depthyn-mmdet3d -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=12.1
python -m pip install -U openmim
mim install "mmengine>=0.10.0"
mim install "mmcv>=2.1.0"
mim install "mmdet>=3.3.0"
python -m pip install mmdet3d
git clone https://github.com/open-mmlab/mmdetection3d.git /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmdet3d
```

Why clone the repo even if `mmdet3d` is installed:
- it gives us direct access to official config files
- project-based models like `DSVT` often live under `projects/`
- Depthyn can add the repo to `PYTHONPATH` during inference with `--mmdet3d-repo`

## Depthyn Usage

Batch replay inference from an exported manifest:

```bash
PYTHONPATH=src python3 -m depthyn.cli run-mmdet3d-replay \
  --manifest-json artifacts/ml-replay/manifest.json \
  --output-json artifacts/centerpoint-predictions.json \
  --model-name centerpoint \
  --mmdet3d-python /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/bin/python \
  --mmdet3d-repo /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmdet3d \
  --ml-config /path/to/centerpoint.py \
  --ml-checkpoint /path/to/centerpoint.pth
```

Single detector replay:

```bash
PYTHONPATH=src python3 -m depthyn.cli replay \
  SampleData/output-26/converted_csv \
  --output artifacts/centerpoint-summary.json \
  --detector centerpoint \
  --mmdet3d-python /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/bin/python \
  --mmdet3d-repo /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmdet3d \
  --ml-config /path/to/centerpoint.py \
  --ml-checkpoint /path/to/centerpoint.pth
```

Baseline vs modern model comparison:

```bash
PYTHONPATH=src python3 -m depthyn.cli compare \
  SampleData/output-26/converted_csv \
  --output-dir artifacts/detector-comparison \
  --detectors baseline centerpoint dsvt \
  --mmdet3d-python /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/bin/python \
  --mmdet3d-repo /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmdet3d \
  --centerpoint-config /path/to/centerpoint.py \
  --centerpoint-checkpoint /path/to/centerpoint.pth \
  --dsvt-config /path/to/dsvt.py \
  --dsvt-checkpoint /path/to/dsvt.pth
```

One-command Stage 1b workflow:

```bash
PYTHONPATH=src python3 -m depthyn.cli compare-mmdet3d-replay \
  SampleData/output-26/converted_csv \
  --output-dir artifacts/centerpoint-stage1 \
  --model-name centerpoint \
  --mmdet3d-python /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/bin/python \
  --mmdet3d-repo /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmdet3d \
  --ml-config /path/to/centerpoint.py \
  --ml-checkpoint /path/to/centerpoint.pth
```

## Important Note About Depthyn Replay Data

The current replay path writes temporary point clouds as `XYZI` float32 with a
synthetic intensity of `0.0`.

That means stock MMDetection3D configs may still need a small config edit if
they expect:
- dataset-specific classes
- a different point feature count
- nuScenes/KITTI preprocessing assumptions

The adapter is now in place, but production-quality results will still require
Depthyn-specific config tuning and likely fine-tuning on Ouster data.
