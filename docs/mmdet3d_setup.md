# MMDetection3D Setup

This is the setup path that was actually verified on this VM.

Depthyn keeps the ML backend out of the main project environment on purpose.
MMDetection3D is treated as an optional detector host, not as the foundation of
the repo.

## Proven Environment Bring-Up

Run from `/home/spriteadmin/Documents/LiDAR-Object-Detection`.

```bash
/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/bin/mamba create -y -n depthyn-mmdet3d python=3.10 pip

source /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/etc/profile.d/conda.sh
conda activate depthyn-mmdet3d

python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install "setuptools<81" mmengine==0.10.7 addict yapf rich termcolor pycocotools cmake ninja psutil

/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/bin/mamba install -y -n depthyn-mmdet3d -c nvidia cuda-toolkit=11.8 cuda-nvcc=11.8
/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/bin/mamba install -y -n depthyn-mmdet3d -c conda-forge gcc_linux-64=11.4.0 gxx_linux-64=11.4.0
```

Clone the helper repos if they do not already exist:

```bash
git clone https://github.com/open-mmlab/mmcv.git -b 2.x /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmcv
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmdet3d
```

Build and install `mmcv` from source:

```bash
source /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/etc/profile.d/conda.sh
conda activate depthyn-mmdet3d

cd /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmcv

export CUDA_HOME=/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/targets/x86_64-linux
export PATH="$CUDA_HOME/bin:/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/nvvm/bin:/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/lib:${LD_LIBRARY_PATH}"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=4
export PIP_USE_PEP517=0

python -m pip install -v . --no-build-isolation
```

Install `mmdet` and `mmdet3d`:

```bash
source /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/etc/profile.d/conda.sh
conda activate depthyn-mmdet3d

python -m pip install "mmdet>=3.0.0,<3.3.0"

cd /home/spriteadmin/Documents/LiDAR-Object-Detection/.mmdet3d
python -m pip install -v -e . --no-build-isolation
```

## Verify The Environment

```bash
source /home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/etc/profile.d/conda.sh
conda activate depthyn-mmdet3d

export CUDA_HOME=/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/targets/x86_64-linux
export PATH="$CUDA_HOME/bin:/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/nvvm/bin:/home/spriteadmin/Documents/LiDAR-Object-Detection/.miniforge3/envs/depthyn-mmdet3d/bin:$PATH"

python - <<'PY'
import torch, mmengine, mmcv, mmdet, mmdet3d, numpy
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("numpy:", numpy.__version__)
print("mmengine:", mmengine.__version__)
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
print("mmdet3d:", mmdet3d.__version__)
PY
```

Verified on this VM:
- `torch 2.0.1+cu118`
- `mmengine 0.10.7`
- `mmcv 2.0.0`
- `mmdet 3.2.0`
- `mmdet3d 1.4.0`

## Notes

- `download.openmmlab.com` timed out from this environment, so `mim install mmcv`
  was not usable here.
- `mmcv` had to be built from local source.
- `CUDA_HOME` must point at the toolkit root under `targets/x86_64-linux`, not the
  env root.
- `nvvm/bin` must be on `PATH` so `nvcc` can find `cicc`.
- CUDA 11.8 needed `gcc/g++ 11.4` in the env for the build to succeed.

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
