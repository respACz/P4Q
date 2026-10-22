export HF_HOME=/home/disk/workspace/hf_download/
export HF_DATASETS_CACHE=/home/disk/workspace/hf_download/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LMUData="/home/disk/workspace/LMUData" #Replace with the LMUData path in your project




#Calibration and evaluation of LLaVA-1.5-7B, run -calibration first, after it completes, run --eval
CUDA_VISIBLE_DEVICES=0,1 python optimize_flatquant.py --config configs/llava-1.5-7b-p4q-w4a4-calibration.yaml

CUDA_VISIBLE_DEVICES=0,1 python optimize_flatquant.py --config configs/llava-1.5-7b-p4q-w4a4-eval.yaml


#Calibration and evaluation of LLaVA-Next-7B, run -calibration first, after it completes, run --eval
CUDA_VISIBLE_DEVICES=0,1 python optimize_flatquant.py --config configs/llava-next-7b-p4q-w4a4-calibration.yaml

CUDA_VISIBLE_DEVICES=0,1 python optimize_flatquant.py --config configs/llava-next-7b-p4q-w4a4-eval.yaml