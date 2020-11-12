nohup python train.py --gpus 2 --accelerator "dp" --deterministic true --max_epochs 1000 --log_gpu_memory "min_max" --profiler "simple" --auto_lr_find true > output.log &

