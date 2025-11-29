@echo off
REM Train on hip dataset
python train.py ^
    --data-root ".\data\hip" ^
    --save-dir ".\saved_models\hip_model" ^
    --img-size 64 128 128 ^
    --epochs 100 ^
    --batch-size 4 ^
    --lr 0.0001 ^
    --feature-size 48 ^
    --wandb ^
    --project "SwinX2S-Hip" ^
    --run-name "hip-baseline"