@echo off
REM Example training commands for SwinX2S (Windows)

REM Basic training with default settings
python train.py

REM Custom dataset path and image size
python train.py ^
    --data-root "C:\Users\jeevan\OneDrive\Desktop\Major Project\swinx2s-project\data\hip" ^
    --img-size 64 128 128

REM Train with specific hyperparameters
python train.py ^
    --epochs 50 ^
    --batch-size 4 ^
    --lr 0.0001 ^
    --feature-size 48

REM Full custom training example
python train.py ^
    --data-root ".\data\femur" ^
    --save-dir ".\experiments\exp_01" ^
    --img-size 64 128 128 ^
    --epochs 100 ^
    --batch-size 2 ^
    --lr 0.0001 ^
    --feature-size 48 ^
    --depths 2 2 2 2 ^
    --num-heads 2 4 6 8 ^
    --window-size 8 8 ^
    --patience 15 ^
    --wandb ^
    --project "SwinX2S-Femur" ^
    --run-name "exp01-small-model"