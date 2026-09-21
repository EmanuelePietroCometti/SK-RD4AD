@echo off
setlocal

REM Array of classes
set "classes=carpet reda_baseline reda_dustOnValidation reda_dustOnValidationAndTrain"

REM Array of seeds
set "seeds=0 1 2 42 101"

for %%c in (%classes%) do (
    for %%s in (%seeds%) do (
        echo ==========================================================
        echo Starting run -^> Class: %%c ^| Seed: %%s ^| Project: skrd4ad_%%c
        echo ==========================================================

        python main.py ^
            --epochs 200 ^
            --res 3 ^
            --learning_rate 0.005 ^
            --batch_size 16 ^
            --seed %%s ^
            --class_ %%c ^
            --L2 2 ^
            --layerloss 1 ^
            --seg 1 ^
            --print_epoch 10 ^
            --data_path mvtec/ ^
            --ckpt_path checkpoints/ ^
            --project_name skrd4ad_%%c ^
            --net wide_res50 ^
            --vis 1 ^
            --aug-config configs\oat_hue.json ^
            --image-size 512 ^
            --image-isize 256 ^
            --rate 0.05 ^
            --img_path results/

        echo Finished run -^> Class: %%c ^| Seed: %%s
        echo.
    )
)

echo Tutti gli esperimenti sono stati completati!
pause