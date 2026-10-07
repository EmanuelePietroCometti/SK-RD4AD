@echo off
setlocal EnableDelayedExpansion

REM Augmentation: aug_off = paper, aug_legacy = pipeline pre-merge del fork
set "AUG=aug_off"

REM TRAIN=1 addestra e poi fa il test. TRAIN=0 salta il training e rifa solo test e calib.json sui checkpoint esistenti
set "TRAIN=1"

REM Array of classes
set "classes=carpet tessuto_nero tessuto_nero_dust_validation tessuto_nero_dust_train"

REM Array of seeds
set "seeds=0 1 2 42 101"

REM Il checkpoint migliore di una run e l'ultimo salvato: dir /o-d lo mette per primo

for %%c in (%classes%) do (
    for %%s in (%seeds%) do (
        set "FAILED=0"

        if "%TRAIN%"=="1" (
            echo ==========================================================
            echo Starting run -^> Aug: %AUG% ^| Class: %%c ^| Seed: %%s
            echo ==========================================================

            python main.py ^
                --epochs 200 ^
                --res 3 ^
                --learning_rate 0.005 ^
                --batch_size 32 ^
                --seed %%s ^
                --class_ %%c ^
                --L2 0 ^
                --layerloss 1 ^
                --seg 1 ^
                --print_epoch 20 ^
                --data_path "D:/emanuele/Code/dataset/" ^
                --ckpt_path checkpoints/ ^
                --aug-config configs\%AUG%.json ^
                --score-preset paper ^
                --project_name skrd4ad_%AUG%_%%c ^
                --net wide_res50 ^
                --vis 1 ^
                --image-size 256 ^
                --image-isize 256 ^
                --rate 0.05 ^
                --img_path results/

            if errorlevel 1 set "FAILED=1"
        )

        if "!FAILED!"=="1" (
            echo ERROR: training fallito -^> Class: %%c ^| Seed: %%s ^| test saltato
        ) else (
            set "CKPT="
            for /f "delims=" %%f in ('dir /b /a-d /o-d "checkpoints\skrd4ad_%AUG%_%%c_ep*_seed%%s_sample_auc=*.pth" 2^>nul') do if not defined CKPT set "CKPT=checkpoints/%%f"

            if not defined CKPT (
                echo ERROR: nessun checkpoint trovato -^> Aug: %AUG% ^| Class: %%c ^| Seed: %%s
            ) else (
                echo Test + calib.json -^> Aug: %AUG% ^| Class: %%c ^| Seed: %%s
                echo Checkpoint: !CKPT!
                python eval.py ^
                    --class_ %%c ^
                    --data_path "D:/emanuele/Code/dataset/" ^
                    --checkpoint_path "!CKPT!" ^
                    --img_path results_eval_%AUG%/%%c/seed_%%s/ ^
                    --seg 1 ^
                    --res 3 ^
                    --net wide_res50
            )
        )

        echo Finished run -^> Aug: %AUG% ^| Class: %%c ^| Seed: %%s
        echo.
    )
)

echo Tutti gli esperimenti sono stati completati.
pause