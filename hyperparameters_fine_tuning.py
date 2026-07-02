import optuna
import argparse
from main import train, setup_seed
import os
import torch

# Add 'args' as a parameter to receive values from the command line
def objective(trial, args):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
    res = trial.suggest_int('res', 1, 3)
    layer_loss = trial.suggest_categorical('layerloss', [0, 1])
    L2 = trial.suggest_categorical('L2', [0, 1, 2])
    net = trial.suggest_categorical('net', ['res18', 'res34', 'res50', 'wide_res50'])
    cut = trial.suggest_int('cut', 0, 1)

    if layer_loss == 1:
        rate = trial.suggest_categorical('rate', [0.1, 0.5, 1.0]) 
    else:
        rate = 0.0

    # Use dynamic parameters passed from the command line (via args)
    seed = 42
    class_ = args.class_
    data_path = args.data_path
    save_path = args.save_path
    img_path = "./tuning_imgs/"
    
    epochs = 30
    print_epoch = 5
    seg = 1 # CRITICAL: must be 1 for Pixel F1 evaluation

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    setup_seed(seed)

    # Execute the optimization
    try:
        auroc_px, auroc_sp, aupro, ap_loc, optimal_f1_sp, optimal_prec_sp, optimal_rec_sp, optimal_f1_px = train(
            class_=class_, epochs=epochs, learning_rate=learning_rate, res=res, 
            batch_size=batch_size, print_epoch=print_epoch, seg=seg, 
            data_path=data_path, save_path=save_path, print_canshu=0, 
            score_num=1, print_loss=0, img_path=img_path, vis=0, cut=cut, 
            layerloss=layer_loss, rate=rate, print_max=0, net=net, L2=L2, seed=seed
        )
        
        # COMBINED METRIC CALCULATION (50/50 Weighting)
        alpha = 0.5
        combined_f1 = (alpha * optimal_f1_sp) + ((1 - alpha) * optimal_f1_px)
        
        print(f"\n[Trial {trial.number} Results] F1 Sample: {optimal_f1_sp:.3f} | F1 Pixel: {optimal_f1_px:.3f} | COMBINED: {combined_f1:.3f}\n")

    except RuntimeError as e:
        # Prune the trial if CUDA runs out of memory
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e

    return combined_f1


if __name__ == "__main__":
    # Configure argparse to read commands from the terminal
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SK-RD4AD")
    parser.add_argument('--class_', type=str, default='carpet', help='Dataset class name')
    parser.add_argument('--data_path', type=str, default='./mvtec/', help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='./fine_tuning_checkpoints/', help='Path to save checkpoints')
    args = parser.parse_args()

    db_url = "sqlite:///sk_rd4ad_tuning.db"

    study = optuna.create_study(
        study_name="sk-rd4ad-hyper-tuning",
        storage=db_url,
        direction="maximize",
        load_if_exists=True
    )

    # Use a lambda function (or wrapper) to pass 'args' to Optuna
    study.optimize(lambda trial: objective(trial, args), n_trials=30)