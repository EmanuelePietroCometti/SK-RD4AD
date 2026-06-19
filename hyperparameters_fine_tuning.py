import optuna
import optuna_dashboard
from main import train, setup_seed
import os
import torch


def objective(trial):
    # Define the hyperparameters to optimize
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
    res = trial.suggest_int('res', 1, 3)
    layer_loss = trial.suggest_categorical('layerloss', [0, 1])
    L2 = trial.suggest_categorical('L2', [0,1,2])
    net = trial.suggest_categorical('net', ['res18', 'res34', 'res50', 'wide_res50'])
    cut = trial.suggest_int('cut', 0, 1)

    if layer_loss == 1:
        rate = trial.suggest_categorical('layerloss', [0,1])
    else:
        rate = 0.0

    # Setup fixed parameters for the tuning process
    seed=111
    class_ = 'reda'
    epochs = 20
    print_epoch = 5
    seg = 0
    data_path = "./mvtec/"
    save_path = "./tuning_ckpoints/"
    img_path = "./tuning_imgs/"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    setup_seed(seed)

    # Execute the optimization
    try:
        auroc_sp = train(class_=class_, epochs=epochs, learning_rate=learning_rate, res=res, batch_size=batch_size, print_epoch=print_epoch, seg=seg, data_path=data_path, save_path=save_path, print_canshu=0, score_num=1, print_loss=0, img_path=img_path, vis=0, cut=cut, layerloss=layer_loss, rate=rate, print_max=0, net=net, L2=L2, seed=seed)
    except RuntimeError as e:
        # Prune the trial if CUDA runs out of memory 
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
        else:
            raise e


    return auroc_sp


if __name__ == "__main__":
    # Define the SQLlite database name and path
    db_url = "sqlite:///sk_rd4ad_tuning.db"

    # Create the study using the persistent storage
    study = optuna.create_study(
        study_name="sk-rd4ad-hyper-tuning",
        storage=db_url,
        direction="maximize",
        load_if_exists=True
    )

    # Run the optimization
    study.optimize(objective, n_trials=50)