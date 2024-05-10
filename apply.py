from train import TrainSimple, TrainSimpleDensity, TrainEnsemble, TrainEnsembleDensity, TrainMCdropout
import torch
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from datasets import Datasets


def cross_validation(ds, batch_size, TrainClass, n_repetitions = 5, params = {}, replicate_train = False, m = 5, n_folds=10, error_count=2, verbose = True):

        params["verbose"] = False
        kf = KFold(n_splits=n_folds)
        errors = [0 for _ in range(error_count)]

        for rep in range(1, n_repetitions+1):
            if verbose:
                print("repetition ", rep)
            for fold, (train_index, test_index) in enumerate(kf.split(ds)):
                print("fold ", fold+1)
            
                train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
                test_sampler = torch.utils.data.SubsetRandomSampler(test_index)
                if replicate_train:
                    train_loader = [torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=train_sampler) for _ in range(m)]
                else:          
                    train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
                test_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=test_sampler)
                t = TrainClass(train_loader, test_loader, **params)
                t.train()
                errors_fold = t.test()
                for i, error_fold in enumerate(errors_fold):
                    errors[i] += error_fold
        avg_errors = [error / (n_folds * n_repetitions) for error in errors]
        if verbose:
            print(f"average errors: {avg_errors}")

        return avg_errors

def make_plot(dir = "plots"):
    datasets = ["boston", "concrete", "energy", "wine", "yacht"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    metrics_rmse = [
        "Ensemble-1 (MSE)",
        "ML-1",
        "Ensemble-5 (MSE)",
        "ML-5",
        "Ensemble-10 (MSE)",
        "ML-10",
        "MC-dropout-5",
        "MC-dropout-10",
    ]
    metrics_nll = [
        "ML-1",
        "Ensemble-5 (MSE)",
        "ML-5",
        "Ensemble-10 (MSE)",
        "ML-10",
        "MC-dropout-5",
        "MC-dropout-10",
    ]
    data_rmse = []
    data_nll = []

    for dataset in datasets:
        print("Dataset: ", dataset)
        ds = Datasets(dataset, transform= lambda x : x.to(device))
        data_scores_rmse = []
        data_scores_nll = []
        # Ensemble-1 (MSE)
        print("Ensemble-1 (MSE)")
        (mse_loss,) = cross_validation(ds,batch_size, TrainSimple, error_count=1)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        
        # ML-1
        print("ML-1")
        (nll_loss, mse_loss) = cross_validation(ds,batch_size, TrainSimpleDensity,)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        data_scores_nll.append(round(nll_loss, 3))

        #"Ensemble-5 (MSE)
        print("Ensemble-5 (MSE)")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainEnsemble, replicate_train = True, m = 5)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        data_scores_nll.append(round(nll_loss, 3))

        # ML-5
        print("ML-5")
        (nll_loss, mse_loss) = cross_validation(ds, batch_size, TrainEnsembleDensity, replicate_train = True, m = 5)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        data_scores_nll.append(round(nll_loss, 3))

        # Ensemble-10 (MSE)
        print("Ensemble-10 (MSE)")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainEnsemble, replicate_train = True, m = 10)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        data_scores_nll.append(round(nll_loss, 3))

        # ML-10
        print("ML-10")
        (nll_loss, mse_loss) = cross_validation(ds, batch_size, TrainEnsembleDensity, replicate_train = True, m = 10)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        data_scores_nll.append(round(nll_loss, 3))

        # MC-dropout-5
        print("MC-dropout-5")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainMCdropout, params = {"m" : 5})
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        data_scores_nll.append(round(nll_loss, 3))

        # MC-dropout-10
        print("MC-dropout-10")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainMCdropout, params = {"m" : 10})
        data_scores_rmse.append(round(np.sqrt(mse_loss), 3))
        data_scores_nll.append(round(nll_loss, 3))

        data_rmse.append(data_scores_rmse)
        data_nll.append(data_scores_nll)


    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(
        cellText=data_rmse, rowLabels=[d.capitalize() for d in datasets], colLabels=metrics_rmse, loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(metrics_rmse))))
    plt.title("RMSE 10-Fold Cross Validation")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "table_rmse"), bbox_inches="tight")
    
    fig, ax = plt.subplots()
    print(data_nll)
    print(metrics_nll)
    ax.axis("off")
    table = ax.table(
        cellText=data_nll, rowLabels=[d.capitalize() for d in datasets], colLabels=metrics_nll, loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(metrics_nll))))
    plt.title("NLL 10-Fold Cross Validation")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "table_nll"), bbox_inches="tight")

        


make_plot()
