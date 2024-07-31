from train import *
import torch
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from datasets import Datasets
import shutil
import utils
from models import SimpleNet, SimpleDensityNet
from utils import *
import math

def cross_validation(ds, batch_size, TrainClass, n_repetitions = 1, params = {}, replicate_train = False, m = 5, n_folds=10, error_count=2, verbose = True):

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

def make_plot_ml_vs_nll(dir = "plots"):
    # datasets = ["boston", "concrete", "energy", "wine", "student", "mpg", "california"]
    datasets = ["california"]
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
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        
        # ML-1
        print("ML-1")
        (nll_loss, mse_loss) = cross_validation(ds,batch_size, TrainSimpleDensity,)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        data_scores_nll.append(round(nll_loss, 8))

        #"Ensemble-5 (MSE)
        print("Ensemble-5 (MSE)")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainEnsemble, replicate_train = True, m = 5)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        data_scores_nll.append(round(nll_loss, 8))

        # ML-5
        print("ML-5")
        (nll_loss, mse_loss) = cross_validation(ds, batch_size, TrainEnsembleDensity, replicate_train = True, m = 5)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        data_scores_nll.append(round(nll_loss, 8))

        # Ensemble-10 (MSE)
        print("Ensemble-10 (MSE)")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainEnsemble, replicate_train = True, m = 10)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        data_scores_nll.append(round(nll_loss, 8))

        # ML-10
        print("ML-10")
        (nll_loss, mse_loss) = cross_validation(ds, batch_size, TrainEnsembleDensity, replicate_train = True, m = 10)
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        data_scores_nll.append(round(nll_loss, 8))

        # MC-dropout-5
        print("MC-dropout-5")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainMCdropout, params = {"m" : 5})
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        data_scores_nll.append(round(nll_loss, 8))

        # MC-dropout-10
        print("MC-dropout-10")
        (mse_loss, nll_loss ) = cross_validation(ds, batch_size, TrainMCdropout, params = {"m" : 10})
        data_scores_rmse.append(round(np.sqrt(mse_loss), 8))
        data_scores_nll.append(round(nll_loss, 8))

        data_rmse.append(data_scores_rmse)
        data_nll.append(data_scores_nll)
        print(data_scores_rmse)
        print(data_scores_nll)


    fig, ax = plt.subplots()
    ax.axis("off")
    table = ax.table(
        cellText=data_rmse, rowLabels=[d.capitalize() for d in datasets], colLabels=metrics_rmse, loc="center"
    )
    print(rowLabels=[d.capitalize() for d in datasets])
    print(data_rmse)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(metrics_rmse))))
    plt.title("RMSE 10-Fold Cross Validation")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "table_rmse_n"), bbox_inches="tight")
    
    
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
    plt.savefig(os.path.join(dir, "table_nll_n"), bbox_inches="tight")

        


def save_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    m = 10
    # utils.recreate_directory(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs"))
    # datasets = ["boston", "concrete", "energy", "wine", "student",  "california", "mpg"]
    datasets = ["student"]
    for dataset in datasets:
        utils.recreate_directory(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset))
        ds = Datasets(dataset, transform= lambda x : x.to(device), shuffle = False)
        train_sampler = torch.utils.data.SubsetRandomSampler(range(int(len(ds) * 0.9)))
        train_loader = [torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=train_sampler) for _ in range(m)]
        utils.recreate_directory(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble"))
        t = TrainEnsemble(train_loader, None, save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble"))
        t.train()
        utils.recreate_directory(os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble_density"))
        t = TrainEnsembleDensity(train_loader, None, save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble_density"))
        t.train()

def plot_normalized_bar(tensor_1, tensor_2, c_names, file_path, label_1, label_2):


    n_1 = (softmax(tensor_1, norm = False)).cpu().numpy().tolist()
    n_2 = (softmax(tensor_2, norm = False)).cpu().numpy().tolist()

    r1 = np.arange(len(c_names))
    bar_width = 0.35
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, n_1, color='slateblue', width=bar_width, edgecolor='grey', label=label_1)
    plt.bar(r2, n_2, color='indianred', width=bar_width, edgecolor='grey', label=label_2)
   
    plt.xlabel('Features', fontweight='bold', fontsize = 15)
    plt.xticks([r + bar_width/2 for r in range(len(c_names))], c_names, rotation='vertical', fontsize = 15)
    # plt.ylabel('Normalized CovLRP-diag', fontsize = 13)
    plt.ylabel('Normalized LRP', fontsize = 13)
    plt.grid()

    max_height = max(max(n_1), max(n_2))
    plt.ylim(0, max_height * 2)
    plt.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def plot_xai():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # datasets = ["boston", "concrete", "energy", "wine", "yacht", "ailerons", "kin8nm"]
    datasets = ["wine"]
    m = 10
    for dataset in datasets:
        ds = Datasets(dataset, transform= lambda x : x.to(device), shuffle = False)
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        input , _ = next(iter(test_loader))
        _, n_features = input.size()
        models_simple = []
        for i in range(m):
            model = SimpleNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble",f"model_{i}.pt")))
            models_simple.append(model)

        models_density = []
        for i in range(m):
            model = SimpleDensityNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble_density",f"model_{i}.pt")))
            models_density.append(model)
        

        for idx, (input, _) in enumerate(ds):
            if idx == 20:
                break
            # t_1 = covlrp_apply_simple_deep_ensembles(models_simple, input.squeeze())[0]
            # t_2 = covlrp_apply_density_deep_ensembles(models_density, input.squeeze())[0]
            # plot_normalized_bar(t_1, t_2,ds.c_names[:-1], f"plots/covlrp_{dataset}_{idx}","MSE-Ensembles","ML-Ensembles")
            t_1 = lrp_apply_simple_deep_ensembles(models_simple, input.squeeze())
            t_2 = lrp_apply_density_deep_ensembles(models_density, input.squeeze())
            plot_normalized_bar(t_1, t_2,ds.c_names[:-1], f"plots/lrp_{dataset}_{idx}","MSE-Ensembles","ML-Ensembles")
           

def kl_div_explanations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["boston", "concrete", "energy", "wine", "student", "mpg", "california"]
    # datasets = ["concrete"]
    m = 10
    for dataset in datasets:
        print(dataset)
        ds = Datasets(dataset, transform= lambda x : x.to(device), shuffle = False)
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        input , _ = next(iter(test_loader))
        _, n_features = input.size()
        models_simple = []
        for i in range(m):
            model = SimpleNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble",f"model_{i}.pt")))
            model.eval()
            models_simple.append(model)

        models_density = []
        for i in range(m):
            model = SimpleDensityNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble_density",f"model_{i}.pt")))
            model.eval()
            models_density.append(model)
        
      

    
        kls = []
        for idx, (input, _) in enumerate(ds):
            simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input)[0]
            simple_exp_normalized = softmax(simple_exp)
            
            density_exp = covlrp_apply_density_deep_ensembles(models_density, input)[0]
            density_exp_normalized = softmax(density_exp)
        

            kl = 0.5 * kl_divergence(simple_exp_normalized, density_exp_normalized) +  0.5 * kl_divergence(density_exp_normalized, simple_exp_normalized)
            kls.append(kl)

        print("CovLRP - diag")
        print(torch.tensor(kls).mean().item())


        kls = []
        for idx, (input, _) in enumerate(ds):
            simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input)[1]
            simple_exp_normalized = softmax(simple_exp)

            density_exp = covlrp_apply_density_deep_ensembles(models_density, input)[1]
            density_exp_normalized = softmax(density_exp)
            kl = 0.5 * kl_divergence(simple_exp_normalized, density_exp_normalized) +  0.5 * kl_divergence(density_exp_normalized, simple_exp_normalized)
            kls.append(kl)

        print("CovLRP - marg")
        print(torch.tensor(kls).mean().item())

        kls = []
        for idx, (input, _) in enumerate(ds):
            simple_exp = covgi_apply_simple_deep_ensembles(models_simple, input)[0]
            simple_exp_normalized = softmax(simple_exp)

            density_exp = covgi_apply_density_deep_ensembles(models_density, input)[0]
            density_exp_normalized = softmax(density_exp)
            kl = 0.5 * kl_divergence(simple_exp_normalized, density_exp_normalized) +  0.5 * kl_divergence(density_exp_normalized, simple_exp_normalized)
            kls.append(kl)

        print("CovGI - diag")
        print(torch.tensor(kls).mean().item())


        kls = []
        for idx, (input, _) in enumerate(ds):
            simple_exp = covgi_apply_simple_deep_ensembles(models_simple, input)[1]
            simple_exp_normalized = softmax(simple_exp)

            density_exp = covgi_apply_density_deep_ensembles(models_density, input)[1]
            density_exp_normalized = softmax(density_exp)
            kl = 0.5 * kl_divergence(simple_exp_normalized, density_exp_normalized) +  0.5 * kl_divergence(density_exp_normalized, simple_exp_normalized)
            kls.append(kl)

        print("CovGI - marg")
        print(torch.tensor(kls).mean().item())


        kls = []
        for idx, (input, _) in enumerate(ds):
            simple_exp = lrp_apply_simple_deep_ensembles(models_simple, input)
            simple_exp_normalized = softmax(simple_exp)

            density_exp = lrp_apply_density_deep_ensembles(models_density, input)
            density_exp_normalized = softmax(density_exp)
            kl = 0.5 * kl_divergence(simple_exp_normalized, density_exp_normalized) +  0.5 * kl_divergence(density_exp_normalized, simple_exp_normalized)
            kls.append(kl)

        print("LRP")
        print(torch.tensor(kls).mean().item())



        kls = []
        for idx, (input, _) in enumerate(ds):
            simple_exp = gi_apply_simple_deep_ensembles(models_simple, input)
            simple_exp_normalized = softmax(simple_exp)

            density_exp = gi_apply_density_deep_ensembles(models_density, input)
            density_exp_normalized = softmax(density_exp)
            kl = 0.5 * kl_divergence(simple_exp_normalized, density_exp_normalized) +  0.5 * kl_divergence(density_exp_normalized, simple_exp_normalized)
            kls.append(kl)

        print("GI")
        print(torch.tensor(kls).mean().item())





def feature_flipping():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["boston", "concrete", "energy", "wine", "student", "mpg", "california"]
    m = 10

    for dataset in datasets:
        print(dataset)
        ds = Datasets(dataset, transform= lambda x : x.to(device), shuffle = False)
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        input , _ = next(iter(test_loader))
        _, n_features = input.size()
        models_simple = []
        for i in range(m):
            model = SimpleNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble",f"model_{i}.pt")))
            model.eval()
            models_simple.append(model)

        models_density = []
        for i in range(m):
            model = SimpleDensityNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble_density",f"model_{i}.pt")))
            model.eval()
            models_density.append(model)
    

        print("CovLRP - diag")

        a_simples = [[], [], [], []]
        a_densitys = [[], [], [], []]
        for idx, (input, _) in enumerate(ds):
            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input_simple)[0]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:
            
                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
        
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covlrp_apply_density_deep_ensembles(models_density, input_density)[0]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:
                
                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())
            
            for limit_idx, limit in enumerate(["1", 0.25, 0.5, 1.0]):
                a_simple, a_density = aufc(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), limit) 
                a_simples[limit_idx].append(a_simple)
                a_densitys[limit_idx].append(a_density)
            

        print("simple")
        print(np.array(a_simples).mean(axis = -1))
        print("density")
        print(np.array(a_densitys).mean(axis = -1))   
        



        print("CovLRP - marg")

        a_simples = [[], [], [], []]
        a_densitys = [[], [], [], []]
        for idx, (input, _) in enumerate(ds):
            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input_simple)[1]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:

                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
        
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covlrp_apply_density_deep_ensembles(models_density, input_density)[1]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:
                
                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 0.25, 0.5, 1.0]):
                a_simple, a_density = aufc(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), limit) 
                a_simples[limit_idx].append(a_simple)
                a_densitys[limit_idx].append(a_density)
            

        print("simple")
        print(np.array(a_simples).mean(axis = -1))
        print("density")
        print(np.array(a_densitys).mean(axis = -1)) 
        




        print("CovGI - diag")

        a_simples = [[], [], [], []]
        a_densitys = [[], [], [], []]
        for idx, (input, _) in enumerate(ds):
            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covgi_apply_simple_deep_ensembles(models_simple, input_simple)[0]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:
            
                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
        
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covgi_apply_density_deep_ensembles(models_density, input_density)[0]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:
                
                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())
            
            for limit_idx, limit in enumerate(["1", 0.25, 0.5, 1.0]):
                a_simple, a_density = aufc(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), limit) 
                a_simples[limit_idx].append(a_simple)
                a_densitys[limit_idx].append(a_density)
            

        print("simple")
        print(np.array(a_simples).mean(axis = -1))
        print("density")
        print(np.array(a_densitys).mean(axis = -1))  
        



        print("CovGI - marg")

        a_simples = [[], [], [], []]
        a_densitys = [[], [], [], []]
        for idx, (input, _) in enumerate(ds):
            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covgi_apply_simple_deep_ensembles(models_simple, input_simple)[1]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:
                
                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
        
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covgi_apply_density_deep_ensembles(models_density, input_density)[1]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:
                
                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 0.25, 0.5, 1.0]):
                a_simple, a_density = aufc(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), limit) 
                a_simples[limit_idx].append(a_simple)
                a_densitys[limit_idx].append(a_density)
            

        print("simple")
        print(np.array(a_simples).mean(axis = -1))
        print("density")
        print(np.array(a_densitys).mean(axis = -1))  



        print("LRP")

        a_simples = [[], [], [], []]
        a_densitys = [[], [], [], []]
        for idx, (input, _) in enumerate(ds):
            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = lrp_apply_simple_deep_ensembles(models_simple, input_simple)
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:
                
                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
        
        
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = lrp_apply_density_deep_ensembles(models_density, input_density)
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:
                
                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 0.25, 0.5, 1.0]):
                a_simple, a_density = aufc(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), limit) 
                a_simples[limit_idx].append(a_simple)
                a_densitys[limit_idx].append(a_density)
            

        print("simple")
        print(np.array(a_simples).mean(axis = -1))
        print("density")
        print(np.array(a_densitys).mean(axis = -1))



        print("GI")

        a_simples = [[], [], [], []]
        a_densitys = [[], [], [], []]
        for idx, (input, _) in enumerate(ds):

            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = gi_apply_simple_deep_ensembles(models_simple, input_simple)
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:
                
                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
        
            
        
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = gi_apply_density_deep_ensembles(models_density, input_density)
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:

                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 0.25, 0.5, 1.0]):
                a_simple, a_density = aufc(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), limit) 
                a_simples[limit_idx].append(a_simple)
                a_densitys[limit_idx].append(a_density)
            

        print("simple")
        print(np.array(a_simples).mean(axis = -1))
        print("density")
        print(np.array(a_densitys).mean(axis = -1)) 


def plot_flipping_graph(y_values_1, y_values_2, file_name_1, file_name_2):
    x_values = range(len(y_values_1))  
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values_1, marker='o', linestyle='-', color='slateblue', label = "MSE-Ensembles")
    plt.fill_between(x_values, y_values_1, alpha=0.2, color='slateblue')
    plt.xticks(list(range(len(y_values_1))), fontsize=15)
    plt.xlabel('Number of Features Flipped', fontsize = 15)
    plt.ylabel('Uncertainty', fontsize = 15)
    max_height = max(y_values_1)
    plt.ylim(0, max_height * 2)
    plt.grid(True)
    plt.legend(fontsize=12, loc='upper right')
    plt.savefig(os.path.join("plots", file_name_1), bbox_inches="tight")

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values_2, marker='o', linestyle='-', color='indianred', label = "ML-Ensembles")
    plt.fill_between(x_values, y_values_2, alpha=0.2, color='indianred')
    plt.xticks(list(range(len(y_values_2))), fontsize=15)
    plt.xlabel('Number of Features Flipped', fontsize = 15)
    plt.ylabel('Uncertainty', fontsize = 15)
    max_height = max(y_values_2)
    plt.ylim(0, max_height * 2)
    plt.grid(True)
    plt.legend(fontsize=12, loc='upper right')
    plt.savefig(os.path.join("plots", file_name_2), bbox_inches="tight")
    plt.close()

def plot_aufc():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # datasets = ["boston", "concrete", "energy", "wine", "student", "mpg", "california"]
    datasets = ["wine"]
    m = 10
    for dataset in datasets:
        print(dataset)
        ds = Datasets(dataset, transform= lambda x : x.to(device), shuffle = False)
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        input , _ = next(iter(test_loader))
        _, n_features = input.size()
        # kde = get_best_kde(ds.inputs)
        models_simple = []
        for i in range(m):
            model = SimpleNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble",f"model_{i}.pt")))
            model.eval()
            models_simple.append(model)

        models_density = []
        for i in range(m):
            model = SimpleDensityNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble_density",f"model_{i}.pt")))
            model.eval()
            models_density.append(model)
        
        print("CovLRP - diag")

        for idx, (input, _) in enumerate(ds):
            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input_simple)[0]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:
                
                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
                
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input).item())
            density_exp = covlrp_apply_density_deep_ensembles(models_density, input)[0]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:
                
                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())


        
            plot_flipping_graph(np.array(uncertainty_simple, dtype = np.float32),np.array(uncertainty_density, dtype = np.float32),
                                 f"aufc_covlrp_simple_{dataset}_{idx}.png",  f"aufc_covlrp_density_{dataset}_{idx}.png")
            if idx == 100:
                    break


        print("LRP")

        for idx, (input, _) in enumerate(ds):
            
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = lrp_apply_simple_deep_ensembles(models_simple, input_simple)
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]

            for idx_sorted in idxs_sorted:
                
                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                
           
           
            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input).item())
            density_exp = lrp_apply_density_deep_ensembles(models_density, input)
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:
                
                input_density[idx_sorted] = 0
                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            plot_flipping_graph(np.array(uncertainty_simple, dtype = np.float32),np.array(uncertainty_density, dtype = np.float32),
                                 f"aufc_lrp_simple_{dataset}_{idx}.png", f"aufc_lrp_density_{dataset}_{idx}.png")


            if idx == 100:
                break



def dec_compare():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["boston", "concrete", "energy", "wine", "student", "mpg", "california"]
    m = 10
    for dataset in datasets:
        print(dataset)
        ds = Datasets(dataset, transform= lambda x : x.to(device), shuffle = False)
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        input , _ = next(iter(test_loader))
        _, n_features = input.size()
        models_simple = []
        for i in range(m):
            model = SimpleNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble",f"model_{i}.pt")))
            model.eval()
            models_simple.append(model)

        models_density = []
        for i in range(m):
            model = SimpleDensityNet(
                        n_features, 50
                    ).to(device)
            model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", dataset, "train_ensemble_density",f"model_{i}.pt")))
            model.eval()
            models_density.append(model)
        
        print("CovLRP - diag")

        n_true = [0, 0]
        n_false = [0, 0]
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        for idx, (input, _) in enumerate(test_loader):
        
            input = input.squeeze()
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input_simple)[0]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]
        
            for idx_sorted in idxs_sorted:

                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                

            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covlrp_apply_density_deep_ensembles(models_density, input_density)[0]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:

                input_density[idx_sorted] = 0

                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 1.0]):
                d_simple, d_density = dec_measure(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), simple_exp, density_exp, limit) 
                
                if d_density>d_simple:
                    n_true[limit_idx] += 1
                else:
                    n_false[limit_idx] += 1


        n_true = np.array(n_true)    
        n_false = np.array(n_false)
        print((n_true/(n_true + n_false)).tolist())


        print("CovLRP - marg")

        n_true = [0, 0]
        n_false = [0, 0]
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        for idx, (input, _) in enumerate(test_loader):
        
            input = input.squeeze()
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input_simple)[1]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]
        
            for idx_sorted in idxs_sorted:

                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                

            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covlrp_apply_density_deep_ensembles(models_density, input_density)[1]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:

                input_density[idx_sorted] = 0

                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 1.0]):
                d_simple, d_density = dec_measure(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), simple_exp, density_exp, limit) 
                
                if d_density>d_simple:
                    n_true[limit_idx] += 1
                else:
                    n_false[limit_idx] += 1


        n_true = np.array(n_true)    
        n_false = np.array(n_false)
        print((n_true/(n_true + n_false)).tolist())



        print("CovGI - diag")

        n_true = [0, 0]
        n_false = [0, 0]
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        for idx, (input, _) in enumerate(test_loader):
        
            input = input.squeeze()
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covgi_apply_simple_deep_ensembles(models_simple, input_simple)[0]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]
        
            for idx_sorted in idxs_sorted:

                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                

            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covgi_apply_density_deep_ensembles(models_density, input_density)[0]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:

                input_density[idx_sorted] = 0

                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 1.0]):
                d_simple, d_density = dec_measure(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), simple_exp, density_exp, limit) 
                
                if d_density>d_simple:
                    n_true[limit_idx] += 1
                else:
                    n_false[limit_idx] += 1


        n_true = np.array(n_true)    
        n_false = np.array(n_false)
        print((n_true/(n_true + n_false)).tolist())


        print("CovGI - marg")

        n_true = [0, 0]
        n_false = [0, 0]
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        for idx, (input, _) in enumerate(test_loader):
        
            input = input.squeeze()
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = covgi_apply_simple_deep_ensembles(models_simple, input_simple)[1]
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]
        
            for idx_sorted in idxs_sorted:

                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                

            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = covgi_apply_density_deep_ensembles(models_density, input_density)[1]
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:

                input_density[idx_sorted] = 0

                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 1.0]):
                d_simple, d_density = dec_measure(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), simple_exp, density_exp, limit) 
                
                if d_density>d_simple:
                    n_true[limit_idx] += 1
                else:
                    n_false[limit_idx] += 1


        n_true = np.array(n_true)    
        n_false = np.array(n_false)
        print((n_true/(n_true + n_false)).tolist())



        print("LRP")

        n_true = [0, 0]
        n_false = [0, 0]
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        for idx, (input, _) in enumerate(test_loader):
        
            input = input.squeeze()
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = lrp_apply_simple_deep_ensembles(models_simple, input_simple)
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]
        
            for idx_sorted in idxs_sorted:

                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                

            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = lrp_apply_density_deep_ensembles(models_density, input_density)
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:

                input_density[idx_sorted] = 0

                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 1.0]):
                d_simple, d_density = dec_measure(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), simple_exp, density_exp, limit) 
                
                if d_density>d_simple:
                    n_true[limit_idx] += 1
                else:
                    n_false[limit_idx] += 1


        n_true = np.array(n_true)    
        n_false = np.array(n_false)
        print((n_true/(n_true + n_false)).tolist())


        print("GI")

        n_true = [0, 0]
        n_false = [0, 0]
        test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
        test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
        for idx, (input, _) in enumerate(test_loader):
        
            input = input.squeeze()
            input_simple = input.clone()
            uncertainty_simple = []
            uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
            simple_exp = gi_apply_simple_deep_ensembles(models_simple, input_simple)
            idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]
        
            for idx_sorted in idxs_sorted:

                input_simple[idx_sorted] = 0
                uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
                

            input_density = input.clone()
            uncertainty_density = []
            uncertainty_density.append(models_density_infer(models_density, input_density).item())
            density_exp = gi_apply_density_deep_ensembles(models_density, input_density)
            idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
            for idx_sorted in idxs_sorted:

                input_density[idx_sorted] = 0

                uncertainty_density.append(models_density_infer(models_density, input_density).item())


            for limit_idx, limit in enumerate(["1", 1.0]):
                d_simple, d_density = dec_measure(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), simple_exp, density_exp, limit) 
                
                if d_density>d_simple:
                    n_true[limit_idx] += 1
                else:
                    n_false[limit_idx] += 1


        n_true = np.array(n_true)    
        n_false = np.array(n_false)
        print((n_true/(n_true + n_false)).tolist())

def train_utk():
    for i in range(10):
        t = TrainSimpleUTK(save_path = f"model_dirs_face/train_ensemble/model_{i}.pt")
        t.train()

    for i in range(10):
        t = TrainDensityUTK(save_path = f"model_dirs_face/train_ensemble_density/model_{i}.pt")          
        t.train()
                


make_plot_ml_vs_nll()