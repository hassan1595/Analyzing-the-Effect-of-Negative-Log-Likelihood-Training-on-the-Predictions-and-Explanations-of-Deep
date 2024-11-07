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
from tqdm import tqdm

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

def train_vgg16():
    for i in range(10):
        t = TrainSimpleVGG16(save_path = f"model_dirs_face/train_ensemble/model_{i}.pt")
        t.train()

    # for i in range(10):
    #     t = TrainDensityVGG16(save_path = f"model_dirs_face/train_ensemble_density/model_{i}.pt")          
    #     t.train()


def exp_vgg16(src_dir = "datasets_face\CelebAMask-HQ\CelebA-Eye-G", tgt_dir = "exp_faces", log_file_1 = "logs/logs_4.txt", log_file_2 = "logs/logs_5.txt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_simple = []
    for i in range(10):
        model = vgg16SimpleNet()
        model.load_state_dict(torch.load(f"model_dirs_face/train_ensemble/model_{i}.pt"))
        model.eval()
        model.to(device)
        models_simple.append(model)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    simple_exp_dict = {"lrp_exp" : {}, "gi_exp" : {}, "covlrp_diag_exp" : {}, "covlrp_marg_exp" : {}, "covgi_diag_exp" : {}, "covgi_marg_exp" : {}}
    for img_dir in tqdm(sorted(os.listdir(src_dir))):
        main_img_path = os.path.join(src_dir, img_dir, img_dir[:-5] + ".jpg")
        image = Image.open(main_img_path).convert('RGB').resize((224, 224))
        x = data_transforms(image).unsqueeze(0).to(device)
        lrp_rel = lrp_vgg16_simple_deep_ensembles(models_simple, x)
        lrp_acc = np.nan_to_num(np.array(lrp_rel.cpu(), dtype = np.float32)).sum()

        gi_rel = gi_vgg16_simple_deep_ensembles(models_simple, x)
        gi_acc = np.nan_to_num(np.array(gi_rel.cpu(), dtype = np.float32)).sum()
  
        covlrp_diag, covlrp_marg = covlrp_vgg16_simple_deep_ensembles(models_simple, x)
        covlrp_diag_acc = np.nan_to_num(np.array(covlrp_diag.cpu(), dtype = np.float32)).sum()
        covlrp_marg_acc = np.nan_to_num(np.array(covlrp_marg.cpu(), dtype = np.float32)).sum()

        covgi_diag, covgi_marg = covgi_vgg16_simple_deep_ensembles(models_simple, x)
        covgi_diag_acc = np.nan_to_num(np.array(covgi_diag.cpu(), dtype = np.float32)).sum()
        covgi_marg_acc = np.nan_to_num(np.array(covgi_marg.cpu(), dtype = np.float32)).sum()

        img_exp_dir_path = os.path.join(tgt_dir, img_dir[:-5] + "_simple_exp")
        utils.recreate_directory(img_exp_dir_path)
        image.save(os.path.join(img_exp_dir_path, "main_img.png"))
        for img_att_rel_path in os.listdir(os.path.join(src_dir, img_dir)):
            if img_att_rel_path ==  img_dir[:-5] + ".jpg":
                continue

            img_att_path = os.path.join(src_dir, img_dir,img_att_rel_path)
            img_att_name = ("_".join(img_att_rel_path.split("_")[1:]))[:-4]
            image_att = Image.open(img_att_path).convert('L').resize((224, 224))
            mask_img_att = np.asarray(image_att, np.uint8)/255

            lrp_exp_img = np.nan_to_num(np.array(lrp_rel.cpu(), dtype = np.float32)) 
            lrp_mask = (lrp_exp_img * mask_img_att).sum()
            simple_exp_dict["lrp_exp"].setdefault(img_att_name, 0)
            simple_exp_dict["lrp_exp"][img_att_name] += lrp_mask
            lrp_acc -= lrp_mask
            utils_vgg16.heatmap(lrp_exp_img, 20,20, save_path= os.path.join(img_exp_dir_path, "lrp_exp.png"))

            gi_exp_img = np.nan_to_num(np.array(gi_rel.cpu(), dtype = np.float32)) 
            gi_mask = (gi_exp_img * mask_img_att).sum()
            simple_exp_dict["gi_exp"].setdefault(img_att_name, 0)
            simple_exp_dict["gi_exp"][img_att_name] += gi_mask
            gi_acc -= gi_mask
            utils_vgg16.heatmap(gi_exp_img, 20,20, save_path= os.path.join(img_exp_dir_path, "gi_exp.png"))

            covlrp_exp_diag_img = np.nan_to_num(np.array(covlrp_diag.cpu(), dtype = np.float32)) 
            covlrp_diag_mask = (covlrp_exp_diag_img* mask_img_att).sum()
            simple_exp_dict["covlrp_diag_exp"].setdefault(img_att_name, 0)
            simple_exp_dict["covlrp_diag_exp"][img_att_name] += covlrp_diag_mask
            covgi_diag_acc -= covlrp_diag_mask
            utils_vgg16.heatmap(covlrp_exp_diag_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covlrp_diag_exp.png"))

            covlrp_exp_marg_img = np.nan_to_num(np.array(covlrp_marg.cpu(), dtype = np.float32))
            covlrp_marg_mask = (covlrp_exp_marg_img * mask_img_att).sum()
            simple_exp_dict["covlrp_marg_exp"].setdefault(img_att_name, 0)
            simple_exp_dict["covlrp_marg_exp"][img_att_name] += covlrp_marg_mask
            covlrp_marg_acc -= covlrp_marg_mask
            utils_vgg16.heatmap(covlrp_exp_marg_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covlrp_marg_exp.png"))

            covgi_exp_diag_img = np.nan_to_num(np.array(covgi_diag.cpu(), dtype = np.float32))
            covgi_diag_mask = ( covgi_exp_diag_img * mask_img_att).sum()
            simple_exp_dict["covgi_diag_exp"].setdefault(img_att_name, 0)
            simple_exp_dict["covgi_diag_exp"][img_att_name] += covgi_diag_mask
            covgi_diag_acc -= covgi_diag_mask
            utils_vgg16.heatmap(covgi_exp_diag_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covgi_diag_exp.png"))

            covgi_exp_marg_img = np.nan_to_num(np.array(covgi_marg.cpu(), dtype = np.float32)) 
            covgi_marg_mask = (covgi_exp_marg_img * mask_img_att).sum()
            simple_exp_dict["covgi_marg_exp"].setdefault(img_att_name, 0)
            simple_exp_dict["covgi_marg_exp"][img_att_name] += covgi_marg_mask
            covgi_marg_acc -= covgi_marg_mask
            utils_vgg16.heatmap(covgi_exp_marg_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covgi_marg_exp.png"))

        
        simple_exp_dict["lrp_exp"]["background"] = lrp_acc
        simple_exp_dict["gi_exp"]["background"] = gi_acc
        simple_exp_dict["covlrp_diag_exp"]["background"] = covlrp_diag_acc
        simple_exp_dict["covlrp_marg_exp"]["background"] = covlrp_marg_acc
        simple_exp_dict["covgi_diag_exp"]["background"] = covgi_diag_acc
        simple_exp_dict["covgi_marg_exp"]["background"] = covgi_marg_acc


    with open(log_file_1, 'w') as log_file: 
        # log_file.write('Sum Features: \n')

        for exp_name, atts_dict in simple_exp_dict.items():
            log_file.write(exp_name + ":\n\n")
            sum_atts = sum(list(atts_dict.values()))
            for att_name, att_value in atts_dict.items():
                log_file.write(f"{att_name}: {att_value/sum_atts}\n")

            log_file.write("\n\n\n")

    del models_simple



    models_density = []
    for i in range(10):
        load_path = f"model_dirs_face/train_ensemble_density/model_{i}.pt"
        model = vgg16DensityNet()
        model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), load_path)))
        model.eval()
        model.to(device)
        models_density.append(model)


    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    density_exp_dict = {"lrp_exp" : {}, "gi_exp" : {}, "covlrp_diag_exp" : {}, "covlrp_marg_exp" : {}, "covgi_diag_exp" : {}, "covgi_marg_exp" : {}}


    for img_dir in tqdm(sorted(os.listdir(src_dir))):
        main_img_path = os.path.join(src_dir, img_dir, img_dir[:-5] + ".jpg")
        image = Image.open(main_img_path).convert('RGB').resize((224, 224))
        x = data_transforms(image).unsqueeze(0).to(device)
        lrp_rel = lrp_vgg16_density_deep_ensembles(models_density, x)
        lrp_acc = np.nan_to_num(np.array(lrp_rel.cpu(), dtype = np.float32)).sum()

        gi_rel = gi_vgg16_density_deep_ensembles(models_density, x)
        gi_acc = np.nan_to_num(np.array(gi_rel.cpu(), dtype = np.float32)).sum()
  
        covlrp_diag, covlrp_marg = covlrp_vgg16_density_deep_ensembles(models_density, x)
        covlrp_diag_acc = np.nan_to_num(np.array(covlrp_diag.cpu(), dtype = np.float32)).sum()
        covlrp_marg_acc = np.nan_to_num(np.array(covlrp_marg.cpu(), dtype = np.float32)).sum()

        covgi_diag, covgi_marg = covgi_vgg16_density_deep_ensembles(models_density, x)
        covgi_diag_acc = np.nan_to_num(np.array(covgi_diag.cpu(), dtype = np.float32)).sum()
        covgi_marg_acc = np.nan_to_num(np.array(covgi_marg.cpu(), dtype = np.float32)).sum()

        img_exp_dir_path = os.path.join(tgt_dir, img_dir[:-5] + "_density_exp")
        utils.recreate_directory(img_exp_dir_path)
        image.save(os.path.join(img_exp_dir_path, "main_img.png"))
        for img_att_rel_path in os.listdir(os.path.join(src_dir, img_dir)):
            if img_att_rel_path ==  img_dir[:-5] + ".jpg":
                continue

            img_att_path = os.path.join(src_dir, img_dir,img_att_rel_path)
            img_att_name = ("_".join(img_att_rel_path.split("_")[1:]))[:-4]
            image_att = Image.open(img_att_path).convert('L').resize((224, 224))
            mask_img_att = np.asarray(image_att, np.uint8)/255

            lrp_exp_img = np.nan_to_num(np.array(lrp_rel.cpu(), dtype = np.float32)) 
            lrp_mask = (lrp_exp_img * mask_img_att).sum()
            density_exp_dict["lrp_exp"].setdefault(img_att_name, 0)
            density_exp_dict["lrp_exp"][img_att_name] += lrp_mask
            lrp_acc -= lrp_mask
            utils_vgg16.heatmap(lrp_exp_img, 20,20, save_path= os.path.join(img_exp_dir_path, "lrp_exp.png"))

            gi_exp_img = np.nan_to_num(np.array(gi_rel.cpu(), dtype = np.float32)) 
            gi_mask = (gi_exp_img * mask_img_att).sum()
            density_exp_dict["gi_exp"].setdefault(img_att_name, 0)
            density_exp_dict["gi_exp"][img_att_name] += gi_mask
            gi_acc -= gi_mask
            utils_vgg16.heatmap(gi_exp_img, 20,20, save_path= os.path.join(img_exp_dir_path, "gi_exp.png"))

            covlrp_exp_diag_img = np.nan_to_num(np.array(covlrp_diag.cpu(), dtype = np.float32)) 
            covlrp_diag_mask = (covlrp_exp_diag_img* mask_img_att).sum()
            density_exp_dict["covlrp_diag_exp"].setdefault(img_att_name, 0)
            density_exp_dict["covlrp_diag_exp"][img_att_name] += covlrp_diag_mask
            covgi_diag_acc -= covlrp_diag_mask
            utils_vgg16.heatmap(covlrp_exp_diag_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covlrp_diag_exp.png"))

            covlrp_exp_marg_img = np.nan_to_num(np.array(covlrp_marg.cpu(), dtype = np.float32))
            covlrp_marg_mask = (covlrp_exp_marg_img * mask_img_att).sum()
            density_exp_dict["covlrp_marg_exp"].setdefault(img_att_name, 0)
            density_exp_dict["covlrp_marg_exp"][img_att_name] += covlrp_marg_mask
            covlrp_marg_acc -= covlrp_marg_mask
            utils_vgg16.heatmap(covlrp_exp_marg_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covlrp_marg_exp.png"))

            covgi_exp_diag_img = np.nan_to_num(np.array(covgi_diag.cpu(), dtype = np.float32))
            covgi_diag_mask = ( covgi_exp_diag_img * mask_img_att).sum()
            density_exp_dict["covgi_diag_exp"].setdefault(img_att_name, 0)
            density_exp_dict["covgi_diag_exp"][img_att_name] += covgi_diag_mask
            covgi_diag_acc -= covgi_diag_mask
            utils_vgg16.heatmap(covgi_exp_diag_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covgi_diag_exp.png"))

            covgi_exp_marg_img = np.nan_to_num(np.array(covgi_marg.cpu(), dtype = np.float32)) 
            covgi_marg_mask = (covgi_exp_marg_img * mask_img_att).sum()
            density_exp_dict["covgi_marg_exp"].setdefault(img_att_name, 0)
            density_exp_dict["covgi_marg_exp"][img_att_name] += covgi_marg_mask
            covgi_marg_acc -= covgi_marg_mask
            utils_vgg16.heatmap(covgi_exp_marg_img, 20,20, save_path= os.path.join(img_exp_dir_path, "covgi_marg_exp.png"))

        
        density_exp_dict["lrp_exp"]["background"] = lrp_acc
        density_exp_dict["gi_exp"]["background"] = gi_acc
        density_exp_dict["covlrp_diag_exp"]["background"] = covlrp_diag_acc
        density_exp_dict["covlrp_marg_exp"]["background"] = covlrp_marg_acc
        density_exp_dict["covgi_diag_exp"]["background"] = covgi_diag_acc
        density_exp_dict["covgi_marg_exp"]["background"] = covgi_marg_acc

    
    with open(log_file_2, 'w') as log_file: 

        for exp_name, atts_dict in density_exp_dict.items():
            log_file.write(exp_name + ":\n\n")
            sum_atts = sum(list(atts_dict.values()))
            for att_name, att_value in atts_dict.items():
                log_file.write(f"{att_name}: {att_value/sum_atts}\n")

            log_file.write("\n\n\n")
        

def plot_exp_bar():
    # simple

    # x_labels= ["Eyeglasses", "Eyes", "Eyebrows", "Mouth", "Nose", "Hair", "Neck", "Ears", "Skin", "Clothes"]
    # y_values_lrp = [13.54, 3.34, 2.23, 9.6, 5.4, 18.2, 12.2, 0,  15.2, 20.42]
    # y_values_gi =  [30.54, 1.66, 1.24, 5.6, 4.3, 6.8, 25.56, 6.32, 18.23, 0]
    # y_values_covlrp_diag = [27.54, 3.56, 2.54, 4.7, 3.4, 18.45, 7.2, 6.3, 20.23, 6.23]
    # y_values_covlrp_marg = [25.54, 4.56, 3.21, 3.23, 3.56, 30.64, 5.4, 12.3, 0, 11.54]
    # y_values_covgi_diag = [18.9, 2.56, 3.6, 3.8, 3.1, 20.6, 4.1, 3.1, 24.3, 16.2]
    # y_values_covgi_marg  = [24.3, 2.3, 2.1, 4.2, 4.3, 17.3, 6.7, 5.3, 23.2, 10.2]

    # plt.bar(x_labels, y_values_covgi_marg, color = "slateblue")
    # plt.xticks(rotation=90, fontsize=15)
    # plt.xlabel('Visual Features', fontsize=20)
    # plt.ylabel('Uncertainty [%]', fontsize=20)  
    # ax = plt.gca()
    # labels = ax.get_xticklabels()
    # labels[0].set_fontweight('bold') 
    # plt.tight_layout()
    # plt.show()

    # density

    x_labels= ["Eyeglasses", "Eyes", "Eyebrows", "Mouth", "Nose", "Hair", "Neck", "Ears", "Skin", "Clothes"]

    y_values_lrp = [25.6, 3.4,  2.1, 2.9, 1.2, 15.3, 16.2, 1.6, 16.2, 15.5]
    y_values_gi =  [42.54, 4.66, 5.6, 4.6, 4.3, 12.3, 0, 6.32, 12.3, 7.3]
    y_values_covlrp_diag = [22.6, 3.26, 3.34, 5.2, 6.3, 14.2, 3.3, 2.3, 20.4, 19.24]
    y_values_covlrp_marg = [40.04, 4.2, 5.2, 2.3, 6.4,0, 3.3, 5.5, 20.56, 12.3]
    y_values_covgi_diag = [25.6, 3.48, 5.6, 4.3, 6.4, 10.3, 4.5, 3.1, 29.5, 7.3]
    y_values_covgi_marg= [18.6, 3.2, 12.3, 6.6, 8.2, 28.3, 5.2, 11.4, 0, 6.2]


    plt.bar(x_labels, y_values_covgi_marg, color = "indianred")
    plt.xticks(rotation=90, fontsize=15)
    plt.xlabel('Visual Features', fontsize=20)
    plt.ylabel('Uncertainty [%]', fontsize=20)  
    ax = plt.gca()
    labels = ax.get_xticklabels()
    labels[0].set_fontweight('bold') 
    plt.tight_layout()
    plt.show()
 




    # img_path = "datasets_face/CelebAMask-HQ/CelebA-Eye-G/4460_atts/4460.jpg"
    # image = Image.open(img_path).convert('RGB')
    # x = data_transforms(image).unsqueeze(0).to(device)


    # lrp_rel = lrp_vgg16_simple_deep_ensembles(models_simple, x)
    # utils_vgg16.heatmap(np.array(lrp_rel.cpu(), dtype = np.float32),20,20)
    # gi_rel = gi_vgg16_simple_deep_ensembles(models_simple, x)
    # utils_vgg16.heatmap(np.array(gi_rel.cpu(), dtype = np.float32),20,20)
    # covlrp_diag, covlrp_marg = covlrp_vgg16_simple_deep_ensembles(models_simple, x)
    # utils_vgg16.heatmap(np.array(covlrp_diag.cpu(), dtype = np.float32),20,20)
    # utils_vgg16.heatmap(np.array(covlrp_marg.cpu(), dtype = np.float32),20,20)
    # covgi_diag, covgi_marg = covgi_vgg16_simple_deep_ensembles(models_simple, x)
    # utils_vgg16.heatmap(np.array(covgi_diag.cpu(), dtype = np.float32),20,20)
    # utils_vgg16.heatmap(np.array(covgi_marg.cpu(), dtype = np.float32),20,20)



    



    # lrp_rel = lrp_vgg16_density_deep_ensembles(models_density, x)
    # utils_vgg16.heatmap(np.array(lrp_rel.cpu(), dtype = np.float32),20,20)
    # gi_rel = gi_vgg16_density_deep_ensembles(models_density, x)
    # utils_vgg16.heatmap(np.array(gi_rel.cpu(), dtype = np.float32),20,20)
    # covlrp_diag, covlrp_marg = covlrp_vgg16_density_deep_ensembles(models_density, x)
    # utils_vgg16.heatmap(np.array(covlrp_diag.cpu(), dtype = np.float32),20,20)
    # utils_vgg16.heatmap(np.array(covlrp_marg.cpu(), dtype = np.float32),20,20)

    # covgi_diag, covgi_marg = covgi_vgg16_density_deep_ensembles(models_density, x)
    # utils_vgg16.heatmap(np.array(covgi_diag.cpu(), dtype = np.float32),20,20)
    # utils_vgg16.heatmap(np.array(covgi_marg.cpu(), dtype = np.float32),20,20)

# 1151

                
def main():

    plot_exp_bar()

if __name__ == '__main__':
    main()