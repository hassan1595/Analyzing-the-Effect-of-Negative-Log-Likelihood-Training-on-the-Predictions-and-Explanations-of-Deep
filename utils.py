import math
import torch
import numpy as np
import os
import shutil
from models import SimpleNet, SimpleDensityNet
from datasets import Datasets
from sklearn.neighbors import KernelDensity
import copy
import utils_vgg16
def recreate_directory(dir_path):
    """
    Checks if the directory exists, and if it does, deletes and recreates it.
    If the directory does not exist, creates it.

    Parameters:
    dir_path (str): The path of the directory to recreate.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    print(f"Directory '{dir_path}' has been recreated.")

def mse_loss(y_true, y_pred):

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return ((y_true - y_pred) ** 2).mean()


def nll_loss(y_true, y_pred, eps = 1e-05):

    y_true = y_true.flatten()
    mu,var = y_pred
    mu = mu.flatten()
    var = var.flatten()
    
    return torch.nn.GaussianNLLLoss(eps=eps)(mu, y_true, var)

def nll_laplace_loss(y_true, y_pred, eps = 1e-05):

    mu, var = y_pred
    var = torch.clamp(var, min=eps)
    loss = torch.log(var) + torch.abs(mu - y_true) / var
    return loss.mean()


def new_layer(layer, g):
    layer = copy.deepcopy(layer)
    try: layer.weight = torch.nn.Parameter(g(layer.weight))
    except AttributeError : pass
    try: layer.bias = torch.nn.Parameter(g(layer.bias))
    except AttributeError : pass
    return layer

def gi_apply_simple_deep_ensembles(models, input):
    input = input.clone()
    input_ex = input.requires_grad_(True)
    if input_ex.grad is not None:
        input_ex.grad.zero_()
    outputs = []
    for model in models:
        outputs.append(model(input_ex))
    var = torch.stack(outputs).var(correction=0)
    var.backward()
    x = (input_ex * input_ex.grad).data
    return x

def gi_apply_density_deep_ensembles(models, input):
    input = input.clone()
    input_ex = input.requires_grad_(True)
    if input_ex.grad is not None:
        input_ex.grad.zero_()
    mus =[]
    vars = []
    for model in models:
        mu, var = model(input_ex)
        mus.append(mu)
        vars.append(var)
    cat_mu_tenosr = torch.stack(mus,)

    cat_var_tensor = torch.stack(vars)
    vars_mu = (cat_var_tensor).mean()
    mus_var = cat_mu_tenosr.var(correction=0)
    var_comp = vars_mu + mus_var
    var_comp.backward()
    x = (input_ex * input_ex.grad).data
    return x



def lrp_apply_simple_deep_ensembles(models, input):
    input = input.clone()
    activations_all = []
    ro = lambda p : p + 0.2*p.clamp(min = 0) 
  
    for model in models:

        model.eval()
        layers = [model.l_1, model.relu, model.l_2]
        activations = []
        activations.append(input)
        for idx, layer in enumerate(layers):
            if idx == 0:
                activations.append(layer(input).data)
            else:
                activations.append(layer(activations[idx]).data)

        activations_all.append(activations)

    last_layer_activations = [activations_all[m][-1].squeeze() for m in range(len(models))]
    last_layer_activations_var = (torch.stack(last_layer_activations) - torch.stack(last_layer_activations).mean() )**2
    last_layer_relevances = (last_layer_activations_var/len(models)).data
    relevances_all = []

    for m, model in enumerate(models):
        model.eval()
        layers = [model.l_1, model.relu, model.l_2]
        relevances = [None] * 3
        relevances.append(last_layer_relevances[m])
        activations = activations_all[m]
        for layer_idx in range(3)[::-1]:

            if layer_idx == 2:
                activations[layer_idx] = activations[layer_idx].requires_grad_(True)
                if activations[layer_idx].grad is not None:
                    activations[layer_idx].grad.zero_()             
                z = new_layer(layers[layer_idx],ro)(activations[layer_idx])
                s = (last_layer_relevances[m]/(z+1e-9)).data 
                (z * s).sum().backward()
                c = activations[layer_idx].grad
                relevances[layer_idx] = (c * activations[layer_idx ]).data
   
            if layer_idx == 1:
                relevances[layer_idx] = relevances[layer_idx +1]  
    
            if layer_idx == 0:
                activations[layer_idx] = activations[layer_idx].requires_grad_(True)  
                if activations[layer_idx].grad is not None:
                    activations[layer_idx].grad.zero_()          
                z = new_layer(layers[layer_idx],ro)(activations[layer_idx])
                s = (relevances[layer_idx +1]/(z+1e-9)).data 
                (z * s).sum().backward()
                c = activations[layer_idx].grad
                relevances[layer_idx] = (c * activations[layer_idx ]).data
            
        relevances_all.append(relevances)

    return torch.stack([re[0] for re in relevances_all]).sum(dim = 0)
    

def lrp_apply_density_deep_ensembles(models, input):
    input = input.clone()
    ro = lambda p : p + 0.2*p.clamp(min = 0) 
    activations_all = []
    for model in models:

        model.eval()
        layers = [model.l_1, model.relu, model.l_2]
        var_layer = model.l_3
        activations = []
        activations.append(input)
        for idx, layer in enumerate(layers):
            if idx == 0:
                activations.append(layer(input).data)
            elif idx == len(layers)- 1:
                activations.append(torch.cat( (layer(activations[idx]).data, var_layer(activations[idx]).abs().data)) )
            else:
                activations.append(layer(activations[idx]).data)

        activations_all.append(activations)

    mu_activations = [activations_all[m][-1][0].squeeze() for m in range(len(models))]
    
    last_layer_activations_mu = (torch.stack(mu_activations) - torch.stack(mu_activations).mean() )**2
    last_layer_activations_mu = (last_layer_activations_mu/len(models)).data 
    last_layer_activations_var = (torch.stack([activations_all[m][-1][1].squeeze() for m in range(len(models))]).data)/len(models)
 
    relevances_all = []

    for m, model in enumerate(models):
        model.eval()
        layers = [model.l_1, model.relu, model.l_2]
        var_layer = model.l_3
        relevances = [None] * 3
        relevances.append(torch.stack([last_layer_activations_mu[m],last_layer_activations_var[m]] ))
        activations = activations_all[m]

        for layer_idx in range(3)[::-1]:

            if layer_idx == 2:
                activations[layer_idx] = activations[layer_idx].requires_grad_(True)   
                if activations[layer_idx].grad is not None:
                    activations[layer_idx].grad.zero_()          
                z_1 = new_layer(layers[layer_idx],ro)(activations[layer_idx])
                z_2 = new_layer(var_layer,ro)(activations[layer_idx]).abs()
                s_1 = (relevances[layer_idx +1][0]/(z_1+1e-9)).data 
                s_2 = (relevances[layer_idx +1][1]/(z_2+1e-9)).data
                ((z_1 * s_1).sum() + (z_2 * s_2).sum()) .backward()
                c = activations[layer_idx].grad
                relevances[layer_idx] = (c * activations[layer_idx ]).data
        

            if layer_idx == 1:
                relevances[layer_idx] = relevances[layer_idx +1]  
        
      
    
            if layer_idx == 0:
            
                activations[layer_idx] = activations[layer_idx].requires_grad_(True) 
                if activations[layer_idx].grad is not None:
                    activations[layer_idx].grad.zero_()
                z = new_layer(layers[layer_idx],ro)(activations[layer_idx])
                s = (relevances[layer_idx +1]/(z+1e-9)).data 
                (z * s).sum().backward()
                c = activations[layer_idx].grad
                relevances[layer_idx] = (c * activations[layer_idx ]).data
        

        relevances_all.append(relevances)

    return torch.stack([re[0] for re in relevances_all]).sum(dim = 0)



def lrp_density(model, input, mu_propagate = True):
    model.eval()
    ro = lambda p : p + 0.2*p.clamp(min = 0) 
    layers = [model.l_1, model.relu, model.l_2]
    var_layer = model.l_3
    activations = []
    activations.append(input)
    for idx, layer in enumerate(layers):
        if idx == 0:
            activations.append(layer(input).data)
        elif idx == len(layers)- 1:
            activations.append(torch.cat( (layer(activations[idx]).data, var_layer(activations[idx]).abs().data)) )
        else:
            activations.append(layer(activations[idx]).data)

    relevances = [None] * 3
    relevances.append(activations[-1])

    for layer_idx in range(3)[::-1]:

        if layer_idx == 2:
            activations[layer_idx] = activations[layer_idx].requires_grad_(True)   
            if activations[layer_idx].grad is not None:
                activations[layer_idx].grad.zero_() 
            if mu_propagate:
                z = new_layer(layers[layer_idx],ro)(activations[layer_idx])
                s = (relevances[layer_idx +1][0]/(z+1e-9)).data 
            else:
                z = new_layer(var_layer,ro)(activations[layer_idx]).abs()
                s = (relevances[layer_idx +1][1]/(z+1e-9)).data
            ((z * s).sum()) .backward()
            c = activations[layer_idx].grad
            relevances[layer_idx] = (c * activations[layer_idx ]).data
  

        if layer_idx == 1:
            relevances[layer_idx] = relevances[layer_idx +1]  
            



        if layer_idx == 0:
        
            activations[layer_idx] = activations[layer_idx].requires_grad_(True) 
            if activations[layer_idx].grad is not None:
                activations[layer_idx].grad.zero_()
            z = new_layer(layers[layer_idx],ro)(activations[layer_idx])
            s = (relevances[layer_idx +1]/(z+1e-9)).data 
            (z * s).sum().backward()
            c = activations[layer_idx].grad
            relevances[layer_idx] = (c * activations[layer_idx ]).data
      
    return relevances[0]




def covlrp_apply_density_deep_ensembles(models, input):
    input = input.clone()
    relevances_mu = []
    relevances_var = []
    for model in models:
        relevances_mu.append(lrp_density(model, input, mu_propagate = True))
        relevances_var.append(lrp_density(model, input, mu_propagate = False))

    relevances_mu_tensor = torch.stack(relevances_mu)    
    relevances_mu_tensor -=  relevances_mu_tensor.mean(dim = 0)
    cov_mat = (relevances_mu_tensor.T @ relevances_mu_tensor)/(len(models))
    diag = cov_mat.diag()

    d = cov_mat.shape[0]
    marg = torch.zeros_like(diag)

    for i in range(d):
        # Diagonal element contribution
        marg[i] = cov_mat[i, i]
        
        # Off-diagonal element contributions
        for j in range(d):
            if i != j:
                marg[i] += 0.5 * cov_mat[i, j] + 0.5 * cov_mat[j, i] 


    relevances_var_tensor = torch.stack(relevances_var).mean(dim = 0)
    return diag + relevances_var_tensor, marg + relevances_var_tensor

def lrp_simple(model, input):
    ro = lambda p : p + 0.2*p.clamp(min = 0) 
    model.eval()
    layers = [model.l_1, model.relu, model.l_2]
    activations = []
    activations.append(input)
    for idx, layer in enumerate(layers):
        if idx == 0:
            activations.append(layer(input).data)
        else:
            activations.append(layer(activations[idx]).data)

    relevances = [None] * 3
    relevances.append(activations[-1])
    
    for layer_idx in range(3)[::-1]:
        activations[layer_idx] = activations[layer_idx].requires_grad_(True)
        if activations[layer_idx].grad is not None:
            activations[layer_idx].grad.zero_()             
        z = new_layer(layers[layer_idx],ro)(activations[layer_idx])
        s = (relevances[layer_idx +1]/(z+1e-9)).data 
        (z * s).sum().backward()
        c = activations[layer_idx].grad
        relevances[layer_idx] = (c * activations[layer_idx ]).data

    return relevances[0].data


def covlrp_apply_simple_deep_ensembles(models, input):
    input = input.clone()
    relevances_all = []
    for model in models:
        relevances_all.append(lrp_simple(model, input))

    relevances_tensor = torch.stack(relevances_all)    
    relevances_tensor -=  relevances_tensor.mean(dim = 0)
    cov_mat = (relevances_tensor.T @ relevances_tensor)/(len(models))
    diag = cov_mat.diag()

    d = cov_mat.shape[0]
    marg = torch.zeros_like(diag)

    for i in range(d):
        # Diagonal element contribution
        marg[i] = cov_mat[i, i]
        
        # Off-diagonal element contributions
        for j in range(d):
            if i != j:
                marg[i] += 0.5 * cov_mat[i, j] + 0.5 * cov_mat[j, i] 

    return diag, marg
    
def gi_apply_simple(model, input):
    input = input.clone()
    input_ex = input.requires_grad_(True)
    if input_ex.grad is not None:
        input_ex.grad.zero_()
    y = model(input_ex)
    y.backward()
    return (input_ex * input_ex.grad).data


def gi_apply_density(model, input, mu_propagate = True):
    input = input.clone()
    input_ex = input.requires_grad_(True)
    if input_ex.grad is not None:
        input_ex.grad.zero_()
    mu, var = model(input_ex)
    if mu_propagate:
        mu.backward()
    else:
        var.backward()
    return (input_ex * input_ex.grad).data


def covgi_apply_simple_deep_ensembles(models, input):
    relevances_all = []
    for model in models:
        relevances_all.append(gi_apply_simple(model, input))

    relevances_tensor = torch.stack(relevances_all)    
    relevances_tensor -=  relevances_tensor.mean(dim = 0)
    cov_mat = (relevances_tensor.T @ relevances_tensor)/(len(models))
    diag = cov_mat.diag()

    d = cov_mat.shape[0]
    marg = torch.zeros_like(diag)

    for i in range(d):
        # Diagonal element contribution
        marg[i] = cov_mat[i, i]
        
        # Off-diagonal element contributions
        for j in range(d):
            if i != j:
                marg[i] += 0.5 * cov_mat[i, j] + 0.5 * cov_mat[j, i] 

    return diag, marg

def covgi_apply_density_deep_ensembles(models, input):
    relevances_mu = []
    relevances_var = []
    for model in models:
        relevances_mu.append(gi_apply_density(model, input, mu_propagate=True))
        relevances_var.append(gi_apply_density(model, input, mu_propagate=False))

    relevances_mu_tensor = torch.stack(relevances_mu)    
    relevances_mu_tensor -=  relevances_mu_tensor.mean(dim = 0)
    cov_mat = (relevances_mu_tensor.T @ relevances_mu_tensor)/(len(models))
    diag = cov_mat.diag()

    d = cov_mat.shape[0]
    marg = torch.zeros_like(diag)

    for i in range(d):
        # Diagonal element contribution
        marg[i] = cov_mat[i, i]
        
        # Off-diagonal element contributions
        for j in range(d):
            if i != j:
                marg[i] += 0.5 * cov_mat[i, j] + 0.5 * cov_mat[j, i] 

    relevances_var_tensor = torch.stack(relevances_var).mean(dim = 0)
    return diag + relevances_var_tensor, marg + relevances_var_tensor


def kl_divergence(p ,q, epsilon = 1e-5):
    p = torch.clamp(p + epsilon, min=epsilon)
    q = torch.clamp(q + epsilon, min=epsilon)
    return (p * ( (p / q) ).log()).sum()


def softmax(x, temp = 1, norm = True):

    if norm:
        x_max = torch.max(x)
        return torch.exp((x - x_max)/temp)/(torch.exp((x-x_max)/temp)).sum()
    else:
        return torch.exp(x /temp)/(torch.exp((x)/temp)).sum()


def models_simple_infer(models, inp):

    mus = []
    for model in models:
        mu = model(inp)
        mus.append(mu.squeeze())
    return torch.stack(mus).var(correction = 0)

def models_density_infer(models, inp):

  
    mus = []
    vars = []
    for model in models:
        mu, var = model(inp)
        mus.append(mu.squeeze())
        vars.append(var.squeeze())

    mus_var = torch.stack(mus).var(correction = 0)
    vars_mu = torch.stack(vars).mean()
    return mus_var + vars_mu



def dec_measure(t_1, t_2, rep_1, rep_2, limit = None):

    if type(limit) == str:
        end = int(limit)
    else:
        end = math.ceil((rep_1.shape[0]) * limit)

    s_rep_1 = torch.sort(softmax(rep_1).data, descending=True)[0]
    d_1 = 0
    for i in range(end):
        d_1 += s_rep_1[i] * (t_1[i]/(t_1[i+1]))

    d_2 = 0
    s_rep_2 = torch.sort(softmax(rep_2).data, descending=True)[0]
    for i in range(end):
        d_2 += s_rep_2[i] * (t_2[i]/(t_2[i+1]))

    return d_1.cpu(), d_2.cpu()


def trapezoidal_area(y_values, limit):
  

    area = 0.0
    if type(limit) == str:
        end = int(limit) + 1
    else:
        end = math.ceil((y_values.shape[0] - 1) * limit) + 1
    for i in range(1, end):
        area += (y_values[i] + y_values[i-1]) / 2  
    return area

def aufc(t_1, t_2, limit):

    return trapezoidal_area(t_1, limit), trapezoidal_area(t_2, limit)

def get_best_kde(training_data):
    b_max = -1
    res_max = 0
    for b in np.linspace(0.1, 5, 100):
        
        kde = KernelDensity(kernel='gaussian', bandwidth= b).fit(training_data[:int(0.9 * training_data.shape[0])])
        res = kde.score_samples(training_data[int(0.9 * training_data.shape[0]):]).sum()
        if b_max == -1 or res_max < res:
            b_max = b 
            res_max = res

    kde = KernelDensity(kernel='gaussian', bandwidth= b_max).fit(training_data[:int(0.9 * training_data.shape[0])])
    return kde


def conditional_sample(kde, training_data, values, column_idx, steps, temp = 1):

    
    X_test = np.zeros((steps,training_data.shape[1]))

    col_data = training_data[:, column_idx]

    X_test +=  values
    X_test[:, column_idx] = np.linspace(col_data.min(),col_data.max(),steps)
    prob_dist = np.exp(kde.score_samples(X_test)/temp)/np.exp(kde.score_samples(X_test)/temp).sum()
    row_idx = np.random.choice(range(len(prob_dist)), p=prob_dist)
    return X_test[row_idx][column_idx]
     
 
def lrp_vgg16_simple(model, X):
    layers = list(model.vgg16._modules['features']) + utils_vgg16.toconv(list(model.vgg16._modules['classifier']))
    L = len(layers)
    A = [X]+[None]*L
    for l in range(L): A[l+1] = layers[l].forward(A[l])
    R = [None]*L + [(A[-1]).data]

    for l in range(0,L)[::-1]:
    
        A[l] = (A[l].data).requires_grad_(True)

        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)

        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):

            if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9

            z = incr(utils_vgg16.newlayer(layers[l],rho).forward(A[l]))  # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data                                   # step 4
            
        else:
            
            R[l] = R[l+1]


    A[0] = (A[0].data).requires_grad_(True)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1).to(X.device)
    std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1).to(X.device)

    lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= utils_vgg16.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= utils_vgg16.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data 

    utils_vgg16.heatmap(np.array(R[0][0].cpu(), dtype = np.float32).sum(axis=0),20,20)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ds = Datasets("boston", transform= lambda x : x.to(device), shuffle=False)
# m = 10

# kde = get_best_kde(ds.inputs)


# for inp, _ in ds:
#     l_1 = []
#     l_2 = []
#     for i in range(ds.n_features):
#         l_1.append(np.random.rand())
#         l_2.append(conditional_sample(kde, ds.inputs, inp.squeeze().cpu().numpy(), i, 200))

#     print()
#     print(((inp.squeeze().cpu().numpy() - np.array(l_1))**2).sum())
#     print(((inp.squeeze().cpu().numpy() - np.array(l_2))**2).sum())

#     print()



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ds = Datasets("student", transform= lambda x : x.to(device), shuffle=False)
# m = 10
# models_simple = []
# for i in range(m):
#     model = SimpleNet(
#                 ds.n_features, 50
#             ).to(device)
#     model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", "student", "train_ensemble",f"model_{i}.pt")))
#     model.eval()
#     models_simple.append(model)



# models_density = []
# for i in range(m):
#     model = SimpleDensityNet(
#                 ds.n_features, 50
#             ).to(device)
#     model.load_state_dict(torch.load( os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_dirs", "student", "train_ensemble_density",f"model_{i}.pt")))
#     model.eval()
#     models_density.append(model)



# n_true = 0
# n_false = 0
# d_simples = []
# d_densitys = [] 
# test_sampler = torch.utils.data.SubsetRandomSampler(range((int(len(ds) * 0.9)), len(ds) ))
# test_loader = torch.utils.data.DataLoader(ds, sampler=test_sampler)
# for idx, (input, _) in enumerate(test_loader):
   
#     input = input.squeeze()
#     input_simple = input.clone()
#     uncertainty_simple = []
#     uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
#     simple_exp = covlrp_apply_simple_deep_ensembles(models_simple, input_simple)[0]
#     idxs_sorted = simple_exp.cpu().numpy().argsort()[::-1]
   
#     for idx_sorted in idxs_sorted:

#         input_simple[idx_sorted] = 0
#         uncertainty_simple.append(models_simple_infer(models_simple, input_simple).item())
        

#     input_density = input.clone()
#     uncertainty_density = []
#     uncertainty_density.append(models_density_infer(models_density, input_density).item())
#     density_exp = covlrp_apply_density_deep_ensembles(models_density, input_density)[0]
#     idxs_sorted = density_exp.cpu().numpy().argsort()[::-1]
#     for idx_sorted in idxs_sorted:

#         input_density[idx_sorted] = 0

#         uncertainty_density.append(models_density_infer(models_density, input_density).item())

#     d_simple, d_density = dec_measure(np.array(uncertainty_simple, dtype= np.float32), np.array(uncertainty_density, dtype=np.float32), simple_exp, density_exp) 
    
#     if d_density>d_simple:
#         n_true += 1
#     else:
#         n_false += 1

    
#     print(n_true/(n_true + n_false))
#     d_simples.append(d_simple)
#     d_densitys.append(d_density)

    # print("simple")
    # print(np.array(d_simples).mean())
    # print("density")
    # print(np.array(d_densitys).mean())   

    



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ds = Datasets("boston", transform= lambda x : x.to(device))
# # Density
# models = [SimpleDensityNet(
#                         13, 50
#                     ).to(device) for _ in range(5)]



# with torch.no_grad():
#     mus = []
#     vars = []
#     for model in models:
#         mu, var = model(ds[0][0])
#         mus.append(mu.squeeze())
#         vars.append(var.squeeze())

#     mus_var = torch.stack(mus).var(correction = 0)
#     vars_mu = torch.stack(vars).mean()



# print(mus_var + vars_mu)
# print(covlrp_apply_density_deep_ensembles(models, ds[0][0])[1].sum())
# print(covgi_apply_density_deep_ensembles(models, ds[0][0])[1].sum())
# print(lrp_apply_density_deep_ensembles(models, ds[0][0]).sum())
# print(gi_apply_density_deep_ensembles(models, ds[0][0]).sum())


# Simple

# models = [SimpleNet(
#                         13, 50
#                     ).to(device) for _ in range(5)]


# with torch.no_grad():
#     mus = []
#     for model in models:
#         mu = model(ds[0][0])
#         mus.append(mu.squeeze())
#     mus_var = torch.stack(mus).var(correction = 0)


# print(mus_var)

# print(covlrp_apply_simple_deep_ensembles(models, ds[0][0])[1].sum())
# print(lrp_apply_simple_deep_ensembles(models, ds[0][0]).sum())
# print(gi_apply_simple_deep_ensembles(models, ds[0][0]).sum())
