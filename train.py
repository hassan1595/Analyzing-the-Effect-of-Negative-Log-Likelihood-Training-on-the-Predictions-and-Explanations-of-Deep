import torch
import numpy as np
from models import SimpleNet, SimpleDensityNet
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
from utils import mse_loss, nll_loss


class TrainSimple:
    def __init__(
        self,
        train_loader,
        test_loader,
        n_epochs=400,
        lr=0.01,
        n_hidden_neurons=50,
        device = None,
        verbose = True
    ):
        
        inputs, _ = next(iter(train_loader))
        _, n_features = inputs.size()
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = SimpleNet(
                        n_features, n_hidden_neurons
                    ).to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose
        

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        for epoch_n in range(1, self.n_epochs + 1):
            train_loss = 0.0 
            inputs_len = 0
            for i, data in enumerate(self.train_loader):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                inputs_len += inputs.size(0)

            train_loss /= inputs_len
            if self.verbose:
                print(f"Epoch {epoch_n} loss: {train_loss}")

        return train_loss



    def test(self):
        self.model.eval()
        test_loss =0.0
        inputs_len = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, targets = data
                outputs = self.model(inputs)
                loss = mse_loss(targets, outputs)
                test_loss += loss.item() * inputs.size(0)
                inputs_len += inputs.size(0)
        
        test_loss /= inputs_len
        if self.verbose:
            print(f"Test loss: {test_loss}")
        return (test_loss,)


class TrainSimpleDensity:
    def __init__(
        self,
        train_loader,
        test_loader,
        n_epochs=400,
        lr=0.01,
        n_hidden_neurons=50,
        device = None,
        verbose = True
    ):
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        inputs, _ = next(iter(train_loader))
        _, n_features = inputs.size()
        self.model = SimpleDensityNet(
                        n_features, n_hidden_neurons
                    ).to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose
        

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()

        for epoch_n in range(1, self.n_epochs + 1):
            train_loss = 0.0 
            inputs_len = 0
            for i, data in enumerate(self.train_loader):
                inputs, targets = data
                optimizer.zero_grad()
                mu, var = self.model(inputs)
                loss = nll_loss(targets, (mu,var))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                inputs_len += inputs.size(0)

            train_loss /= inputs_len
            if self.verbose:
                print(f"Epoch {epoch_n} loss: {train_loss}")


        return train_loss



    def test(self):
        self.model.eval()
        test_loss =0.0
        test_loss_2=0.0
        inputs_len = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, targets = data
                mu, var = self.model(inputs)
     
                loss = nll_loss(targets, (mu,var))
                test_loss += loss.item() * inputs.size(0)
                loss_2 = mse_loss(targets, mu) 
                test_loss_2 += loss_2.item() * inputs.size(0)
                inputs_len += inputs.size(0)

            test_loss /= inputs_len
            test_loss_2 /= inputs_len

        if self.verbose:
            print(f"Test loss nll : {test_loss}")
            print(f"Test loss mse : {test_loss_2}")
        

        return test_loss, test_loss_2
    



        
class TrainEnsemble:
    def __init__(
        self,
        train_loaders,
        test_loader,
        n_epochs=400,
        lr=0.01,
        n_hidden_neurons=50,
        device = None,
        verbose = True
    ):
        
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        inputs, _ = next(iter(train_loaders[0]))
        _, n_features = inputs.size()
        self.m = len(train_loaders)
        self.models = [SimpleNet(
                        n_features, n_hidden_neurons
                    ).to(self.device) for _ in range(self.m)]
        
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose

    def train(self):
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.models
        ]
        for model in self.models:
            model.train()

        
        for epoch_n in range(1, self.n_epochs + 1):
            train_loss = 0.0 
            inputs_len = 0

            for i, ensemble_data in enumerate(zip(*self.train_loaders)):
                for m, data in enumerate(ensemble_data):

                    inputs, targets = data
                    optimizers[m].zero_grad()
                    outputs = self.models[m](inputs)
                    loss = mse_loss(targets, outputs)
                    loss.backward()
                    optimizers[m].step()
                    train_loss += loss.item() * inputs.size(0)
                    inputs_len += inputs.size(0)

            train_loss /= inputs_len
            if self.verbose:
                print(f"Epoch {epoch_n} loss: {train_loss}")

        return train_loss
        
    def test(self):
        for model in self.models:
            model.eval()

        test_loss =0.0
        test_loss_2=0.0
        inputs_len = 0

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, targets = data
                outputs = []
                for model in self.models:
                    outputs.append(model(inputs).unsqueeze(0))

                cat_tenosr = torch.cat(outputs, dim = 0)
                mu = cat_tenosr.mean(dim = 0)
                var = cat_tenosr.var(dim = 0, correction=0)
                test_loss += mse_loss(targets, mu).item() * inputs.size(0)
                test_loss_2 += nll_loss(targets, (mu,var)).item() * inputs.size(0)
                
                inputs_len  += inputs.size(0)


        test_loss /= inputs_len
        test_loss_2 /= inputs_len

        if self.verbose:
            print(f"Test loss mse : {test_loss }")
            print(f"Test loss nll : {test_loss_2}")
        

        return test_loss, test_loss_2
    


class TrainEnsembleDensity:

    def __init__(
        self,
        train_loaders,
        test_loader,
        n_epochs=400,
        lr=0.01,
        n_hidden_neurons=50,
        device = None,
        verbose = True
    ):
        
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        inputs, _ = next(iter(train_loaders[0]))
        _, n_features = inputs.size()
        self.m = len(train_loaders)
        self.models = [SimpleDensityNet(
                        n_features, n_hidden_neurons
                    ).to(self.device) for _ in range(self.m)]
        
        self.train_loaders = train_loaders
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.verbose = verbose


    def train(self):
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.lr) for model in self.models
        ]

        for model in self.models:
            model.train()

        inputs_len = 0
        for epoch_n in range(1, self.n_epochs + 1):
            train_loss = 0.0 

            for i, ensemble_data in enumerate(zip(*self.train_loaders)):
                for m, data in enumerate(ensemble_data):

                    inputs, targets = data
                    optimizers[m].zero_grad()
                    mu, var = self.models[m](inputs)
                    loss = nll_loss(targets, (mu,var))
                    loss.backward()
                    optimizers[m].step()
                    train_loss += loss.item() * inputs.size(0)
                    inputs_len += inputs.size(0)

            train_loss /= inputs_len
            if self.verbose:
                print(f"Epoch {epoch_n} loss: {train_loss}")
        return train_loss
    

    def test(self):
        for model in self.models:
            model.eval()

        test_loss =0.0
        test_loss_2=0.0
        inputs_len = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, targets = data
                mus =[]
                vars = []
                for model in self.models:
                    mu, var = model(inputs)
                    mus.append(mu.unsqueeze(0))
                    vars.append(var.unsqueeze(0))

                cat_mu_tenosr = torch.cat(mus, dim = 0)
                cat_var_tensor = torch.cat(vars, dim = 0)
                mus_mu = cat_mu_tenosr.mean(dim = 0)
                vars_mu = (cat_var_tensor).mean(dim = 0)
                mus_var = cat_mu_tenosr.var(dim = 0, correction=0)
                var_comp = vars_mu + mus_var
                test_loss += nll_loss(targets, (mus_mu,var_comp)).item() * inputs.size(0)
                test_loss_2 += mse_loss(targets, mus_mu).item() * inputs.size(0)
                inputs_len += inputs.size(0)


        test_loss /= inputs_len
        test_loss_2 /= inputs_len

        if self.verbose:
            print(f"Test loss nll : {test_loss }")
            print(f"Test loss mse : {test_loss_2}")

        return test_loss, test_loss_2
    
class TrainMCdropout:
    
    def __init__(
        self,
        train_loader,
        test_loader,
        m,
        dropout = 0.2,
        n_epochs=400,
        lr=0.01,
        n_hidden_neurons=50,
        device = None,
        verbose = True
    ):
        
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        inputs, _ = next(iter(train_loader))
        _, n_features = inputs.size()
        self.m = m

        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = SimpleNet(
                        n_features, n_hidden_neurons, dropout=dropout
                    ).to(self.device)
        self.verbose = verbose
        

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        

        for epoch_n in range(1, self.n_epochs + 1):
            train_loss = 0.0 
            inputs_len = 0
            for i, data in enumerate(self.train_loader):
                inputs, targets = data
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = mse_loss(targets, outputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                inputs_len += inputs.size(0)

            train_loss /= inputs_len
            if self.verbose:
                print(f"Epoch {epoch_n} loss: {train_loss}")

        return train_loss
    

    def test(self):
        self.model.train()

        test_loss =0.0
        test_loss_2=0.0
        inputs_len = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, targets = data
                outputs = []
                for _ in range(self.m):
                    outputs.append(self.model(inputs).unsqueeze(0))

                cat_tenosr = torch.cat(outputs, dim = 0)
                mu = cat_tenosr.mean(dim = 0)
                var = cat_tenosr.var(dim = 0, correction=0)
                test_loss += mse_loss(targets, mu).item() * inputs.size(0)
                test_loss_2 += nll_loss(targets, (mu,var)).item() * inputs.size(0)
                inputs_len += inputs.size(0)
                


        test_loss /= inputs_len
        test_loss_2 /= inputs_len

        if self.verbose:
            print(f"Test loss mse : {test_loss }")
            print(f"Test loss nll : {test_loss_2}")

        return test_loss, test_loss_2
    
                    




                
                
            
        



