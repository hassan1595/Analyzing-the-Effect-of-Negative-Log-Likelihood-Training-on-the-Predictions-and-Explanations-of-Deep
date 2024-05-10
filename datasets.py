import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

class Datasets(Dataset):

    def __init__(self, dataset_name, transform=None):

        self.dataset_name = dataset_name
        if dataset_name != "boston" and dataset_name != "concrete" and dataset_name != "energy" and dataset_name != "wine" and dataset_name != "yacht":
            raise ValueError("Dataset not defined")
        self.transform = transform

        if dataset_name == "boston":
            dataset_file = "datasets/boston-housing-dataset.csv"
        if dataset_name == "concrete":
            dataset_file = "datasets/concrete_data.csv"
        if dataset_name == "energy":
            dataset_file = "datasets/energy.csv"
        if dataset_name == "wine":
            dataset_file = "datasets/wine.csv"
        if dataset_name == "yacht":
            dataset_file =  "datasets/yacht_hydro.csv"

        dataset = pd.read_csv(dataset_file)
        if dataset_name == "wine":
            dataset = dataset.drop(columns=['type'])
        print(dataset.head())
        

        scaler = StandardScaler()
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        input_np_array = np.array(dataset[:,1:-1].tolist(), dtype='float32')
        input_np_array = scaler.fit_transform(input_np_array)
        self.n_features = input_np_array.shape[1]
        self.inputs = torch.from_numpy(input_np_array)
        targets_np_array = np.array(dataset[:,-1].tolist(), dtype='float32')
        targets_np_array_rs = np.reshape(targets_np_array, (targets_np_array.shape[0],1))
        self.targets = torch.from_numpy(targets_np_array_rs)
      
      
    def __getitem__(self, index):
         
        if self.transform == None:
            return self.inputs[index], self.targets[index]
        
        return self.transform(self.inputs[index]), self.transform(self.targets[index])
    
    def __len__(self):
        return len(self.inputs)
    
                                                                    
