import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import os
from torchvision import transforms
from models import *
class Datasets(Dataset):

    def __init__(self, dataset_name, transform=None, shuffle = True):

        self.dataset_name = dataset_name
        if dataset_name != "boston" and dataset_name != "concrete" and dataset_name != "energy" and dataset_name != "wine" and dataset_name != "student" and dataset_name != "california" and dataset_name != "mpg":
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
        if dataset_name == "student":
            dataset_file =  "datasets/Student_Performance.csv"
        if dataset_name == "california":
            dataset_file =  "datasets/housing.csv"
        if dataset_name == "mpg":
            dataset_file =  "datasets/auto-mpg.csv"


        dataset = pd.read_csv(dataset_file)
        dataset.dropna(inplace=True)
        if dataset_name == "wine":
            dataset = dataset.drop(columns=['type'])
        print(dataset.head())
        self.c_names = list(dataset.columns)
        scaler = StandardScaler()
        dataset = np.array(dataset)
        if shuffle:
            np.random.shuffle(dataset)
        input_np_array = np.array(dataset[:,:-1].tolist(), dtype='float32')
        input_np_array = scaler.fit_transform(input_np_array)
        self.n_features = input_np_array.shape[1]
        self.inputs = torch.from_numpy(input_np_array)
        self.dstmat = None
        targets_np_array = np.array(dataset[:,-1].tolist(), dtype='float32')
        targets_np_array_rs = np.reshape(targets_np_array, (targets_np_array.shape[0],1))
        self.targets = torch.from_numpy(targets_np_array_rs)
    

    def __getitem__(self, index):
         
        if self.transform == None:
            return self.inputs[index], self.targets[index]
        
        return self.transform(self.inputs[index]), self.transform(self.targets[index])
    
    def __len__(self):
        return len(self.inputs)
    
                                                                
# ds = Datasets("boston", shuffle= False)
# inp = ds[0][0]
# ds.get_mean_knn(inp)

class UTKFaceDataset(Dataset):
    def __init__(self, transform=None):
        self.root_dir = "datasets_face\CelebAMask-HQ\CelebA-HQ-img"
        # self.root_dir = "datasets_face/UTKFace"
        self.transform = transform
        self.image_paths = [os.path.join(self.root_dir, img_name) for img_name in os.listdir(self.root_dir)]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        

        img_name = os.path.basename(img_path)
 
        return image, torch.tensor(int(2), dtype = torch.float32)
        # age, *_ = img_name.split('_')
        # return image, torch.tensor(int(age), dtype = torch.float32)
    

# def main():
#     data_transforms = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     vg = vgg16SimpleDensity()
#     dataset = UTKFaceDataset(transform=data_transforms)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=10)

#     for inps, labels in dataloader:
#         print(vg(inps))
#         break

# if __name__ == '__main__':
#     main()
