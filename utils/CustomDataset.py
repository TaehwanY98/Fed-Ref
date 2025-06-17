import torch
import numpy as np
import os
import h5py
import SimpleITK as sitk
from skimage.transform import resize
from skimage.util import random_noise
from functools import reduce
from torchvision.transforms import ToTensor, Normalize, Compose


class Fets2022(object):
    def __init__(self, data_dir, norm=True) -> None:
        self.dir = data_dir
        self.patients = [patient for patient in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, patient))]
        self.file_names = [[x for x in os.listdir(os.path.join(data_dir,patient)) if os.path.isfile(os.path.join(data_dir, patient, x))] for patient in self.patients]
        self.width = 240
        self.height = 240
        self.depth = 155
        self.normalization = norm
    
    def __getitem__(self, i):
        directory = self.dir
        patient = self.patients[i]
        files = self.file_names[i]
        for file in files:
            if "flair" in file:
                xpath = os.path.join(directory, patient, file)
            if "seg" in file:
                ypath = os.path.join(directory, patient, file)
                
        x = self.OpenFile(xpath)        
        y = self.OpenFile(ypath)
        y[y==4] = 3
        
        ret={
            "x" : torch.Tensor(x),
            'y' : torch.Tensor(y)
        }
        return ret
    
    def OpenFile(self, path):
        img = sitk.ReadImage(path, sitk.sitkFloat32)
        return sitk.GetArrayFromImage(img)
            
    def __len__(self) :
        return len(self.patients)

class Fets2022Noise(object):
    def __init__(self, data_dir, noise=77) -> None:
        self.dir = data_dir
        self.patients = [patient for patient in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, patient))]
        self.file_names = [[x for x in os.listdir(os.path.join(data_dir,patient)) if os.path.isfile(os.path.join(data_dir, patient, x))] for patient in self.patients]
        self.width = 240
        self.height = 240
        self.depth = 155
        self.noise = noise
    
    def __getitem__(self, i):
        directory = self.dir
        patient = self.patients[i]
        files = self.file_names[i]
        for file in files:
            if "flair" in file:
                xpath = os.path.join(directory, patient, file)
            if "seg" in file:
                ypath = os.path.join(directory, patient, file)
                
        x = self.OpenFile(xpath, True)        
        y = self.OpenFile(ypath, 0)
        y[y==4] = 3
        
        ret={
            "x" : torch.Tensor(x),
            'y' : torch.Tensor(y)
        }
        return ret
    
    def OpenFile(self, path, noise:bool):
        img = sitk.ReadImage(path, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img)
        if noise:
            img = random_noise(img)
        return img
            
    def __len__(self) :
        return len(self.patients)
    

class BRATS(object):
    def __init__(self, data_dir, norm=True) -> None:
        self.dir = data_dir
        self.patients = [patient for patient in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, patient))]
        self.file_names = [sorted([x for x in os.listdir(os.path.join(data_dir,patient)) if os.path.isfile(os.path.join(data_dir, patient, x))], key= lambda s: int(s.split("_")[-1].replace(".h5", ""))) for patient in self.patients]
        self.width = 240
        self.height = 240
        self.depth = 155
        self.normalization = norm
    
    def __getitem__(self, i):
        directory = self.dir
        patient = self.patients[i]
        files = self.file_names[i]
        X = np.zeros((self.depth, self.width, self.height, 4), dtype=np.float32)
        Y = np.zeros((self.depth, self.width, self.height, 4), dtype=np.int32)
        for indx, file in enumerate(files):
            path = os.path.join(directory, patient, file)    
            x, y = self.OpenFile(path)        
            X[indx] = x
            Y[indx,..., 1:] = y
        Y = torch.Tensor(Y).permute(0,3,2,1)
        background, _ = Y.max(dim=1)
        background = background==0
        Y[:,0,...] = background.type(torch.int32)
        ret={
            "x" : torch.Tensor(X).permute(0,3,2,1)[:,0,...],
            'y' : Y.argmax(dim=1)
        }
        return ret
    
    def OpenFile(self, path):

        with h5py.File(path) as f:
           x = np.array(f["image"])
           y = np.array(f["mask"])
           return x, y
            
    def __len__(self) :
        return len(self.patients)


class BRATSNoise(object):
    def __init__(self, data_dir, norm=True) -> None:
        self.dir = data_dir
        self.patients = [patient for patient in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, patient))]
        self.file_names = [sorted([x for x in os.listdir(os.path.join(data_dir,patient)) if os.path.isfile(os.path.join(data_dir, patient, x))], key= lambda s: int(s.split("_")[-1].replace(".h5", ""))) for patient in self.patients]
        self.width = 240
        self.height = 240
        self.depth = 155
        self.normalization = norm
    
    def __getitem__(self, i):
        directory = self.dir
        patient = self.patients[i]
        files = self.file_names[i]
        X = np.zeros((self.depth, self.width, self.height, 4), dtype=np.float32)
        Y = np.zeros((self.depth, self.width, self.height, 4), dtype=np.int32)
        for indx, file in enumerate(files):
            path = os.path.join(directory, patient, file)    
            x, y = self.OpenFile(path)        
            X[indx] = x
            Y[indx,..., 1:] = y
        Y = torch.Tensor(Y).permute(0,3,2,1)
        background, _ = Y.max(dim=1)
        background = background==0
        Y[:,0,...] = background.type(torch.int32)
        ret={
            "x" : torch.Tensor(X).permute(0,3,2,1)[:,0,...],
            'y' : Y.argmax(dim=1)
        }
        return ret
    
    def OpenFile(self, path):

        with h5py.File(path) as f:
           x = np.array(f["image"])
           x = random_noise(x)
           y = np.array(f["mask"])
           return x, y
            
    def __len__(self) :
        return len(self.patients)

class OCTDL(object):
    def __init__(self, data_dir, norm=True) -> None:
        self.dir = data_dir
        self.patients = ['NO','AMD', 'DME','ERM','RAO','RVO','VID']
        self.file_names = [[os.path.join(data_dir, patient, x) for x in os.listdir(os.path.join(data_dir,patient)) if os.path.isfile(os.path.join(data_dir, patient, x))] for patient in self.patients]
        self.flatten_names = reduce(lambda x,y: list(x)+list(y), self.file_names)
        self.label_weight = torch.Tensor([1- len(file)/len(self.flatten_names) for file in self.file_names])
    def __getitem__(self, i):
        
        path = self.flatten_names[i]

        x = self.OpenFile(path)
        x = resize(x, (240, 540))
        for indx, p in enumerate(self.patients):
            if p in path:
                y= torch.Tensor(np.array([indx]))
                break

        ret={
            "x" : torch.Tensor(x),
            'y' : y,
            # 'name': path.split("/")[-1],
            # 'path': path
        }
        return ret
    
    def OpenFile(self, path):
        img = sitk.ReadImage(path, sitk.sitkFloat32)
        return sitk.GetArrayFromImage(img)

    def __len__(self) :
        return len(self.flatten_names)
    
class OCTDLNoise(object):
    def __init__(self, data_dir, norm=True) -> None:
        self.dir = data_dir
        self.patients = ['NO','AMD', 'DME','ERM','RAO','RVO','VID']
        self.file_names = [[os.path.join(data_dir, patient, x) for x in os.listdir(os.path.join(data_dir,patient)) if os.path.isfile(os.path.join(data_dir, patient, x))] for patient in self.patients]
        self.flatten_names = reduce(lambda x,y: list(x)+list(y), self.file_names)
        self.label_weight = torch.Tensor([1- len(file)/len(self.flatten_names) for file in self.file_names])
    def __getitem__(self, i):
        
        path = self.flatten_names[i]

        x = self.OpenFile(path, True)
        x = resize(x, (240, 540))
        for indx, p in enumerate(self.patients):
            if p in path:
                y= torch.Tensor(np.array([indx]))
                break

        ret={
            "x" : torch.Tensor(x),
            'y' : y,
            # 'name': path.split("/")[-1],
            # 'path': path
        }
        return ret
    
    def OpenFile(self, path, noise:bool):
        img = sitk.ReadImage(path, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(img)
        if noise:
            img = random_noise(img)
        return img

    def __len__(self) :
        return len(self.flatten_names)
    
class MNIST(object):
    def __init__(self, samples, norm=True) -> None:
        self.samples = samples
        self.norm = Compose([ToTensor(), Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])
    def __getitem__(self, i):
        data = self.samples.data.to_numpy()[i].astype("float32")
        data = torch.Tensor(data)
        
        ret={
            'x' : self.norm.transforms(torch.stack([data,data,data]).view(28,28,3)),
            'y' : int(self.samples.target.to_numpy()[i])
        }
        return ret

    def __len__(self) :
        return len(self.samples.data)