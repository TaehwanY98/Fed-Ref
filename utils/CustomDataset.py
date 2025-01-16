import torch
import os
import SimpleITK as sitk

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
    

class cDataset(object):
    def __init__(self,X,Y) -> None:
        super(cDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, i):
        ret={'x': self.X[i,...],
             'y': self.Y[i,...]}
        return ret
    
    def __len__(self) :
        return len(self.X)