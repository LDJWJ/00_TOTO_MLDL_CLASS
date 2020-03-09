import h5py
import os
import numpy as np

def isGroup(obj):
    if isinstance(obj, h5py.Group):
        return True
    return False

def isDataset(obj):
    if isinstance(obj, h5py.Dataset):
        return True

def getDatasetFromGroup(datasets, obj):
    
    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetFromGroup(datasets, x)
    else:
        datasets.append(obj)

def getWeightsForLayer(layerName, fileName):
    weights = []
    with h5py.File(fileName, mode='r') as f:
        for key in f:
            if layerName in key:
                obj = f[key]
                datasets = []
                getDatasetFromGroup(datasets, obj)
                
                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
    return weights

weights = getWeightsForLayer("model_weights", "./model/21-0.1165.hdf5")
print(weights)