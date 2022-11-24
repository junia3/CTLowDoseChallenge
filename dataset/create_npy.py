import h5py
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

if __name__ == '__main__':
    mode = "train"
    os.makedirs(f"{mode}_dataset/", exist_ok=True)
    cnt = 0
    for file in os.listdir(f"{mode}"):
        ndct = np.array(h5py.File(f'{mode}/{file}')['f_nd'])
        qdct = np.array(h5py.File(f'{mode}/{file}')['f_qd'])

        assert ndct.shape[0] == qdct.shape[0]

        for i in tqdm(range(ndct.shape[0])):
            with open(f'{mode}_dataset/{str(cnt).zfill(4)}.npy', 'wb') as f:
                np.save(f, qdct[i])
                np.save(f, ndct[i])

            cnt += 1

    mode = "validation"
    os.makedirs(f"{mode}_dataset/", exist_ok=True)
    cnt = 0
    for file in os.listdir(f"{mode}"):
        ndct = np.array(h5py.File(f'{mode}/{file}')['f_nd'])
        qdct = np.array(h5py.File(f'{mode}/{file}')['f_qd'])

        assert ndct.shape[0] == qdct.shape[0]

        for i in tqdm(range(ndct.shape[0])):
            with open(f'{mode}_dataset/{str(cnt).zfill(4)}.npy', 'wb') as f:
                np.save(f, qdct[i])
                np.save(f, ndct[i])

            cnt += 1
        

    mode = "test"
    os.makedirs(f"{mode}_dataset/", exist_ok=True)
    cnt = 0
    for file in os.listdir(f"{mode}"):
        ndct = np.array(h5py.File(f'{mode}/{file}')['f_nd'])
        qdct = np.array(h5py.File(f'{mode}/{file}')['f_qd'])

        assert ndct.shape[0] == qdct.shape[0]

        for i in tqdm(range(ndct.shape[0])):
            with open(f'{mode}_dataset/{str(cnt).zfill(4)}.npy', 'wb') as f:
                np.save(f, qdct[i])
                np.save(f, ndct[i])

            cnt += 1
        
