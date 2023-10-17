
import dataloader as data
from torch.utils.data import  Dataset,random_split,DataLoader
import torch.nn as nn
import utile
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
import tqdm
for i in tqdm.tqdm(range(100)):
    print(i)