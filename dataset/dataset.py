from torch.utils.data import Dataset, Subset
import os
import urllib.request
import zipfile
import numpy as np
import torch
import warnings
import requests
import pickle
import subprocess
from tqdm import tqdm
import math
import warnings

class TEXTDataset(Dataset):
    def __init__(self, path, sequence_length=2048, cutoff_ratio = 1):
        self.sequence_length = sequence_length
        self.data_length = math.ceil(os.path.getsize(path) * cutoff_ratio / self.sequence_length)
        self.data = []
        with open(path, 'rb') as file:
            for _ in range(self.data_length):
                chunk = file.read(sequence_length)
                if len(chunk) == sequence_length:
                    self.data.append(np.frombuffer(chunk, dtype=np.uint8))
                else:
                    break
            
        self.data = np.array(self.data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long) 
    
# Define the Enwik9Dataset class
class Enwik9Dataset(TEXTDataset):
    '''
    This class is used to load the Enwik9 dataset.
    if the type is 'Enwik8', it will only load the first 10% of the dataset.
    if the type is 'Enwik9', it will load the whole dataset.
    if the type is 'Enwik9_without_Enwik8', it will load the last 90% of the dataset.
    '''
    def __init__(self, path, sequence_length=2048, type = 'Enwik8'):        
        if not os.path.exists(path):
            target_dir = os.path.dirname(path) or '.'
            os.makedirs(target_dir, exist_ok=True)
            # Downloading and extracting the dataset.
            print("Downloading and extracting Enwik9 dataset...")
            urllib.request.urlretrieve(
                'https://mattmahoney.net/dc/enwik9.zip',
                f'{path}.zip',
            )
            
            print("Download complete, extracting...")
            with zipfile.ZipFile(f'{path}.zip', 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            os.remove(f'{path}.zip')

        self.sequence_length = sequence_length
        self.data_length = math.ceil(os.path.getsize(path) / self.sequence_length)
        self.data = []
        
        if type == 'Enwik8':
            data_length_range = range(self.data_length // 10)
        elif type == 'Enwik9':
            data_length_range = range(self.data_length)
        elif type == 'Enwik9_without_Enwik8':
            data_length_range = range(self.data_length // 10, self.data_length)
        else:
            raise ValueError("type should be one of ['Enwik8', 'Enwik9', 'Enwik9_without_Enwik8']")
        
        with open(path, 'rb') as file:
            for _ in data_length_range:
                chunk = file.read(sequence_length)
                if len(chunk) == sequence_length:
                    self.data.append(np.frombuffer(chunk, dtype=np.uint8))
                else:
                    break
        self.data = np.array(self.data)

def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo)
    return data_dict

def download_and_extract(url, target_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024*1024*10  # 1 Kilobyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(target_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        # Using system unzip to extract files
        subprocess.run(['unzip', '-q', target_path, '-d', os.path.dirname(target_path)], check=True)
        os.remove(target_path)  # Remove the zip file after extraction

def get_dataset(dataset_name, databudget_ratio = None, databudget = None, data_dir = '../data'):
    '''
    args:
    dataset_name: string, the name of the dataset
    databudget: float, the ratio of the training dataset to the whole dataset
    data_dir: string, the parent directory of the dataset
    if True, defaultly will just load the training dataset
    
    return value:
    training_dataset: torch.utils.data.Subset, the training dataset
    validation_dataset: torch.utils.data.Subset, the validation dataset
    '''    
    if dataset_name == 'enwik8':
        dataset = Enwik9Dataset(os.path.join(data_dir, 'enwik9'), type='Enwik8')
        
        training_range = range(len(dataset) * 5 // 6)
        validation_range = range(len(dataset) * 5 // 6, len(dataset))
        
        training_dataset = Subset(dataset, training_range)
        validation_dataset = Subset(dataset, validation_range)
    elif dataset_name == 'enwik9':
        dataset = Enwik9Dataset(os.path.join(data_dir, 'enwik9'), type='Enwik9')
        
        training_range = range(len(dataset) * 5 // 6)
        validation_range = range(len(dataset) * 5 // 6, len(dataset))
        
        training_dataset = Subset(dataset, training_range)
        validation_dataset = Subset(dataset, validation_range)
    elif dataset_name == 'enwik9_without_enwik8':
        dataset = Enwik9Dataset(os.path.join(data_dir, 'enwik9'), type='Enwik9_without_Enwik8')
        
        training_range = range(len(dataset) * 5 // 6)
        validation_range = range(len(dataset) * 5 // 6, len(dataset))
        
        training_dataset = Subset(dataset, training_range)
        validation_dataset = Subset(dataset, validation_range)
    elif dataset_name == 'shakespeare':
        dataset = TEXTDataset(os.path.join(data_dir, 'shakespeare-dataset.txt'))
        
        training_range = range(len(dataset) * 5 // 6)
        validation_range = range(len(dataset) * 5 // 6, len(dataset))
        
        training_dataset = Subset(dataset, training_range)
        validation_dataset = Subset(dataset, validation_range)
    elif dataset_name == 'code':
        dataset = TEXTDataset(os.path.join(data_dir, 'code.txt'))
        
        training_range = range(len(dataset) * 5 // 6)
        validation_range = range(len(dataset) * 5 // 6, len(dataset))
        
        training_dataset = Subset(dataset, training_range)
        validation_dataset = Subset(dataset, validation_range)
    elif dataset_name == 'math':
        training_dataset = TEXTDataset(os.path.join(data_dir, 'math_dataset/train.txt'))
        validation_dataset = TEXTDataset(os.path.join(data_dir, 'math_dataset/test.txt'))
    else:
        raise ValueError(f"dataset_name should be one of ['enwik8', 'enwik9', 'enwik9_without_enwik8', 'shakespeare', 'code', 'math']")

    num_all = len(training_dataset)
    
    if databudget_ratio is None and databudget is None:
        databudget_ratio = 1
        num_tr = int(num_all * databudget_ratio)
    elif databudget_ratio is None and databudget is not None:
        if databudget > num_all:
            num_tr = num_all
            warnings.warn("databudget is larger than the whole dataset, set databudget to the whole dataset")
        else:
            num_tr = int(databudget)
    elif databudget_ratio is not None and databudget is None:
        num_tr = int(num_all * databudget_ratio)
    elif databudget_ratio is not None and databudget is not None:
        raise ValueError("databudget_ratio and databudget conflict")

    training_idx_range = list(range(num_tr))
    
    training_dataset = Subset(training_dataset, training_idx_range)
    
    return training_dataset, validation_dataset
        
    
    
    
