import os
import os.path as osp
import gdown
import torch
import numpy as np
from torch_sparse import SparseTensor

def download_dataset(url, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    file_id = url.split('/')[-2]
    download_url = f'https://drive.google.com/uc?id={file_id}'
    
    # output_path = os.path.join(output_directory, 'dataset.zip')
    try:
        output_path = gdown.download(download_url, output_directory + osp.sep, quiet=False)
    except:
        print('It looks like Gdown encounters errors, or Google drive exhibits download '
              'number limits during downloading. However, You still can download the file '
              'from a web browser by using this link:\n\n'
              f'{url}\n\n'
              'Then unzip this file (only if it is a Zip file), and manually put all the content to '
              f'{output_directory}.')
        exit(0)

    
    # Unzip the dataset if necessary
    if output_path.endswith('.zip'):
        import zipfile
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(output_directory)
        os.remove(output_path)  # Remove the zip file after extraction
    