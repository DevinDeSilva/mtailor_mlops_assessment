# Download the model weights from https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1
from urllib import request
import os

def download_model(download_path="model_weights"):
    os.makedirs(download_path,exist_ok=True)
    request.urlretrieve("https://www.dropbox.com/s/b7641ryzmkceoc9/pytorch_model_weights.pth?dl=1",f"{download_path}/pytorch_model_weights.pth")
    
if __name__ == "__main__":
    download_model()