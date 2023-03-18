import onnx
import onnxruntime
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_model import *
from convert_to_onnx import init_model

#config file
config = {
    "device":"cpu",
}

# set device as "GPU" if availble 
if config['device'] == 'cpu':
    device = "cpu"
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
def preprocessing_pipeline(x:Image)->torch.Tensor:
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    x = transform_pipeline(x).unsqueeze(0) 
    return x

def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_onnx(file_path = "test_images/n01667114_mud_turtle.jpeg",y_true=35,
              onnx_model_path = "model_onnx/pytorch_model.onnx"):
    #load onnx
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    #load image
    orig_image = Image.open(file_path)
    img = preprocessing_pipeline(orig_image).to(device)

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # compare ONNX Runtime and PyTorch results
    print("File name:- ",file_path,"Model Output:-:",np.argmax(ort_outs[0]),"True Output:- ",y_true)
    assert np.argmax(ort_outs[0]) == y_true

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

if __name__ == "__main__":
    test_onnx(file_path="test_images/n01667114_mud_turtle.JPEG",y_true = 35)
    print("  \n")
    test_onnx(file_path="test_images/n01440764_tench.jpeg",y_true = 0)