import os
import torch.onnx

from pytorch_model import *

#config file
config = {
    "device":"cpu",
}

# set device as "GPU" if availble 
if config['device'] == 'cpu':
    device = "cpu"
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

def init_model(device,weight_path:str="model_weights/pytorch_model_weights.pth")->Classifier:
    """
    Initialize the model with pretrained weights.
    """
    mtailor = Classifier(BasicBlock, [2, 2, 2, 2])
    mtailor.load_state_dict(torch.load(weight_path))
    mtailor.to(device)
    return mtailor

def create_onnx_model(torch_model,batch_size=32,save_dir="model_onnx",save_name="pytorch_model.onnx",device=device):
    os.makedirs(save_dir,exist_ok=True)
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True).to(device)
    torch_model.eval()
    torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  f"{save_dir}/{save_name}",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
    


if __name__ == "__main__":
    mtailor = init_model(device)
    create_onnx_model(mtailor,device=device)
    print("Created")