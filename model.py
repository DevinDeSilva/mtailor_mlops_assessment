import onnxruntime
import numpy as np
import torch
from torchvision import transforms

class PreprocessingPipeline(object):
    
    def __init__(self) -> None:
        self.transform_pipeline = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def forward(self,x):
        x = self.transform_pipeline(x).unsqueeze(0) 
        return x
    


class InferenceONNX(object):
    def __init__(self,model_path:str) -> None:
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        #load onnx
        self.ort_session = onnxruntime.InferenceSession(self.model_path)
    
    @staticmethod
    def to_numpy(tensor:torch.Tensor)->np.ndarray:
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
        
    def forward(self,x):
        ort_inputs = {self.ort_session.get_inputs()[0].name: InferenceONNX.to_numpy(x)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        
        return np.argmax(ort_outs[0])