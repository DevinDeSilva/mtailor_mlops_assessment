from model import *
from PIL import Image
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global preprocess_pipeline
    
    model = InferenceONNX("model_onnx/pytorch_model.onnx")
    preprocess_pipeline = PreprocessingPipeline()
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global preprocess_pipeline

    # Parse Input
    print(model_inputs)
    image = Image.open(model_inputs["file_path"])
    image = preprocess_pipeline.forward(image)
    
    # Run the model
    result = model.forward(image)

    # Return the results as a dictionary
    return result