from model import *
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

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    result = model(prompt)

    # Return the results as a dictionary
    return result