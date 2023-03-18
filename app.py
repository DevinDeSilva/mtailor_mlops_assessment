from model import *
from PIL import Image
import os
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model
    global preprocess_pipeline
    
    model = InferenceONNX("model_onnx/pytorch_model.onnx")
    preprocess_pipeline = PreprocessingPipeline()

def __run_a_test(path_img,result):
    global model
    global preprocess_pipeline
    
    image = Image.open(path_img)
    image = preprocess_pipeline.forward(image)
    
    # Run the model
    model_out = model.forward(image)
    
    return f"{path_img} :- {result == model_out}"

def __run_preset_test():
    results = {}
    
    results["n01440764_tench.jpeg"] = __run_a_test(os.path.join("test_images","n01440764_tench.jpeg"),0)
    results["n01667114_mud_turtle.JPEG"] = __run_a_test(os.path.join("test_images","n01667114_mud_turtle.JPEG"),0)
    
    return results
        
        

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global preprocess_pipeline

    # Parse Input
    
    if "test" not in model_inputs.keys():
        model_inputs["test"] = False
    
    if model_inputs["test"] == False:
        image = Image.open(model_inputs["file_path"])
        image = preprocess_pipeline.forward(image)
        
        # Run the model
        result = model.forward(image)

        # Return the results as a dictionary
        return result
    else:
        return __run_preset_test()