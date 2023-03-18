import banana_dev as banana
import os
from time import time

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
model_key = os.getenv("MODEL_KEY")

def run_request(model_inputs): # anything you want to send to your model
    start_time = time()
    out = banana.run(api_key, model_key, model_inputs)
    end_time = time()
    print(out,f"Time taken :- {end_time-start_time} s")
    
def run_tests():
    run_request({"test":True})
    
def run_single_request(file_path):
    run_request({"file_path":file_path})
    
    
if __name__ == "__main__":
    run_single_request("test_images/n01667114_mud_turtle.JPEG")
    run_tests()

