import banana_dev as banana
import os

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
model_key = os.getenv("MODEL_KEY")

model_inputs = {"file_path":"test_images/n01440764_tench.jpeg"} # anything you want to send to your model

out = banana.run(api_key, model_key, model_inputs)

