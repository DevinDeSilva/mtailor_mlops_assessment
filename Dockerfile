# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

# Adding model.py file 
ADD model.py .

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .

RUN python3 download.py


# Add your custom app code, init() and inference()
ADD app.py .
ADD model_onnx/pytorch_model.onnx model_onnx/pytorch_model.onnx
ADD test_images/n01440764_tench.jpeg test_images/n01440764_tench.jpeg
ADD test_images/n01667114_mud_turtle.JPEG test_images/n01667114_mud_turtle.JPEG

EXPOSE 8000

CMD python3 -u server.py