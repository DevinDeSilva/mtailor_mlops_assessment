## Deploy Classification Neural Network on Serverless GPU platform of Banana Dev

### Initialize

* First git clone this repository 
* Then install the python packages in requirements.txt using 

```python
pip install -r requirements.txt
```

* The install pytorch using following command 
```python

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Note I used this version to match the docker image.

