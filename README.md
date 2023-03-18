## Deploy Classification Neural Network on Serverless GPU platform of Banana Dev

### Initialize

* First git clone this repository 
* Then install the python packages in requirements.txt using 

* create a python enviroment and activate the enviroment

```python
pip install -r requirements.txt
```

* The install pytorch using following command 
```python

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
Note I used this version to match the docker image.

* Now run 

```python
python convert_to_onnx.py
```

* and 

```python
python test_oonx.py 
```

for the first two deliverables

finally running

```python
python test_server.py  
```

should run the tests I create but there is an error which that gives a server 500 But due to time constraints was unable to debug.

## Links 
Github:- github.com/DevinDeSilva/mtailor_mlops_assessment/tree/d417d6174f6dda28938a604280f996348c76accd





convert_to_onnx.py:- www.loom.com/share/c90ef8b336d147eca39f33010da38cad



test_onnx :- www.loom.com/share/b6ea51f66b86478882afd5d172d02147



server files:- www.loom.com/share/c7c56dbedf4f4c7284a424a8740c9792



test_server.py :- www.loom.com/share/1a052c1ee470408394261f1bc4e8c161

