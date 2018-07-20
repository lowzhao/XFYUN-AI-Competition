# XFYUN-AI-Competition - Chinese Dialect Classification

## Competition Background
We are given 6 Chinese Dialect data which contain 6000++ `.pcm` raw file with label. Our task is to create a `inference.tar` file which will be run on a __docker container__ and will save the result into `result/result.txt`

Ours `inference.tar` contain :
  1. `inference.py` A python file which is will use the model trained from given dataset
  2. `inference.tar` A bash script that will handle our `pip imports` and some required library, and then run our `inference.py`
  3. `model.model` A trained Keras CNN model

## 
## Reference

