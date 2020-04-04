#!/bin/bash

# chmod a+x /bin/HCopy
pip install librosa
pip install keras 
pip install 'joblib==0.11' --force-reinstall
apt-get install libav-tools

# cd  ./inference && chmod a+x ./inference.py && python inference.py
cd  /inference/ && chmod a+x ./inference.py && python inference.py

# use ./models/model9.model,modify in /inference/2.inferenceLSTM/inference.py
# cd /inference/2.inferenceLSTM/ && chmod a+x  run.sh && ./run.sh
