# VLJGS

## Start

conda create -n vljgs python=3.8

conda activate vljgs

install [2.4.1 pytorch, torchvision, and torchaudio](https://pytorch.org/get-started/previous-versions/) 

install [mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 

pip install mmcv-full
pip install mmsegmentation


The backbone is pre-trained on the COD10K dataset.   

[Dataset & Pretrained Backbone Link](https://drive.google.com/drive/folders/1EuQyTL3lETJLGCM31Kh4IYmLsLcPoMQn?usp=sharing)

Please put the pretrain model into the ./pretrain folder, and please change the dataset_path.py to your dataset path.

## Train/Test SimGas dataset:

   python kfold_train.py

   python kfold_test.py

## Train/Test GasVid/IGS-Few dataset:

  python normal_train.py
  
  python normal_test.py


## Citing 

