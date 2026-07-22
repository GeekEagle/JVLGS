# JVLGS

📄 **Paper:** [The Visual Computer](https://link.springer.com/article/10.1007/s00371-026-04591-y?status=info&saved-doi=10.1007%2Fs00371-026-04591-y)  
## Start
```
conda create -n jvlgs python=3.8
conda activate jvlgs
```
install [2.4.1 pytorch, torchvision, and torchaudio](https://pytorch.org/get-started/previous-versions/) 

install [mmcv-full](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 
```
pip install mmcv-full
pip install mmsegmentation
```
The backbone is pre-trained on the COD10K dataset.   

[Dataset & Pretrained Backbone & Model Link](https://drive.google.com/drive/folders/1EuQyTL3lETJLGCM31Kh4IYmLsLcPoMQn?usp=sharing)
Our proposed model is trained on the IGS-Few and SimGas datasets, whose training files are available on the Drive link.

Please put the pretrained model into the ./pretrain folder, and please change the dataset_path.py to your dataset path.

## Train/Test SimGas dataset (k-fold training):
```
python kfold_train.py
python kfold_test.py
```
## Train/Test IGS-Few and other datasets (Few-shot or supervised):
```
python normal_train.py
python normal_test.py
```

## Citing 

