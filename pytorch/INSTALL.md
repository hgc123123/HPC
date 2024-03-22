# Create conda environment
```
module load ananconda3
conda create -n pytorch python==3.8.5
source activate pytorch
```

#Install pytorch by pip or conda
```
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
https://pytorch.org/

or

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
https://discuss.pytorch.org/t/cannot-for-the-life-of-me-get-pytorch-and-cuda-to-install-work/197088/9
```


