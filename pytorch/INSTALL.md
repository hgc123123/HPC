# Create conda environment
```
module load ananconda3
conda create -n pytorch python==3.8.5
source activate pytorch
```

#Install pytorch by pip
```
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```


