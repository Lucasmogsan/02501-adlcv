```bash
cd ~
conda create -n dtu-02501-adlcv python=3.8
conda activate dtu-02501-adlcv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd 02501-adlcv
pip install -r requirements.txts
```


# HPC

1. Setting up hpc in .ssh and connect through ssh. Can be done through VSCode or the terminal.
```bash
ssh hpc
```

2. Use GPU node interactively (see [https://www.hpc.dtu.dk/?page_id=2129](https://www.hpc.dtu.dk/?page_id=2129)). Example:
```bash
voltash
``` 

3. Activate conda environment
```bash
conda activate dtu-02501-adlcv
```

4. Install requirements (only first time)
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

5. Update to latest git version
```bash
git pull origin main 
```


## Conda installation
Install from [https://docs.anaconda.com/free/miniconda/](https://docs.anaconda.com/free/miniconda/).
Follow [https://www.hpc.dtu.dk/?page_id=3678](https://www.hpc.dtu.dk/?page_id=3678) for DTU HPC setup.