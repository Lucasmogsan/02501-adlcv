```bash
cd ~
conda create -n dtu-02501-adlcv python=3.8
conda activate dtu-02501-adlcv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia # for ex4
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # for ex5
cd 02501-adlcv
pip install -r requirements.txts
```
got some errors (circular import). solve by:
```bash
pip install --force-reinstall gdown
```

HPC if cuda is not working...
```bash
module avail | grep cuda
module avail | grep torch
module load >cuda or torch version you want (related to python 8 if that's what you're running)<
```


# HPC

1. Setting up hpc in .ssh and connect through ssh. Can be done through VSCode or the terminal.
```bash
ssh hpc
```

You can see the hpc directory from connect to server in Files: sftp://s223093@transfer.gbar.dtu.dk/


2. Use GPU node interactively (see [https://www.hpc.dtu.dk/?page_id=2129](https://www.hpc.dtu.dk/?page_id=2129)). Example:
```bash
voltash
sxm2sh
a100sh
exit
``` 

3. Activate conda environment
```bash
conda activate dtu-02501-adlcv
```

4. Install requirements (only first time)
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

5. Update to latest git version
```bash
git pull origin main
```

6. Run code from ex-folder:
```bash
cd 02501-adlcv/src/adlcv-ex-
python >code-to-run<
```



## Conda installation
Install from [https://docs.anaconda.com/free/miniconda/](https://docs.anaconda.com/free/miniconda/).
Follow [https://www.hpc.dtu.dk/?page_id=3678](https://www.hpc.dtu.dk/?page_id=3678) for DTU HPC setup.