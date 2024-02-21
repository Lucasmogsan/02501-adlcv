### Conda installation
Install from [https://docs.anaconda.com/free/miniconda/](https://docs.anaconda.com/free/miniconda/).
Follow [https://www.hpc.dtu.dk/?page_id=3678](https://www.hpc.dtu.dk/?page_id=3678) for DTU HPC setup.

```bash
cd ~
conda create -n dtu-02501-adlcv python=3.8
conda activate dtu-02501-adlcv
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd 02501-adlcv
pip install -r requirements.txts
```