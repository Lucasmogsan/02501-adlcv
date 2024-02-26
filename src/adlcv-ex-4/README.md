# Diffusion Guidance project

## Installation
* `conda create -n ddpm python=3.10.10`
* `conda activate ddpm`
* `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
* `pip install -r requirements.txt`


## Logging
* The code uses tensorboard to log the train loss. Use the command `tensorboard --logdir=runs` to observe the training loss.
* To train a classifier-free guidance model run python ddmp_train.py --cfg


## Run
```bash
cd 02501-adlcv/src/adlcv-ex-4

python classifier_train.py
python classifier_eval.py

python sample_cg.py

python ddpm_train.py

python ddpm_train.py --cfg
python sample_cFg.py

python ddpm_eval.py

```