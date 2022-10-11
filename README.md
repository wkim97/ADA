# Attentive Diversity Attack (ADA)

Official PyTorch implementation of [Diverse Generative Perturbations on Attention Space for Transferable Adversarial Attacks](https://arxiv.org/abs/2208.05650)
(ICIP 2022).

## Getting Started

### Installation
```shell
git clone https://github.com/wkim97/ADA.git
conda install --file requirements.txt
```

### Preparing Datasets
Download the training and evaluation datasets 
[here](https://drive.google.com/file/d/1aZyLsn81-MQP6Zc2jMj1mNyEkAjrR61X/view?usp=sharing) 
and unzip the file under `ADA/data`.

The official evaluation dataset can also be downloaded from the 
[NIPS 2017 adversarial attack competition](https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set).

### Pretrained Weights
You can download the pretrained weights 
[here](https://drive.google.com/file/d/1Bx30l3fQbgrsrhmYUBI_1F5r21_GzEzt/view?usp=sharing) 
and unzip the file under `ADA/weights`.
 
## Training
```shell
python train.py --surrogate inception_v3 --target_layer Mixed_7c --save_dir ./weights --save_name default
```

## Testing 
```shell 
python test.py --surrogate inception_v3 --target_layer Mixed_7c --load_dir ./weights --load_name default
```

## Acknowledgement
Some parts of the code are borrowed from 
[grad-cam-pytorch](https://github.com/kazuto1011/grad-cam-pytorch)
and from [DSGAN](https://github.com/maga33/DSGAN).

## Citation 
If you find this code useful for your research, please consider citing our paper
````BibTex
@article{kim2022diverse,
  title={Diverse Generative Adversarial Perturbations on Attention Space for Transferable Adversarial Attacks},
  author={Kim, Woo Jae and Hong, Seunghoon and Yoon, Sung-Eui},
  journal={arXiv preprint arXiv:2208.05650},
  year={2022}
}
````