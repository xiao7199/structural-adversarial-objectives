####  Structural Adversarial Objectives


This repository contains the code for[Structural Adversarial Objectives For Self-Supervised Representation Learning](https://arxiv.org/abs/2310.00357))


## Running Code
This repository requires `pytorch > 2.0.0`.

We provide a sample running script for CIFAR-10/100 experiments in `srun.sh`. You can use `bash srun.sh` to launch an experiment with the default hyper-parameters for the CIFAR-10 dataset.

Before running the code, please replace `data_path` in `srun.sh` with the local folder path where you intend to store the CIFAR-10/100 dataset.

During training, we monitor the performance of the discriminator using online linear probing and report the testing accuracy every 5 epochs. A complete training takes 1000 epochs, but the linear probing performance of the discriminator typically converges by 500 epochs.

## Acknowledgment
We implemented our generator by referring to the code of https://github.com/ajbrock/BigGAN-PyTorch.


## Citation
If you find our work or codebase useful in your research, please consider giving a star ⭐ and a citation.
```
@article{zhang2023structural,
  title={Structural Adversarial Objectives for Self-Supervised Representation Learning},
  author={Zhang, Xiao and Maire, Michael},
  journal={arXiv preprint arXiv:2310.00357},
  year={2023}
}
```
