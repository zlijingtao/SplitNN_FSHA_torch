# Unleashing the Tiger (pytorch version)
This repository is a simple pytorch version of the original repository in tensorflow-2. Please give credit to the original repository [github](https://github.com/pasquini-dario/SplitNN_FSHA) and Paper: [arxiv](https://arxiv.org/abs/2012.02670) that is recently accepted by CCS '21.

## Code:

*  *FSHA_torch.py*: It implements the attack and a single-user version of split learning.
* *architectures_torch.py*: It contains the main network architectures we used in the paper.
* *datasets_torch.py*: It contains utility to load and parse datasets.

## Proof Of Concepts 🐯:

We report a set of *jupyter notebooks* that act as brief tutorial for the code and replicate the experiments in the paper. Those are:

* *FSHA.ipynb*: It implements the standard Feature-space hijacking attack on the MNIST dataset.


## Tensorflow/Pytorch Version Comparison:
* *main.py*: Same as *FSHA.ipynb*.
* *main_tf.py*: The original Tensorflow2 version of *FSHA.ipynb*.

The pytorch version runs slower and has a slighly slower convergence performance.

* Migration of other funcitons in progress.....

## Cite the work:
```
@misc{pasquini2020unleashing,
      title={Unleashing the Tiger: Inference Attacks on Split Learning},
      author={Dario Pasquini and Giuseppe Ateniese and Massimo Bernaschi}, 
      year={2020},
      eprint={2012.02670},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```
