## The Official PyTorch Implementation of "Brain-like Variatonal Inference" ([NeurIPS 2025 Paper](https://openreview.net/forum?id=573IcLusXq))

Welcome to the *"Brain-like Variational Inference"* codebase!

## Introduction

Variational free energy (F) is the same thing as negative ELBO from machine learning. Why do we care? Because F minimization unifies popular generative models like VAEs with major cornerstones of theoretical neuroscience like Sparse Coding and Predictive Coding:

![Model Tree](./media/model_tree.png)

Building on this unification potential, we introduced **FOND** (*Free energy Online Natural-gradient Dynamics*): a framework for deriving brain-like adaptive iterative inference algorithms from first principles.

We then applied the FOND framework to derive a family of iterative VAE models, including the spiking iterative Poisson VAE (**iP-VAE**). This repository provides the implementation code for these iterative VAEs.

<p>
Before diving into the code, take a quick detour to watch an iP-VAE neuron in action, where we reproduce the classic Hubel & Wiesel bar of light experiment:
🎥 <a href="https://www.youtube.com/watch?v=-4K49zanvAA" target="_blank">Watch the video with sound</a>
</p>

<a href="https://www.youtube.com/watch?v=-4K49zanvAA" target="_blank">
  <img src="https://img.youtube.com/vi/-4K49zanvAA/hqdefault.jpg" alt="Watch the video"/>
</a>

To learn more, check out:
- Research paper: [https://openreview.net/forum?id=573IcLusXq](https://openreview.net/forum?id=573IcLusXq)
- X summary thread: [https://x.com/hadivafaii/status/1924344415063294287](https://x.com/hadivafaii/status/1924344415063294287)
- Talk: [https://www.youtube.com/watch?v=fTg-S81Ymto](https://www.youtube.com/watch?v=fTg-S81Ymto)


## 1. Code Structure

- **`./main/`**: Full architecture and training code for the iterative VAE models, including iP-VAE and iG-VAE.
- **`./base/`**: Core functionality including distributions, optimization, and dataset handling.
- **`./analysis/`**: Data analysis and result generation code.
- **`./scripts/`**: Model fitting scripts (examples below).

### Stand-alone PyTorch Lightning Implementation

We also provide a minimal PyTorch Lightning implementation of the iP-VAE, stripped down to its essential components. This serves as an excellent starting point for understanding the model. Check it out:

<a target="_blank" href="https://colab.research.google.com/drive/1LeHEuUHSXkgdcuqnwE7eaSby2fk0FD_U?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>



## 2. Training a Model

To train a model, run:

```bash
cd scripts/
./fit_model.sh <device> <dataset> <model> [additional args]
```

### Arguments:
- **`<device>`**: `int`, CUDA device index.
- **`<dataset>`**: `str`, choices = `{'vH16-wht', 'MNIST', ...}`.
- **`<model>`**: `str`, choices = `{'poisson', 'gaussian'}`.

Additional arguments can be passed to customize the training process. For example:

Key parameters include:
- **`t_train`**: Number of inference iterations (default: 16)
- **`n_iters_outer`**: Number of repeats of the outer loop (default: 1)
  - Controls gradient accumulation cycles. When > 1, implements truncated backpropagation through time by performing multiple cycles of "run ```t_train``` inference iterations, accumulate gradients, then update weights". Higher values allow longer effective sequence training while managing memory constraints. In the paper we only use the default value of 1.
- **`n_iters_inner`**: Number of gradient updates during inference (default: 1)
  - When > 1, the KL regularizaiton term kicks in (i.e., the "leak" term in iP-VAE).
- **`beta_outer`**: Beta used during learning (default: 16.0)
  - This value is used when computing the loss for weight update.
- **`beta_inner`**: Beta used during inference (default: 1.0)
  - This value is used only for the inner loop updates, therefore it does not have any effects when ```n_iters_inner = 1```.
- **`n_latents`**: Dimensionality of the latent space (default: 512)

See `./main/config.py` for all available configuration options.

To reproduce Figure 3 from the paper, train models using the following configurations, corresponding to iP-VAE, iG-VAE, and iG<sub>relu</sub>-VAE, respectively:

```bash
./fit_model.sh 0 'vH16-wht' 'poisson' --t_train 16 --n_latents 512 --beta_outer 24.0
./fit_model.sh 0 'vH16-wht' 'gaussian' --t_train 16 --n_latents 512 --beta_outer 8.0
./fit_model.sh 0 'vH16-wht' 'gaussian' --t_train 16 --n_latents 512 --beta_outer 8.0 --latent_act 'relu'
```

## 3. Notebook to Generate Results

- **`results.ipynb`**: Generates figures and analyses from the paper.
- **`load_models.ipynb`**: Visualizes trained models and their features.
- **`hubel_wiesel.ipynb`**: Reproduce the [classic Hubel & Wiesel bar of light experiment](https://www.youtube.com/watch?v=OGxVfKJqX5E) on a model "neuron".

## 4. Model Checkpoints and Data

### Checkpoints

We provide model checkpoints trained on whitened 16 x 16 patches extracted from the van Hateren dataset (vH16-wht). These are the same models you would get from running the scripts above, and they are located in **`./checkpoints/`** and can be loaded/visualized using **`load_models.ipynb`**. If additional model checkpoints would be helpful, feel free to reach out.

### Data

Download the processed datasets from the following links:

- Complete folder: [Drive Link](https://drive.google.com/drive/folders/1mCrsYtxcbNODcCTCLdaTi5v8yN_n5AMA?usp=sharing).
- Or individual datasets:
    1. [van Hateren](https://drive.google.com/drive/folders/1zaQPZm-8LhRXA24wMj4JeJf3s7Z0iIkM?usp=sharing).
    2. [MNIST](https://drive.google.com/drive/folders/1WQVqoUU1vbNTs4fd5jgA3zZR1j_XN3cC?usp=sharing).

Place the downloaded data under **`~/Datasets/`** with the following structure:

1. `~/Datasets/DOVES/vH16`
2. `~/Datasets/MNIST/processed`

For details, see the ```make_dataset()``` function in **`./base/dataset.py`**.

## 5. Citation

If you use our code in your research, please cite our paper:

```bibtex
@inproceedings{
  vafaii2025brainlike,
  title={Brain-like Variational Inference},
  author={Hadi Vafaii and Dekel Galor and Jacob L. Yates},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=573IcLusXq}
}
```

## 6. Contact

- For code-related questions, please open an issue in this repository.
- For paper-related questions, contact me at [vafaii@berkeley.edu](mailto:vafaii@berkeley.edu).
