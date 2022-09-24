# Rotated MNIST with Continuously Indexed Domain Adaptation (CIDA)

This is a re-implementation of the continuously indexed domain adaptation (CIDA) approach on the rotated MNIST dataset.

This experiement is based on and referenced from the research paper titled "[Continuously Indexed Domain Adaptation](https://github.com/hehaodele/CIDA/)" by authors Hao Wang and Hao He.

The purpose of this re-implementation is strictly for educational purposes and for me to better understand continual domain adaptation.

## Table of Contents

- [Technologies](#technologies)
- [Problem](#problem)
- [Approach](#approach)
- [Setup](#setup)
- [Dataset](#dataset)

## Technologies

Experiment is run with:

- Python 3.6
- PyTorch 1.5.1
- NumPy 1.19

## Problem

Back in July 2020, domain adaptation (DA) approachs are mainly focused on settings in which the domains are considered categorically. Categorical domains means that the domain index is just a label (i.e., either source or target). Thus, the authors at CIDA proposed to focus on DA among continuously indexed domains, where the domain index can represent a continuous variable (e.g. age, or in this case, rotation angle).

## Approach

The approach they proposed was an adversarial discrimative approach but with a modified discriminator. Instead of classifying a domain label, their discriminator regresses a domain index.

They also proposed a probabilistic discriminator that models the domain index distribution due to a problem of equilibriums with relatively poor domain alignments (proven by theoretical guarantees analysis). A probabilistic discriminator can help capture the underlying relation among domains, and enjoy better theoretical guarantees in terms of domain alignment.

Their approach, continuously indexed domain adaptation (CIDA), consists mainly of three components:

1. Encoder
2. Predictor
3. Discriminator

### Encoder

The encoder $E$ tries to learn domain-invariant encodings $z = E(x)$ (or $z = E(x, u)$) such the distribution of the encodings from all domains are aligned. It is formally required for the distribution of the encodings to be aligned $p(z|u1) = p(z|u2), ∀u1, u2 ∈ U$ because it implies that $z$ and $u$ are independent. In addition, all labels can be accurately predicted by the shared predictor in the different domains.

### Discriminator

The discriminator $D$ regresses the domain index with encoding as input, instead of classifying the encoding into categorical domains, because in continuously indexed domains, small changes in $u$ should lead to small changes in the encoding.

The other approach, dubbed probabilistic CIDA (PCIDA), is an advanced version of CIDA, which enjoys better theoretical guarantees to match both the mean and variance of the distribution while the vanilla CIDA can only match its mean.

### Predictor

The predictor predicts the class with the domain input data and index.

## Setup

The setup is unsupervised domain adaptation. There is access to labeled data with the domain index from source domains and unlabeled data from target domains. The goal is to accurately predict the labels for data in the target domains.

## Dataset

The dataset used is rotated MNIST. Rotated MNIST is the MNIST dataset augmented with rotation (represents domain index). These rotation angles are random angles sampled from intervals of 45 degrees (i.e., (0, 45), (45, 90), ..., (315, 360)).

### Credits
```
@inproceedings{DBLP:conf/icml/WangHK20,
  author    = {Hao Wang and
               Hao He and
               Dina Katabi},
  title     = {Continuously Indexed Domain Adaptation},
  booktitle = {ICML},
  year      = {2020}
}
```
