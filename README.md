# Rotated MNIST with Continuously Indexed Domain Adaptation (CIDA)

This is a re-implementation of the continuously indexed domain adaptation (CIDA) approach on the rotated MNIST dataset. 

This experiement is based on and referenced from the research paper titled "[Continuously Indexed Domain Adaptation](https://github.com/hehaodele/CIDA/)" by authors Hao Wang and Hao He. 

The purpose of this re-implementation is strictly for educational purposes and for me to better understand continual domain adaptation.

## Table of Contents

* [Technologies](#technologies)
* [Problem](#problem)
* [Approach](#approach)
* [Setup](#setup)
* [Dataset](#dataset)

## Technology

Experiment is run with:
* Python 3.6
* PyTorch 1.5.1
* NumPy 1.19

## Problem

Existing domain adaptation (DA) methods focus on adaptation among categorical domains where the domain index is just a label (e.g., to adapt a model from image dataset like MNIST to another like SVHN. They wanted to focus on adaptation among continuously indexed domains (e.g., to adapt a model across a continuous domain index like adapting disease diagnosis and prognosis across patients of different ages.

## Approach

The proposed approach was a modified traditional adversarial adaptation with a discriminator that regresses the domain index using a distance-based loss (e.g., L2 or L1 loss) because the domain index represents a distance metric (i.e., captures a similarity distance between the domains with respect to the task).

Another proposed approach was to have a probabilistic discriminator that models the domain index distribution due to a problem of equilibriums with relatively poor domain alignments (proven by theoretical guarantees analysis). A probabilistic discriminator can help capture the underlying relation among domains, and enjoy better theoretical guarantees in terms of domain alignment.

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

The setup is unsupervised domain adaptation

## Dataset

The dataset used is rotated MNIST.

