# GenBeamJoint: Controllable diffusion framework for generating realistic Failure images of RC beam–column joints using geometry-conditioned guidance.

## Inference

The code for training pipeline will be added soon. This page provide inference of the model.

## Overview

> **Abstract:** *We introduce a GenBeamJoint, controllable geometry-conditioned diffusion framework for generating realistic failure images of reinforced concrete beam--column joints.* 

The framework focused on generation of three structural failure modes, according to ACI 352R-02 classification: B failure, BJ failure, and J failure, each failure mode characterized by damage area, the pattern of the damage and severity progression. The main goal of this study is to address the scarcity and limitation of the real-world data by generating realistic, physically accurate samples that rely on geometric and visual consistency of the damage. While the common text-to-image models' output unreliable and often ambigious for generating beam-column joints, we adopt a two-stage generation strategy: a diffusion model first synthesizes structurally consistent undamaged beam-column joint images, and an intermediate conditioning module then translates textual failure descriptions into failure-aware edge maps, which are applied to the base image to generate damage patterns. 

GenBeamJoint works in two stage pipeline:

- The first stage: Conditional Edge-to-Edge Generation
- The second stage: Edge-Conditioned Diffusion Model with Multi-Scale Control Adapter

Trained on unique and small dataset of RC Beam, Joint and Beam-Joint failure mode images collected from experimental and education sites, scientific literature, GenBeamJoint generates structurally precise RC beam-joint samples, close to the real images distribution for evaluation and data augmentation purposes. The framework supports generation of different failure class and has full control over the severity of the damage and failure distribution.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/nubcico/GenBeamJoint.git
cd GenBeamJoint
```

2. Set Up Your Environment
Install all the necessary Python packages using the requirements.txt file.

```
pip install -r requirements.txt
```

## References

This work builds upon the following repositories:

- [stable-diffusion](https://github.com/CompVis/stable-diffusion/)
- [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter)
