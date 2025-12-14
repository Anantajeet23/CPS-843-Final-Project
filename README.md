# CPS-843-Final-Project

 **Authors: Anantajeet Devaraj and Natasha Narasimhan.**

This repository contains the code and trained models for the paper:

> **“Comparative Study and Enhancement of Segmentation Architectures for Breast Tumor Segmentation”**

We compare four widely used segmentation architectures: **U-Net**, **U-Net++**, **Attention U-Net**, and **DeepLabV3 (ResNet-50 backbone)** and introduce a lightweight **Gabor Texture Attention Module (GTAM)** that can be plugged into each model to improve breast tumor segmentation performance on the **BUSI** and **UDIAT** breast ultrasound datasets.



# Key Files

* /metrics.py –  Metric implementations (Accuracy, Precision, Recall, Specificity, Dice, IoU).

* /utils.py – Utility functions (logging, helpers for training/eval, etc.).

* train.ipynb – Notebook to train baseline and GTAM-enhanced models.

* test.ipynb – Notebook to evaluate trained models and visualize predictions.

# Models

Baseline architectures

* Models/unet.py – U-Net

* Models/unetpp.py – U-Net++

* Models/att_unet.py – Attention U-Net

* DeepLabV3 (ResNet-50 backbone) used in GTAM variants

# Modules

* Models/Modules/attention.py – Attention blocks / gates for Attention U-Net.

* Models/Modules/gtam.py – Gabor Texture Attention Module (GTAM) implementation.

# GTAM-Enhanced Models

* Models/GTAM Models/unet_gtam.py – U-Net + GTAM

* Models/GTAM Models/unetpp_gtam.py – U-Net++ + GTAM

* Models/GTAM Models/att_unet_gtam.py – Attention U-Net + GTAM

* Models/GTAM Models/DeepLabV3_gtam.py – DeepLabV3 (ResNet-50) + GTAM

* Models/GTAM Models/fcn_resnet50_gtam.py – FCN-ResNet50 + GTAM (extra model)

# Datasets

* Dataset/dataset_BUSI.py – PyTorch dataset wrapper for the BUSI breast ultrasound dataset.

* Dataset/dataset_UDIAT.py – PyTorch dataset wrapper for the UDIAT breast ultrasound dataset.

You must download BUSI and UDIAT yourself and set the correct paths in these dataset files.



# Important: Paths Have Changed

> **The project structure has been cleaned up and reorganized.**

Because of this:

All previously hard-coded paths inside the training and testing code are now outdated.

In particular, you must update paths manually in:

* train.ipynb

* test.ipynb

* Dataset/dataset_BUSI.py

* Dataset/dataset_UDIAT.py

Any other script/notebook that:

* loads images or masks,

* loads/saves checkpoints under Trained Models/...,

or assumes an older directory layout.

Until you update these paths to match your local filesystem, the training and testing scripts will not run successfully.


# Trained Models

Pretrained weights used in the paper are to large to upload and can be found here:

>* https://drive.google.com/drive/folders/1ZNHw8XAcIighbqpGsZcCHIx4RMbqFCSi?usp=sharing

Each .pth file corresponds to the best performing checkpoint for that architecture/dataset combination.

You can load these weights directly in your own PyTorch scripts or via test.ipynb after fixing the paths.

