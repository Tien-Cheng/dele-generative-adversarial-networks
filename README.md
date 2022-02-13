# dele-generative-adversarial-networks
> Name: Oh Tien Cheng
> Student ID: 2012072
> Class: DAAA/FT/2B/01
> Module: ST1504 : DEEP LEARNING


This repository contains info on all the experiments I conducted for my DELE CA2 Part A assignmnent. The results are best viewed in the Weights and Biases report: https://wandb.ai/tiencheng/DELE_CA2_GAN/reports/Building-a-Conditional-GAN-for-CIFAR-10--VmlldzoxNTIyODEy?accessToken=y80x3w62xnmgohl2m2xvukixisz10e04nfbi2zyi559hmdoccdhbaxo90xktc1iy

This is because it contains interactive plots of the model loss and metrics, and contains extensive markdown of my thought process. In addition, there are some bugs with the images in the notebook markdown not loading correctly except when viewed in Google Colab.

## File Directory Structure
📦dele-generative-adversarial-networks
 ┣ 📂experiments (contains past experiments)
 ┃ ┣ 📜Baseline Conditional DCGAN.ipynb
 ┃ ┣ 📜SN Conditional DCGAN.ipynb
 ┃ ┣ 📜SNGAN - 4 Discriminator Steps Per Generator Step.ipynb
 ┃ ┣ 📜SNGAN - 5 Discriminator Steps Per Generator Step.ipynb
 ┃ ┣ 📜SNGAN - Larger Batch Size.ipynb
 ┃ ┣ 📜SNGAN - TTUR.ipynb
 ┃ ┗ 📜SNGAN Attempt 1.ipynb
 ┣ 📂images (final images for submission)
 ┃ ┣ 📜Airplane.png
 ┃ ┣ 📜Bird.png
 ┃ ┣ 📜Car.png
 ┃ ┣ 📜Cat.png
 ┃ ┣ 📜Deer.png
 ┃ ┣ 📜Dog.png
 ┃ ┣ 📜Frog.png
 ┃ ┣ 📜Horse.png
 ┃ ┣ 📜README.md
 ┃ ┣ 📜Ship.png
 ┃ ┗ 📜Truck.png
 ┣ 📂utils
 ┃ ┣ 📜DiffAugment_pytorch.py (official DiffAugment implementation)
 ┃ ┣ 📜layers.py
 ┃ ┣ 📜loss.py
 ┃ ┣ 📜truncate.py
 ┃ ┗ 📜visualize.py
 ┣ 📜.gitignore
 ┣ 📜DELE Part A Slides.pdf (Simple Slide Deck)
 ┣ 📜Exploratory Data Analysis.ipynb (EDA Notebook)
 ┣ 📜Final GAN Output & Evaluation.ipynb (Generating the 1000 Images)
 ┣ 📜Final Report.pdf (Printed out Weights & Biases Report)
 ┣ 📜Final Training Notebook.ipynb (Training the Final Model)
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┗ 📜requirements.txt