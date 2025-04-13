### Learning CUDA
This repository is aimed to document my CUDA and triton learning process. We will start with the process from the simplest of problems and then add complexity along the way. Where possible I will aim to demostrate the performance comparison between traditional methods such as in python, pytorch etc

### Installation 
1. Install pytorch: https://pytorch.org/get-started/locally/
2. Triton library is not made for windows so has to be downloaded from an external source: pip install -U triton-windows (Credit: https://github.com/woct0rdho/triton-windows)

### Goal
The goal is to build the Entire GPT model layer by layer only using Triton and perform a comparison with a GPT model built using torch

### Modules to build
1. FFN layer
2. ReLu activation
3. Self Attention
4. Layer Norm
5. Flash attention - forward and backward pass
6. Softmax layer 
