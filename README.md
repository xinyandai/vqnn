# VQ-NN
vector Quantized Neural Network (BNN) for pytorch
This is the pytorch version for the VQ-NN code, fro VGG and resnet models

Please install torch and torchvision by following the instructions at: http://pytorch.org/
# To run resnet18 for cifar10 dataset use: 

    python main_quantized.py --model resnet_quantized --model-config "{'depth': 18}"  --save resnet18_quantized --dataset cifar10
 
