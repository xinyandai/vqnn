# VQ-NN
Vector Quantized Neural Network (VQNN) for pytorch
This is the pytorch version for the VQ-NN code, fro AlexNet and ResNet models

Please install torch and torchvision by following the instructions at: http://pytorch.org/
# To run resnet18 for cifar10 dataset use: 

    python main_quantized.py --model resnet_quantized --model-config "{'depth': 18}"  --save resnet18_quantized --dataset cifar10
    python main_quantized.py --model resnet_quantized  --save resnet18_quantized --dataset cifar10
 
