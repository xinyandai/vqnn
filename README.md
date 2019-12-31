# VQ-NN

Vector Quantized Neural Network (VQNN) for pytorch
This is the pytorch version for the VQ-NN code, fro AlexNet and ResNet models

Please install torch and torchvision by following the instructions at: http://pytorch.org/

### To run resnet18 for cifar10 without quantization:

    python main_quantized.py --model resnet  --save temp_path --dataset cifar10 --quantize identical

### To run resnet18 for cifar10 with [Binary Connection](https://arxiv.org/pdf/1511.00363):

    python main_quantized.py --model resnet_quantized  --save temp_path --dataset cifar10 --quantize BC

### To run resnet18 for cifar10 with [Binary Neural Networks](https://papers.nips.cc/paper/6573-binarized-neural-networks.pdf):

    python main_quantized.py --model resnet_quantized  --save temp_path --dataset cifar10 --quantize BNN

### To run resnet18 for cifar10 with Vector Quantization, similar to [HSQ](https://arxiv.org/pdf/1911.04655):

    python main_quantized.py --model resnet  --save temp_path --dataset cifar10 --quantize VQ

### To run resnet50 for imagenet without quantization:

    python main_quantized.py --model resnet --model-config "{'depth': 50}"  --save temp_path --dataset imagenet --quantize identical

### To run alexnet for imagenet without quantization:

    python main_quantized.py --model alexnet  --save temp_path --dataset imagenet --quantize identical

### To run resnet18 for cifar10 with vq-optimizer

    python main_quantized.py --model resnet  --save temp_path --dataset cifar10 --quantize VQ --user_vq_optim true
