# cifar10
python main_quantized.py --model resnet_quantized --dataset cifar10 --quantize BNN --optimizer SGD --save cifar10/resnet18q_BNN_SGD
python main_quantized.py --model resnet_quantized --dataset cifar10 --quantize BNN --optimizer Adam --save cifar10/resnet18q_BNN_Adam

# cifar100
python main_quantized.py --model resnet_quantized --dataset cifar100 --quantize BNN --optimizer SGD --save cifar100/resnet18q_BNN_SGD
python main_quantized.py --model resnet_quantized --dataset cifar100 --quantize BNN --optimizer Adam --save cifar100/resnet18q_BNN_Adam

# mnist-linear
python main_quantized.py --model logistic --dataset mnist_linear --quantize BNN --optimizer SGD --save mnist/logistic_BNN_SGD

# rcv1_binary
# python main_quantized.py --model logistic --dataset rcv1_binary --quantize BNN --optimizer SGD --save rcv1/logistic_BNN_SGD

