# cifar10
python main_quantized.py --model resnet_quantized --dataset cifar10 --quantize BC --optimizer SGD --save cifar10/resnet18q_BC_SGD
python main_quantized.py --model resnet_quantized --dataset cifar10 --quantize BC --optimizer Adam --save cifar10/resnet18q_BC_Adam

# cifar100
python main_quantized.py --model resnet_quantized --dataset cifar100 --quantize BC --optimizer SGD --save cifar100/resnet18q_BC_SGD
python main_quantized.py --model resnet_quantized --dataset cifar100 --quantize BC --optimizer Adam --save cifar100/resnet18q_BC_Adam

# mnist-linear
python main_quantized.py --model logistic --dataset mnist_linear --quantize BC --optimizer SGD --save mnist/logistic_BC_SGD
python main_quantized.py --model logistic --dataset mnist_linear --quantize BC --optimizer Adam --save mnist/logistic_BC_Adam
# rcv1_binary
# python main_quantized.py --model logistic --dataset rcv1_binary --quantize BC --optimizer SGD --save rcv1/logistic_BC_SGD
