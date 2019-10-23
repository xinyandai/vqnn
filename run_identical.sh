# cifar10
python main_quantized.py --model resnet --dataset cifar10 --quantize identical --optimizer SGD --save cifar10/resnet18_identical_SGD
python main_quantized.py --model resnet --dataset cifar10 --quantize identical --optimizer Adam --save cifar10/resnet18_identical_Adam

# cifar100
python main_quantized.py --model resnet --dataset cifar100 --quantize identical --optimizer SGD --save cifar100/resnet18_identical_SGD
python main_quantized.py --model resnet --dataset cifar100 --quantize identical --optimizer Adam --save cifar100/resnet18_identical_Adam

# mnist-linear
python main_quantized.py --model logistic --dataset mnist_linear --quantize identical --optimizer SGD --save mnist/logistic_identical_SGD
python main_quantized.py --model logistic --dataset mnist_linear --quantize identical --optimizer Adam --save mnist/logistic_identical_Adam

# rcv1_binary
# python main_quantized.py --model logistic --dataset rcv1_binary --quantize identical --optimizer SGD --save rcv1/logistic_identical_SGD