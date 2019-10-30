# cifar10
python main_quantized.py --model resnet --dataset cifar10 --quantize VQ --optimizer SGD --save cifar10/resnet18_VQ_SGD
python main_quantized.py --model resnet --dataset cifar10 --quantize VQ --optimizer Adam --save cifar10/resnet18_VQ_Adam

# cifar100
python main_quantized.py --model resnet --dataset cifar100 --quantize VQ --optimizer SGD --save cifar100/resnet18_VQ_SGD
python main_quantized.py --model resnet --dataset cifar100 --quantize VQ --optimizer Adam --save cifar100/resnet18_VQ_Adam

# mnist-linear
python main_quantized.py --model logistic --dataset mnist_linear --quantize VQ --optimizer SGD --save mnist/logistic_VQ_SGD
python main_quantized.py --model logistic --dataset mnist_linear --quantize VQ --optimizer Adam --save mnist/logistic_VQ_Adam
# rcv1_binary
# python main_quantized.py --model logistic --dataset rcv1_binary --quantize VQ --optimizer SGD --save rcv1/logistic_VQ_SGD