# AlexNet Implementation

Whilst implementing AlexNet isn't hard, I want to better understand how to make a model.

## Quick start

### Use my pretrained model

```python
import torch
pretrained_alexnet = torch.hub.load("rjkilpatrick/alexnet:master", "alexnet", pretrained=True)
```

### Train your own model

```bash
git clone https://github.com/rjkilpatrick/alexnet
cd alexnet
conda install
jupyter notebook .
```

## License

### By Author

My code is under MIT.

### Dataset

We're (presently) using the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.
Generated for [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.
