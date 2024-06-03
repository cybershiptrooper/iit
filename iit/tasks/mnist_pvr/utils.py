import torchvision.datasets as datasets
import torch as t
import torchvision

MNIST_CLASS_MAP = {k: [1, 1, 1, 1, 2, 2, 2, 3, 3, 3][k] for k in range(10)}
mnist_size = 28

def make_mnist_dataset():
    mnist_train = datasets.MNIST("./data", download=True)
    mnist_test = datasets.MNIST("./data", train=False, download=True)
    return mnist_train, mnist_test, 

def visualize_datapoint(dataset, index):
    image, label, intermediate_vars = dataset[index]
    print(f"Label: {label}")
    print(f"Intermediate vars: {intermediate_vars}")
    print(f"Image shape: {image.shape}")
    image = torchvision.transforms.functional.to_pil_image(image)
    image.show()


def visualize_image(input):
    image = torchvision.transforms.functional.to_pil_image(input)
    image.show()
