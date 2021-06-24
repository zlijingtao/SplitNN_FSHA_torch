import tensorflow as tf
import numpy as np
import tqdm
import torch.nn.functional as F
# import sklearn
import matplotlib.pyplot as plt
import torch
# import tensorflow_datasets as tfds
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
from scipy.misc import imresize
BUFFER_SIZE = 10000
SIZE = 32

# getImagesDS = lambda X, n: np.concatenate([x[0].numpy()[None,] for x in X.take(n)])


def getImagesDS(X, n):
    image_list = []
    for i in range(n):
        image_list.append(X[i][0].numpy()[None,])
    return np.concatenate(image_list)

def parse(x):
    x = x[:,:,None]
    x = tf.tile(x, (1,1,3))    
    x = tf.image.resize(x, (SIZE, SIZE))
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def parseC(x):
    x = x / (255/2) - 1
    x = tf.clip_by_value(x, -1., 1.)
    return x

def make_dataset(X, Y, f):
    x = tf.data.Dataset.from_tensor_slices(X)
    y = tf.data.Dataset.from_tensor_slices(Y)
    x = x.map(f)
    xy = tf.data.Dataset.zip((x, y))
    xy = xy.shuffle(BUFFER_SIZE)
    return xy


def load_mnist():
    xpriv = datasets.MNIST(root='./data', train=True, download=True)

    xpub = datasets.MNIST(root='./data', train=False)


    # xpriv = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
    #         transforms.Resize(32, interpolation=2),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,)),
    #     ]))

    # xpub = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
    #         transforms.Resize(32, interpolation=2),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,)),
    #     ]))
    
    x_train = np.array(xpriv.data)
    y_train = np.array(xpriv.targets)
    x_test = np.array(xpub.data)
    y_test = np.array(xpub.targets)
    
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    x_train = np.tile(x_train, (1,3,1,1))
    x_test = np.tile(x_test, (1,3,1,1))

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train  = x_train / (255/2) - 1
    x_test  = x_test / (255/2) - 1
    x_train = torch.clip(x_train, -1., 1.)
    x_test = torch.clip(x_test, -1., 1.)
    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub

def load_cifar():
    xpriv = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))

    xpub = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]))
    
    return xpriv, xpub



def load_mnist_mangled(class_to_remove):
    xpriv = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomCrop(32, padding=4)
            
        ]))

    xpub = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomCrop(32, padding=4)
            
        ]))
    

    x_train = np.array(xpriv.data)
    y_train = np.array(xpriv.targets)
    x_test = np.array(xpub.data)
    y_test = np.array(xpub.targets)
    # remove class from Xpub
    (x_test, y_test), _ = remove_class(x_test, y_test, class_to_remove)
    # for evaluation
    (x_train_seen, y_train_seen), (x_removed_examples, y_removed_examples) = remove_class(x_train, y_train, class_to_remove)
    
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test)

    xpub = TensorDataset(x_test, y_test)

    x_removed_examples = torch.Tensor(x_removed_examples) # transform to torch tensor
    y_removed_examples = torch.Tensor(y_removed_examples)

    x_train_seen = torch.Tensor(x_train_seen)
    y_train_seen = torch.Tensor(y_train_seen)

    xremoved_examples = TensorDataset(x_removed_examples, y_removed_examples)
    xpriv_other = TensorDataset(x_train_seen, y_train_seen)
    
    return xpriv, xpub, xremoved_examples, xpriv_other


def load_fashion_mnist():
    xpriv = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomCrop(32, padding=4)
        ]))

    xpub = datasets.FashionMNIST(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomCrop(32, padding=4)
        ]))
    
    return xpriv, xpub

def remove_class(X, Y, ctr):
    mask = Y!=ctr
    XY = X[mask], Y[mask]
    mask = Y==ctr
    XYr = X[mask], Y[mask]
    return XY, XYr

def plot(X, label='', norm=True):
    n = len(X)
    X = (X+1) / 2 
    fig, ax = plt.subplots(1, n, figsize=(n*3,3))
    for i in range(n):
        if X[i].shape[0] == 1:
            ax[i].imshow(X[i].squeeze(), cmap=plt.get_cmap('gray'));  
        else:
            ax[i].imshow(X[i]);  
        ax[i].set(xticks=[], yticks=[], title=label)
