import argparse
from pprint import pprint
import random
from copy import deepcopy

import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper
import numpy as np

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_datasets(dataset_name = "cifar10"):
    if dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize((28,28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    elif dataset_name == "mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    elif dataset_name == "fashion_mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    elif dataset_name == "tiny_imagenet":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(3 * [0.5], 3 * [0.5]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(3 * [0.5], 3 * [0.5]),
            ]
        )
    elif dataset_name == "svhn":
        # greyscale 
        transform = transforms.Compose(
            [
                transforms.Resize((28,28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    # define train and test set
    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(
            ".", train=True, transform=transform, target_transform=None, download=True
        )
        test_set = datasets.CIFAR10(
            ".", train=False, transform=transform, target_transform=None, download=True
        )
    elif dataset_name == "mnist":
        train_set = datasets.MNIST(
            ".", train=True, transform=transform, target_transform=None, download=True
        )
        test_set = datasets.MNIST(
            ".", train=False, transform=transform, target_transform=None, download=True
        )
    elif dataset_name == "fashion_mnist":
        train_set = datasets.FashionMNIST(
            ".", train=True, transform=transform, target_transform=None, download=True
        )
        test_set = datasets.FashionMNIST(
            ".", train=False, transform=transform, target_transform=None, download=True
        )
    elif dataset_name == "tiny_imagenet":
        train_set = datasets.ImageFolder(
                    "./data/tinyimagenet/tiny-imagenet-200/train", transform=transform
                )
        test_set = datasets.ImageFolder(
                    "./data/tinyimagenet/tiny-imagenet-200/test", transform=test_transform
                )
    elif dataset_name == "svhn":
        train_set = datasets.SVHN(
            root="./data/svhn", split='train', transform=transform, download=True
        )
        test_set = datasets.SVHN(
            root="./data/svhn", split='test', transform=transform, download=True
        )
    # randomly sample 10000 samples from the training set
    seed_everything(42)
    # downsample class 7, 8, 9 
    
    if dataset_name == "svhn":
        idx = [i for i, x in enumerate(train_set.labels) if x in [7, 8, 9]]
        original_idx = np.arange(len(train_set.labels))
    else:
        idx = [i for i, x in enumerate(train_set.targets) if x in [7, 8, 9]]
        original_idx = np.arange(len(train_set.targets))

    if dataset_name == "svhn":
        #left_idx = original_idx
        idx = random.sample(idx, 12000)
        left_idx = np.setdiff1d(original_idx, idx)
    else:
        idx = random.sample(idx, 12000)
        left_idx = np.setdiff1d(original_idx, idx)

    train_set.data = train_set.data[left_idx]

    if dataset_name == "svhn":
        train_set.labels = [train_set.labels[i] for i in left_idx]
        train_set.data = train_set.data[:10000]
        train_set.labels = train_set.labels[:10000]
        test_set.data = test_set.data[:10000]
        test_set.labels = test_set.labels[:10000]
    else:
        train_set.targets = [train_set.targets[i] for i in left_idx]
        train_set.data = train_set.data[:10000]
        train_set.targets = train_set.targets[:10000]
        test_set.data = test_set.data[:10000]
        test_set.targets = test_set.targets[:10000]

    active_set = ActiveLearningDataset(train_set, pool_specifics={"transform": transform})

    return active_set, test_set, train_set

def get_human_datasets(human_labeler, dataset_name = "cifar10"):
    if dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize((28,28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    elif dataset_name == "mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    elif dataset_name == "fashion_mnist":
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    elif dataset_name == "tiny_imagenet":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(3 * [0.5], 3 * [0.5]),
            ]
        )
    elif dataset_name == "svhn":
        transform = transforms.Compose(
            [
                transforms.Resize((28,28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    # Note: We use the test set here as an example. You should make your own validation set.
    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(
            ".", train=True, transform=transform, target_transform=None, download=True
        )
        test_set = datasets.CIFAR10(
            ".", train=False, transform=transform, target_transform=None, download=True
        )
    elif dataset_name == "mnist":
        train_set = datasets.MNIST(
            ".", train=True, transform=transform, target_transform=None, download=True
        )
        test_set = datasets.MNIST(
            ".", train=False, transform=transform, target_transform=None, download=True
        )
    elif dataset_name == "fashion_mnist":
        train_set = datasets.FashionMNIST(
            ".", train=True, transform=transform, target_transform=None, download=True
        )
        test_set = datasets.FashionMNIST(
            ".", train=False, transform=transform, target_transform=None, download=True
        )
    elif dataset_name == "tiny_imagenet":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(3 * [0.5], 3 * [0.5]),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(3 * [0.5], 3 * [0.5]),
            ]
        )
        train_set = datasets.ImageFolder(
            "./data/tinyimagenet/tiny-imagenet-200/train", transform=transform
        )
        test_set = datasets.ImageFolder(
            "./data/tinyimagenet/tiny-imagenet-200/test", transform=test_transform
        )
    elif dataset_name == "svhn":
        train_set = datasets.SVHN(
            root="./data/svhn", split='train', transform=transform, download=True
        )
        test_set = datasets.SVHN(
            root="./data/svhn", split='test', transform=transform, download=True
        )
    seed_everything(42)
    # downsample class 7, 8, 9 

    if dataset_name == "svhn":
        idx = [i for i, x in enumerate(train_set.labels) if x in [7, 8, 9]]
        idx = random.sample(idx, 12000)
        original_idx = np.arange(len(train_set.labels))
        left_idx = np.setdiff1d(original_idx, idx)
        #left_idx = original_idx

        train_set.data = train_set.data[left_idx]
        train_set.labels = [train_set.labels[i] for i in left_idx]

        train_set.data = train_set.data[:10000]
        train_set.labels = train_set.labels[:10000]

        train_set.labels = [human_labeler.hbm_init(x, label)[1] for x, label in zip(train_set.data, train_set.labels)]
    else:
        idx = [i for i, x in enumerate(train_set.targets) if x in [7, 8, 9]]
        idx = random.sample(idx, 12000)
        original_idx = np.arange(len(train_set.targets))
        left_idx = np.setdiff1d(original_idx, idx)

        train_set.data = train_set.data[left_idx]
        train_set.targets = [train_set.targets[i] for i in left_idx]

        train_set.data = train_set.data[:10000]
        train_set.targets = train_set.targets[:10000]

        train_set.targets = [human_labeler.hbm_init(x, label)[1] for x, label in zip(train_set.data, train_set.targets)]
    active_set = ActiveLearningDataset(train_set, pool_specifics={"transform": transform})

    return active_set