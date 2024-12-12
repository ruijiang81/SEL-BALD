from copy import deepcopy
from pprint import pprint

import pandas as pd
import numpy as np
import torch.cuda
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_classification

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baal import ActiveLearningDataset, ModelWrapper
from baal.active import ActiveLearningLoop, ActiveLearningLoopHuman, ActiveLearningLoopHuman_ColdStart, ActiveLearningLoopHuman_ColdStart_SELBALD
from baal.active.heuristics import Variance, Entropy, BALD, Random, Margin,Naive_BALD, e_BALD, Joint_BALD, Joint_Naive_BALD, \
    Joint_BALD_UCB, Joint_BALD_TS
from baal.bayesian.dropout import patch_module
from baal.utils.metrics import Accuracy, PRAuC
from baal.utils.array_utils import to_prob

import uci_dataset as ucidataset

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#sans_serif 
plt.rcParams['font.family'] = 'sans-serif'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='law')
parser.add_argument('--method', type=str, default='random')
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--qs', type=int, default=20)
parser.add_argument('--nepoch', type=int, default=1000)
parser.add_argument('--cost_examine', type=float, default=1)
parser.add_argument('--cost_label', type=float, default=5)
parser.add_argument('--beta', type=float, default=0.75)
parser.add_argument('--num_iter', type=int, default=40)

args = parser.parse_args()
dataset = args.dataset
method = args.method
seed = args.seed

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f'cuda available: {use_cuda}')

def weight_init_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal(m.weight, mode="fan_in", nonlinearity="relu")


# DATASET : https://archive.ics.uci.edu/ml/datasets/
# Physicochemical+Properties+of+Protein+Tertiary+Structure#


class FeatureDataset(Dataset):
    def __init__(self, x,y):
        self.x = x 
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]


# Change the following paths with respect to the location of `CASP.csv`
# You might need to add the path to the list of python system paths.

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
    torch.backends.cudnn.benchmark = False 

seed_everything(seed)

if dataset == 'synthetic_circle':
    n_samples = 1000
    from sklearn.datasets import make_circles
    x, y = make_circles(n_samples=n_samples, noise=0.1, random_state=seed, factor=0.5)
    idx = np.where(x[:,1] < 0.5)[0]
    idx = np.random.choice(idx, int(len(idx) * 0.9), replace=False)
    x = np.delete(x, idx, axis=0)
    y = np.delete(y, idx, axis=0)

    train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=0.3, random_state=seed)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long).reshape(-1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long).reshape(-1)
    color = ['r' if i == 0 else 'blue' for i in y]
    plt.scatter(x[:,0], x[:,1], c=color)
    plt.savefig(f'./fig/{dataset}_data.png')
    plt.close()
if dataset == 'mushroom':
    df = pd.read_csv('./data/agaricus-lepiota.data', header=None)
    df.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', \
                  'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', \
                 'veil-type', 'veil-color', 'ring-number', 'ring-type', \
                    'spore-print-color', 'population', 'habitat']
    # categorical encoding
    y = (df.iloc[:,0] == 'p').astype(int).values
    df = df.drop(columns=['class'])
    df = pd.get_dummies(df)
    x = df.iloc[:,1:].values
    sc = StandardScaler()
    x = sc.fit_transform(x)
    train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=0.3, random_state=seed)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long).reshape(-1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long).reshape(-1)
elif dataset == 'adult':
    df = pd.read_csv('./data/adult.data', header=None)
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', \
                  'race','sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    df = df.replace('?', np.nan)
    df = df.dropna()
    y = (df['income'] == ' >50K').astype(int).values
    df = df.drop(columns=['income'])
    # categorical encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    x = df.values

    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    x, y = rus.fit_resample(x, y)
    
    sc = StandardScaler()
    x = sc.fit_transform(x)
    train_idx, test_idx = train_test_split(np.arange(len(x)), test_size=0.3, random_state=seed)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long).reshape(-1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long).reshape(-1)

# create a human class that has a get_probability method to reject labeling the samples
class Human:
    def __init__(self, model = None):
        # x_train.shape[1]
        self.threshold = np.random.randn(x_train.shape[1])
        self.model = model

    def train_human_behavior_model(self, dataset):
        # train a model to predict whether to label or not
        pass 

    def hbm_init(self, x, y=None):
        #human will follow their first decision 
        np.random.seed(int(np.abs(sum(x)*1e6)))
        rs = np.random.RandomState(int(np.abs(sum(x)*1e6)))
        prob = 1 / (1 + np.exp(-np.dot(x, self.threshold)))
        if dataset == 'synthetic_circle':
            #if x[0] <= -0.5 and x[1]<=-0.25:
            #    prob = 0.1
            #elif x[0] <= -0.5 and x[1]>0.25:
            #    prob = 0.1 
            #elif x[0] < -0.5:
            #    prob = 0.6
            #elif x[0] > 0.5:
            #    prob = 0.9 
            #else:
            #    prob = 0.6 
            if x[1] >= 0:
                prob = 0 
            elif x[0] >= 0.:
                prob = 0.9 
            elif x[0] < 0.:
                prob = 0.3
        elif dataset == 'mushroom':
            if x[1] < 0 and x[4] > 0:
                prob = 0.0
            elif x[1] < 0 and x[4] <= 0:
                prob = 0.
            elif x[1] >= 0 and x[4] > 0:
                prob = 0.3 
            else:
                prob = 0.9
        elif dataset == 'adult':
            if x[0] > np.quantile(x_train[:,0], 0.5) and x[1] > np.quantile(x_train[:,1], 0.5):
                prob = 0.
            elif x[0] > np.quantile(x_train[:,0], 0.5) and x[1] <= np.quantile(x_train[:,1], 0.5):
                prob = 0.
            elif x[0] <= np.quantile(x_train[:,0], 0.5) and x[1] > np.quantile(x_train[:,1], 0.5):
                prob = 0.3
            else:
                prob = 0.9
        # flip a coin to decide whether to label or not
        #label = np.random.binomial(1, prob)
        label = rs.binomial(1, prob)
        return prob, label        

    def hbm(self, x):
        np.random.seed(int(np.abs(sum(x)*1e6)))
        prob = 1 / (1 + np.exp(-np.dot(x, self.threshold)))
        if dataset == 'synthetic':
            if x[0] < 2:
                prob = 1 
            elif x[0] > 6:
                prob = 0 
            else:
                prob = 1
        elif dataset == 'synthetic_moon':
            prob_predicted = to_prob(self.ml_model.predict_on_batch(torch.tensor(x).reshape(-1,2).float().to(device), iterations=40).cpu().numpy()).mean(-1)
            prob_predicted = prob_predicted[0, 1]
            algorithm_aversion = 0.5 
            algorithm_compliance = 0.5            
            if x[0] <= 6 and x[1] >= 3 and x[0] > 2:
                prob = 0.6 
                prob = 0.6 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[0] <= 6 and x[1] < 3 and x[0] > 2:
                prob = 0.4
                prob = 0.4 * (algorithm_aversion) + prob_predicted * (1 - algorithm_aversion)
            elif x[0] <= 2:
                prob = 0 * (algorithm_aversion) + prob_predicted * (1 - algorithm_aversion)
            else:
                prob = 0.
        elif dataset == 'gmc':
            prob_predicted = to_prob(self.ml_model.predict_on_batch(torch.tensor(x).reshape(-1,x.shape[0]).float().to(device), iterations=40).cpu().numpy()).mean(-1)
            prob_predicted = prob_predicted[0, 1]
            algorithm_aversion = 0.5
            algorithm_compliance = 0.5

            if x[3] <= np.quantile(x_train[:,3], 0.3) and x[4] <= np.quantile(x_train[:,4], 0.7):
                # low debt and low income
                prob = 0.4 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[3] <= np.quantile(x_train[:,3], 0.3) and x[4] > np.quantile(x_train[:,4], 0.7):
                # low debt and high income
                prob = 0.8 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[3] > np.quantile(x_train[:,3], 0.3) and x[4] <= np.quantile(x_train[:,4], 0.7):
                # high debt and low income
                prob = 0.
            else:
                # high debt and high income
                prob = 0. * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
        elif dataset == 'law':
            prob_predicted = to_prob(self.ml_model.predict_on_batch(torch.tensor(x).reshape(-1,x.shape[0]).float().to(device), iterations=40).cpu().numpy()).mean(-1)
            prob_predicted = prob_predicted[0, 1]
            algorithm_aversion = 0.5
            if x[2] <= np.quantile(x_train[:,1], 0.5) and x[3] <= np.quantile(x_train[:,3], 0.5):
                prob = 0. 
            elif x[2] <= np.quantile(x_train[:,1], 0.5) and x[3] > np.quantile(x_train[:,3], 0.5):
                prob = 0.4 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[2] > np.quantile(x_train[:,1], 0.5) and x[3] <= np.quantile(x_train[:,3], 0.5):
                prob = 0. * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[2] > np.quantile(x_train[:,1], 0.5) and x[3] > np.quantile(x_train[:,3], 0.5):
                prob = 0.8 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            else:
                prob = 0.5 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
        elif dataset == 'credit':
            prob_predicted = to_prob(self.ml_model.predict_on_batch(torch.tensor(x).reshape(-1,x.shape[0]).float().to(device), iterations=40).cpu().numpy()).mean(-1)
            prob_predicted = prob_predicted[0, 1]
            algorithm_aversion = 0.5
            if x[9] <= np.quantile(x_train[:,9], 0.5) and x[10] <= np.quantile(x_train[:,10], 0.5):
                prob = 0. 
            elif x[9] <= np.quantile(x_train[:,9], 0.5) and x[10] > np.quantile(x_train[:,10], 0.5):
                prob = 0.4 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[9] > np.quantile(x_train[:,9], 0.5) and x[10] <= np.quantile(x_train[:,10], 0.5):
                prob = 0. * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[9] > np.quantile(x_train[:,9], 0.5) and x[10] > np.quantile(x_train[:,10], 0.5):
                prob = 0.8 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            else:
                prob = 0.5 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
        elif dataset == 'compas':
            prob_predicted = to_prob(self.ml_model.predict_on_batch(torch.tensor(x).reshape(-1,x.shape[0]).float().to(device), iterations=40).cpu().numpy()).mean(-1)
            prob_predicted = prob_predicted[0, 1]
            algorithm_aversion = 0.5
            if x[12] <= np.quantile(x_train[:,12], 0.2):
                prob = 0.8 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[12] > np.quantile(x_train[:,12], 0.2) and x[12] <= np.quantile(x_train[:,12], 0.4):
                prob = 0.6 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[12] > np.quantile(x_train[:,12], 0.4) and x[12] <= np.quantile(x_train[:,12], 0.6):
                prob = 0.4 * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            elif x[12] > np.quantile(x_train[:,12], 0.6) and x[12] <= np.quantile(x_train[:,12], 0.8):
                prob = 0. * algorithm_aversion + prob_predicted * (1 - algorithm_aversion)
            else:
                prob = 0. 
        # flip a coin to decide whether to label or not
        label = np.random.binomial(1, prob)
        return prob, label

human_labeler = Human()

al_dataset = ActiveLearningDataset(FeatureDataset(x,y))
al_ini_dataset = ActiveLearningDataset(FeatureDataset(x,y))
test_ds = FeatureDataset(x_test, y_test)

# get the corresponding human labels 
h_train_init = [human_labeler.hbm_init(x_train[i])[1] for i in range(len(x_train))]
h_test_init = [human_labeler.hbm_init(x_test[i])[1] for i in range(len(x_test))]


hidden = 32
model = nn.Sequential(
    nn.Flatten(), nn.Linear(x.shape[1], hidden), nn.LeakyReLU() , nn.Dropout(), \
    nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Dropout(), \
    nn.Linear(hidden, 2)
)

model = patch_module(model)  # Set dropout layers for MC-Dropout.
model.apply(weight_init_normal)

hbm_model = nn.Sequential(
    nn.Flatten(), nn.Linear(x.shape[1], hidden), nn.LeakyReLU() , nn.Dropout(), \
    nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Dropout(), \
    nn.Linear(hidden, 2)
)

human_model = patch_module(hbm_model)  # Set dropout layers for MC-Dropout.
human_model.apply(weight_init_normal)

import copy 
init_model = copy.deepcopy(model)

if use_cuda:
    model = model.cuda()
    hbm_model = hbm_model.cuda()
    init_model = init_model.cuda()

wrapper = ModelWrapper(model=model, criterion=nn.CrossEntropyLoss())

if 'syn' in dataset:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer_hbm = optim.SGD(human_model.parameters(), lr=args.lr, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_hbm = optim.Adam(human_model.parameters(), lr=args.lr)

wrapper_hbm = ModelWrapper(model=human_model, criterion=nn.CrossEntropyLoss())

#optimizer_init = optim.Adam(init_model.parameters(), lr=args.lr)
optimizer_init = optim.SGD(init_model.parameters(), lr=args.lr, momentum=0.9)
wrapper_init = ModelWrapper(model=init_model, criterion=nn.CrossEntropyLoss())


# We will use Variance as our heuristic for regression problems.
variance = Variance()

if method == 'random':
    criterion = Random()
elif method == 'bald':
    criterion = BALD()
elif method == 'entropy':
    criterion = Entropy()
elif method == 'margin':
    criterion = Margin()
elif method == 'naive_bald':
    criterion = Naive_BALD()
elif method == 'e_bald':
    criterion = e_BALD()
elif method == 'joint_bald':
    criterion = Joint_BALD()
elif method == 'joint_naive_bald':
    criterion = Joint_Naive_BALD()
elif method == 'joint_bald_ucb':
    criterion = Joint_BALD_UCB(beta = args.beta)
elif method == 'joint_bald_ts':
    criterion = Joint_BALD_TS()
elif method == 'e_random':
    criterion = e_Random()

# Setup our active learning loop for our experiments
if method == 'e_bald' or method == 'joint_bald' or method == 'joint_naive_bald' or method == 'joint_bald_ucb' or method == 'joint_bald_ts':
    al_loop = ActiveLearningLoopHuman_ColdStart_SELBALD(
        dataset=al_dataset,
        get_probabilities=wrapper.predict_on_dataset,
        heuristic=criterion, 
        query_size=args.qs,  # We will label 20 examples per step.
        # KWARGS for predict_on_dataset
        iterations=args.num_iter,  # 20 sampling for MC-Dropout
        batch_size=16,
        use_cuda=use_cuda,
        verbose=False,
        workers=0,
        human = human_labeler, 
        human_get_probabilities = wrapper_hbm.predict_on_dataset,
    )
else:
    # naive bald and random
    al_loop = ActiveLearningLoopHuman_ColdStart(
        dataset=al_dataset,
        get_probabilities=wrapper.predict_on_dataset,
        heuristic=criterion, 
        query_size=args.qs,  # We will label 20 examples per step.
        # KWARGS for predict_on_dataset
        iterations=args.num_iter,  # 20 sampling for MC-Dropout
        batch_size=16,
        use_cuda=use_cuda,
        verbose=False,
        workers=0,
        human = human_labeler,
    )

initial_weights = deepcopy(model.state_dict())
hbm_initial_weights = deepcopy(human_model.state_dict())

# humans first label some data to train a ml model 
if dataset == 'synthetic_circle':
    init_num = 50 
elif dataset == 'mushroom':
    init_num = 10 
elif dataset == 'adult':
    init_num = 50 
else:
    init_num = 10 

oracle_ask_map = np.zeros_like(al_dataset.labelled_map)
tried_labeling = al_dataset.label_randomly_human_init(init_num, human_labeler) 
oracle_ask_map[tried_labeling] = 1
init_labeled = np.where(al_dataset.labelled)[0]

acc = []
auc = []
loss = []
f1 = []
aucroc = []
num_samples = []
total_costs = []
examine_costs = []
label_costs = []
current_labeled = al_dataset.labelled 


train_loss = wrapper_init.train_on_dataset(
    al_dataset, optimizer=optimizer_init, batch_size=args.bs, epoch=args.nepoch, use_cuda=use_cuda, workers=0
)

human_labeler.ml_model = wrapper_init

#al_dataset.label_randomly_human(init_num, human_labeler)

h_train = [human_labeler.hbm_init(x_train[i])[1] for i in range(len(x_train))]
h_test = [human_labeler.hbm_init(x_test[i])[1] for i in range(len(x_test))]
human_dataset = ActiveLearningDataset(FeatureDataset(x, h_train))



num_current_examined = init_num
num_curent_labeled = len(init_labeled)
initial_costs = num_current_examined * args.cost_examine + num_curent_labeled * args.cost_label

#total_costs.append(initial_costs)
#examine_costs.append(init_num * args.cost_examine)
#label_costs.append(len(init_labeled) * args.cost_label)

seed_everything(seed)
for step in range(args.steps):
    print(f"STEP {step}")
    model.load_state_dict(initial_weights)
    human_model.load_state_dict(hbm_initial_weights)


    if method in ['e_bald', 'joint_bald', 'joint_naive_bald', 'joint_bald_ucb', 'joint_bald_ts']:
        newlylabeled = tried_labeling
        # get the corresponding human labels
        if len(newlylabeled) > 0:
            human_need_label = human_dataset._oracle_to_pool_index(newlylabeled)
            human_dataset.label(human_need_label)
            # train the human behavior model
            human_train_loss = wrapper_hbm.train_on_dataset(
                human_dataset, optimizer=optimizer_hbm, batch_size=args.bs, epoch=args.nepoch, use_cuda=use_cuda, workers=0
            )
    
    if step > 0:
        num_current_examined += len(tried_labeling)
        num_curent_labeled = len(al_dataset)

    print('Training on', len(al_dataset), 'samples')
    train_loss = wrapper.train_on_dataset(
        al_dataset, optimizer=optimizer, batch_size=args.bs, epoch=args.nepoch, use_cuda=use_cuda, workers=0
    )
    print('Testing on', len(test_ds), 'samples')
    test_loss = wrapper.test_on_dataset(test_ds, batch_size=args.bs, use_cuda=use_cuda, workers=0)
    print('Test loss', test_loss)

    if dataset in ['synthetic', 'synthetic_moon', 'synthetic_moon2', 'synthetic_rect', 'synthetic_circle']:
        # visualize the acquired samples 
        labeled_data = []
        labeled_label = []
        for i in range(len(al_dataset)):
            thisx = al_dataset[i][0].numpy()
            thisy = al_dataset[i][1].numpy()
            labeled_data.append(thisx)
            labeled_label.append(thisy)
        labeled_data = np.array(labeled_data)
        labeled_label = np.array(labeled_label)
        color = ['r' if i == 0 else 'blue' for i in labeled_label]
        #plt.scatter(labeled_data[:,0], labeled_data[:,1], c=color)
        plt.scatter(labeled_data[:,0][labeled_label==0], labeled_data[:,1][labeled_label==0], c='r', label='class 0')
        plt.scatter(labeled_data[:,0][labeled_label==1], labeled_data[:,1][labeled_label==1], c='blue', label='class 1')

        tried_labeling_idx = tried_labeling
        tried_labeling_data = []
        tried_labeling_label = []
        for i in tried_labeling_idx:
            thisx = al_dataset.get_raw(i)[0].numpy()
            thisy = al_dataset.get_raw(i)[1].numpy()
            tried_labeling_data.append(thisx)
            tried_labeling_label.append(thisy)
        tried_labeling_data = np.array(tried_labeling_data)
        tried_labeling_label = np.array(tried_labeling_label)
        if len(tried_labeling_data) > 0:
            plt.scatter(tried_labeling_data[:,0], tried_labeling_data[:,1], c='green', marker='x', label = 'sent to human', alpha=0.5)
        plt.legend(loc = 'upper center', ncol = 3, bbox_to_anchor=(0.5, 1.1), fancybox=True, shadow=True, fontsize=10)

        # visualize the decision boundary
        Y = y
        #x = np.linspace(0, 8, 100)
        #y = np.linspace(0, 6, 100)
        x = np.linspace(x_train[:,0].min()-0.05, x_train[:,0].max()+0.05, 100)
        y = np.linspace(x_train[:,1].min()-0.05, x_train[:,1].max()+0.05, 100)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        xy = torch.tensor(xy, dtype=torch.float32)

        probs = wrapper.predict_on_batch(xy.to(device), iterations=args.num_iter)
        probs = to_prob(probs.cpu().numpy()).mean(-1)
        probs = probs[:, 1]
        probs = probs.reshape(xx.shape)
        plt.contourf(xx, yy, probs, cmap='RdBu', alpha=0.1)
        plt.colorbar()
        plt.savefig(f'./fig/coldstart_{dataset}_{method}_{args.beta}_labeled_data_step{step}_seed{args.seed}.png')
        plt.close()

        if method in ['e_bald', 'joint_bald', 'joint_naive_bald', 'joint_bald_ucb', 'joint_bald_ts', 'naive_bald']:
            # visualize human decision boundary
            #x = np.linspace(0, 8, 100)
            #y = np.linspace(0, 6, 100)
            x = np.linspace(x_train[:,0].min()-0.05, x_train[:,0].max()+0.05, 100)
            y = np.linspace(x_train[:,1].min()-0.05, x_train[:,1].max()+0.05, 100)
            xx, yy = np.meshgrid(x, y)
            xy = np.vstack([xx.ravel(), yy.ravel()]).T
            xy = torch.tensor(xy, dtype=torch.float32)
            probs = wrapper_hbm.predict_on_batch(xy.to(device), iterations=args.num_iter)
            probs = to_prob(probs.cpu().numpy()).mean(-1)
            probs = probs[:, 1]
            probs = probs.reshape(xx.shape)
            # red to green
            plt.contourf(xx, yy, probs, cmap='RdBu', alpha=0.1)
            plt.colorbar()

            # plot the underlying data distribution
            color = ['r' if i == 0 else 'blue' for i in y_train]
            #plt.scatter(x_train[:,0], x_train[:,1], c=color, alpha=0.1)
            plt.scatter(labeled_data[:,0][labeled_label==0], labeled_data[:,1][labeled_label==0], c='r', label='class 0')
            plt.scatter(labeled_data[:,0][labeled_label==1], labeled_data[:,1][labeled_label==1], c='blue', label='class 1')            
            plt.savefig(f'./fig/coldstart_{dataset}_{method}_{args.beta}_human_decision_boundary_step{step}_seed{args.seed}.png')
            plt.close()

    # predict f1 score and roc_auc_score on the test set
    from sklearn.metrics import f1_score, roc_auc_score, log_loss
    y_pred = wrapper.predict_on_batch(x_test.to(device), iterations=args.num_iter)
    y_pred = to_prob(y_pred.cpu().numpy()).mean(-1)
    
    if dataset == 'synthetic_circle':
        exam_idx = (x_test[:,1] >= 0)
        exam_idx = ~exam_idx
        #exam_idx = np.ones(len(x_test), dtype=bool)
    elif dataset == 'mushroom':
        exam_idx = (x_test[:,1] < 0)
        exam_idx = ~exam_idx
    elif dataset == 'adult':
        exam_idx = (x_test[:,0] > np.quantile(x_train[:,0], 0.5))
        exam_idx = ~exam_idx
    
    f1_ = f1_score(y_test[exam_idx], np.argmax(y_pred, axis=1)[exam_idx])
    aucroc_ = roc_auc_score(y_test[exam_idx], np.argmax(y_pred, axis=1)[exam_idx])
    acc_ = np.mean(np.argmax(y_pred, axis=1)[exam_idx] == y_test[exam_idx].numpy())
    logloss_ = log_loss(y_test[exam_idx], y_pred[:,1][exam_idx])
    
    f1.append(f1_)
    aucroc.append(aucroc_)
    acc.append(acc_)
    loss.append(logloss_)
    num_samples.append(len(al_dataset))
    total_costs.append(num_current_examined * args.cost_examine + num_curent_labeled * args.cost_label)
    examine_costs.append(num_current_examined * args.cost_examine)
    label_costs.append(num_curent_labeled * args.cost_label)

    print(f'dataset size: {len(al_dataset)}')
    print(f'f1: {f1_}')
    print(f'aucroc: {aucroc_}')
    print(f'acc: {acc_}')
    print(f'logloss: {logloss_}')
    print(f'total cost: {num_current_examined * args.cost_examine + num_curent_labeled * args.cost_label}')
    print(f'examine cost: {num_current_examined * args.cost_examine}')
    print(f'label cost: {num_curent_labeled * args.cost_label}')

    seed_everything(seed*100 + step)
    if method in ['e_bald', 'joint_bald', 'joint_naive_bald', 'joint_bald_ucb', 'joint_bald_ts']:
        flag, tried_labeling = al_loop.step(oracle_ask_map = oracle_ask_map)
        if len(tried_labeling) > 0:
            oracle_ask_map[tried_labeling] = 1
    else:
        flag, tried_labeling = al_loop.step()

    # update oracle_ask_map
    if len(tried_labeling) > 0:
        oracle_ask_map[tried_labeling] = 1
    if not flag:
        # We are done labelling! stopping
        break
            
    # now have vector of acc, auc and loss 
    results = pd.DataFrame({'acc': acc, 'loss': loss, 'f1': f1, 'aucroc': aucroc, 'seed': args.seed, \
                    'total_costs': total_costs, 'examine_costs': examine_costs, 'label_costs': label_costs, \
                    'num_samples': num_samples, 'dataset': dataset, 'method': method, 'bs': args.bs, 'lr': args.lr})
    results['step'] = np.arange(len(acc))
    results.to_csv(f'./log/coldstart_{dataset}_{method}_{seed}_{args.beta}_results.csv', index=False)
