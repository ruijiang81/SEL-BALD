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
from baal.active import ActiveLearningLoop, ActiveLearningLoopHuman, ActiveLearningLoopHuman_ColdStart 
from baal.active.heuristics import Variance, Entropy, BALD, Random, Margin,Naive_BALD, e_BALD, Joint_BALD, Joint_Naive_BALD, \
    Joint_BALD_UCB, Joint_BALD_TS, e_Entropy, e_BALD_UCB
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

if dataset == 'synthetic':
    n_samples = 1000
    x, y = make_classification(n_samples=n_samples, n_features=10, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=seed)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long).reshape(-1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long).reshape(-1)

    # visualize the data
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.savefig(f'./fig/{dataset}_syndata.png')
    plt.close()
elif dataset == 'synthetic_circle':
    n_samples = 20000
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
elif dataset == 'synthetic_rect':
    n_samples = 2000 
    # x1 sample from U[0,1] x U[1,2] - y = 0
    # x2 sample from U[1,2] x U[1,2] - y = 1
    # x3 sample from U[0,1] x U[-2,-1] - y = 0 
    # x4 sample from U[1,2] x U[-2,-1] - y = 1
    # x5 sample from U[-5,-4] x U[1,1.5] - y = 0
    # x6 sample from U[-5,-4] x U[0.5,1] - y = 1
    # x7 sample from U[4,5] x U[0.5,1] - y = 0
    # x8 sample from U[4,5] x U[1,1.5] - y = 1
    multiplier = 8 
    x1 = np.concatenate([np.random.uniform(0,1, n_samples//8 * multiplier).reshape(-1,1), np.random.uniform(0,2, n_samples//8 * multiplier).reshape(-1,1)], axis=1)
    x2 = np.concatenate([np.random.uniform(1,2, n_samples//8 * multiplier).reshape(-1,1), np.random.uniform(0,2, n_samples//8 * multiplier).reshape(-1,1)], axis=1)
    x3 = np.concatenate([np.random.uniform(0,1, n_samples//8 * multiplier).reshape(-1,1), np.random.uniform(-2,0, n_samples//8 * multiplier).reshape(-1,1)], axis=1)
    x4 = np.concatenate([np.random.uniform(1,2, n_samples//8 * multiplier).reshape(-1,1), np.random.uniform(-2,0, n_samples//8 * multiplier).reshape(-1,1)], axis=1)
    x5 = np.concatenate([np.random.uniform(-5,-4, n_samples//8).reshape(-1,1), np.random.uniform(-0.5,0., n_samples//8).reshape(-1,1)], axis=1)
    x6 = np.concatenate([np.random.uniform(-5,-4, n_samples//8).reshape(-1,1), np.random.uniform(-1,-0.5, n_samples//8).reshape(-1,1)], axis=1)
    x7 = np.concatenate([np.random.uniform(4,5, n_samples//8).reshape(-1,1), np.random.uniform(-1,-0.5, n_samples//8).reshape(-1,1)], axis=1)
    x8 = np.concatenate([np.random.uniform(4,5, n_samples//8).reshape(-1,1), np.random.uniform(-0.5,0., n_samples//8).reshape(-1,1)], axis=1)
    x = np.concatenate([x1,x2,x3,x4,x5,x6,x7,x8], axis=0)
    y = np.concatenate([np.zeros(n_samples//8 * multiplier), np.ones(n_samples//8 * multiplier), \
        np.zeros(n_samples//8 * multiplier), np.ones(n_samples//8 * multiplier), \
        np.zeros(n_samples//8), np.ones(n_samples//8), np.zeros(n_samples//8), np.ones(n_samples//8)], axis=0)
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
elif dataset == 'synthetic_guassian':
    n_samples = 2000
    x1 = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n_samples//2)
    x2 = np.random.multivariate_normal([2,2], [[1,0],[0,1]], n_samples//2)
elif dataset == 'synthetic_moon':
    nsamples = 20000
    x, y = make_moons(n_samples=nsamples, noise=0.1)
    y = 1 - y 
    x = (x+1) * 2.5
    # downsample the part with x[0]<=6 
    idx = np.where(x[:,0]<=6)[0]
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
    #plt.scatter(x[:,0], x[:,1], c=color)
    plt.scatter(x[:,0][y==0], x[:,1][y==0], c='r', label='class 0')
    plt.scatter(x[:,0][y==1], x[:,1][y==1], c='blue', label='class 1')
    plt.legend()
    plt.savefig(f'./fig/{dataset}_data.png')
    plt.close()
elif dataset == 'synthetic_moon2':
    nsamples = 20000
    x, y = make_moons(n_samples=nsamples, noise=0.1)
    y = 1 - y 
    x = (x+1) * 2.5
    # downsample the part with x[0]<=6 
    idx = np.where(x[:,0]<=6)[0]
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
    #plt.scatter(x[:,0], x[:,1], c=color)
    plt.scatter(x[:,0][y==0], x[:,1][y==0], c='r', label='class 0')
    plt.scatter(x[:,0][y==1], x[:,1][y==1], c='blue', label='class 1')
    plt.legend()
    plt.savefig(f'./fig/{dataset}_data.png')
    plt.close()    
elif dataset == 'law':
    # load arff 
    #import arff
    #df = arff.load(open('./data/law_dataset.arff'))
    from scipy.io import arff
    data = arff.loadarff('./data/law_dataset.arff')
    df = pd.DataFrame(data[0])
    del df['zgpa'], df['zfygpa'], df['fulltime']
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    # down sampling the dataset to balance the classes
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
elif dataset == 'gmc':
    df = pd.read_csv("./data/gmc.csv")
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    # down sampling the dataset to balance the classes
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
elif dataset == 'gmcuniform':
    df = pd.read_csv("./data/gmc.csv")
    x = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    # down sampling the dataset to balance the classes
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
elif dataset == 'abolone':
    df = ucidataset.load_abolone()
elif dataset == 'default':
    df = pd.read_excel('./data/default_of_credit_card_clients.xls', header=1)
    df = df.drop(columns=['ID'])
    df = df.rename(columns={'PAY_0': 'PAY_1', 'default payment next month': 'default'})
    x = df.drop(columns=['default']).values
    y = df['default'].values
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
    df = ucidataset.load_adult()
elif dataset == 'compas':
    raw_data = pd.read_csv('./data/compas-scores-two-years.csv')
    df = raw_data[((raw_data['days_b_screening_arrest'] <=30) & 
      (raw_data['days_b_screening_arrest'] >= -30) &
      (raw_data['is_recid'] != -1) &
      (raw_data['c_charge_degree'] != 'O') & 
      (raw_data['score_text'] != 'N/A')
     )]
    import numpy as np
    from datetime import datetime
    from scipy.stats import pearsonr
    def date_from_str(s):
        return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
    df['length_of_stay'] = (df['c_jail_out'].apply(date_from_str) - df['c_jail_in'].apply(date_from_str)).dt.total_seconds()
    df_crime = pd.get_dummies(df['c_charge_degree'],prefix='crimefactor',drop_first=True)
    df_age = pd.get_dummies(df['age_cat'],prefix='age')
    df_race = pd.get_dummies(df['race'],prefix='race')
    df_gender = pd.get_dummies(df['sex'],prefix='sex',drop_first=True)
    df_score = pd.get_dummies(df['score_text'] != 'Low',prefix='score_factor',drop_first=True)
    df_lr = pd.concat([df_crime, df_age,df_race,df_gender,
                    df['priors_count'],df['length_of_stay'],df['two_year_recid']
                    ],axis=1)
    x = df_lr.iloc[:,:-1].values
    y = df_lr.iloc[:,-1].values
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
elif dataset == 'credit':
    data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    df = (
        pd.read_excel(io=data_url, header=1)
        .drop(columns=["ID"])
        .rename(
            columns={"PAY_0": "PAY_1", "default payment next month": "default"}
        )
        )
    x = df.drop(columns=["default"]).values
    y = df["default"].values
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
elif dataset == 'fraud':
    df = pd.read_csv('./data/Fraud_Data.csv')
    # one hot encoding for the categorical features
    cat_features = ['source', 'browser']
    df = pd.get_dummies(df, columns=cat_features)
    df['sex'] = (df.sex == 'M').astype(int)
    # difference between signup time and purchase time
    df['time_diff'] = (pd.to_datetime(df.purchase_time) - pd.to_datetime(df.signup_time)).dt.total_seconds()
    x = df.drop(columns=['user_id', 'signup_time', 'purchase_time', 'class','device_id','ip_address']).values
    y = df['class'].values
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
elif dataset == 'claims':
    df = pd.read_csv('./data/insurance_claims.csv')
    #cat_features = ['policy_state','insured_education_level']
    features = ['months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', \
        'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', \
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 'total_claim_amount', 'injury_claim', \
        'property_claim', 'vehicle_claim', 'auto_year']
    x = df[features].values
    y = (df['fraud_reported']=='Y').values
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
        np.random.seed(int(np.abs(sum(x)*1e6)))
        prob = 1 / (1 + np.exp(-np.dot(x, self.threshold)))
        if dataset == 'synthetic':
            if x[0] < 1.5:
                prob = 1 
            elif x[0] >= 6:
                prob = 0 
            else:
                prob = 0
        elif dataset == 'synthetic_moon':
            # accept with high probability if x[0] <= 6 and x[1] >= 3
            if x[0] <= 6 and x[1] >= 3 and x[0] > 3:
                prob = 0.6 
            elif x[0] <= 6 and x[1] < 3 and x[0] > 3:
                # more c
                prob = 0.4
            elif x[0] <= 3:
                prob = 0.2
            else:
                prob = 0.
        elif dataset == 'synthetic_moon2':
            # accept with high probability if x[0] <= 6 and x[1] >= 3
            if x[0] <= 6 and x[1] >= 3 and x[0] > 3:
                prob = 0.4
            elif x[0] <= 6 and x[1] < 3 and x[0] > 3:
                # more c
                prob = 0.4
            elif x[0] <= 3:
                prob = 0.8
            else:
                prob = 0.1            
        elif dataset == 'synthetic_rect':
            if x[0] <= 2 and x[0] >= 0 and x[1] >= 0:
                prob = 0 
            elif x[0] >= 4 and x[0] <= 5 and x[1] >= -0.5 and x[1] <= 0:
                prob = 0.3
            elif x[0] >= 4 and x[0] <= 5 and x[1] >= -0.5 and x[1] <= 0.:
                prob = 0.7
            elif x[0] <= -4 and x[0] >= -5 and x[1] >= -1 and x[1] <= -0.5:
                prob = 0.7 
            elif x[0] <= -4 and x[0] >= -5 and x[1] >= -0.5 and x[1] <= 0.:
                prob = 0.3
            else:
                prob = 1
        elif dataset == 'synthetic_circle':
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
        elif dataset == 'gmc':
            # use two segments of age and monthly income 
            if x[3] <= np.quantile(x_train[:,3], 0.7) and x[4] <= np.quantile(x_train[:,4], 0.7):
                # low debt and low income
                prob = 0.
            elif x[3] <= np.quantile(x_train[:,3], 0.7) and x[4] > np.quantile(x_train[:,4], 0.7):
                # low debt and high income
                prob = 0.
            elif x[3] > np.quantile(x_train[:,3], 0.7) and x[4] <= np.quantile(x_train[:,4], 0.7):
                # high debt and low income
                prob = 0.3
            else:
                # high debt and high income
                prob = 0.9
        elif dataset == 'gmcuniform':
            prob = 0.8              
        elif dataset == 'law':
            # low LSAT and low ugpa 
            if x[2] <= np.quantile(x_train[:,1], 0.5) and x[3] <= np.quantile(x_train[:,3], 0.5):
                prob = 0.
            # low LSAT and high ugpa
            elif x[2] <= np.quantile(x_train[:,1], 0.5) and x[3] > np.quantile(x_train[:,3], 0.5):
                prob = 0.4
            # high LSAT and low ugpa
            elif x[2] > np.quantile(x_train[:,1], 0.5) and x[3] <= np.quantile(x_train[:,3], 0.5):
                prob = 0.
            # high LSAT and high ugpa
            elif x[2] > np.quantile(x_train[:,1], 0.5) and x[3] > np.quantile(x_train[:,3], 0.5):
                prob = 0.8
            else:
                prob = 0.5
        elif dataset == 'credit':
            # low pay 5 and low pay 6
            if x[9] <= np.quantile(x_train[:,9], 0.5) and x[10] <= np.quantile(x_train[:,10], 0.5):
                prob = 0.
            # low pay 5 and high pay 6
            elif x[9] <= np.quantile(x_train[:,9], 0.5) and x[10] > np.quantile(x_train[:,10], 0.5):
                prob = 0.4
            # high pay 5 and low pay 6
            elif x[9] > np.quantile(x_train[:,9], 0.5) and x[10] <= np.quantile(x_train[:,10], 0.5):
                prob = 0.
            # high pay 5 and high pay 6
            elif x[9] > np.quantile(x_train[:,9], 0.5) and x[10] > np.quantile(x_train[:,10], 0.5):
                prob = 0.8
        elif dataset == 'compas':
            if x[12] <= np.quantile(x_train[:,12], 0.2):
                prob = 0.8 
            elif x[12] > np.quantile(x_train[:,12], 0.2) and x[12] <= np.quantile(x_train[:,12], 0.4):
                prob = 0.6
            elif x[12] > np.quantile(x_train[:,12], 0.4) and x[12] <= np.quantile(x_train[:,12], 0.6):
                prob = 0.4
            elif x[12] > np.quantile(x_train[:,12], 0.6) and x[12] <= np.quantile(x_train[:,12], 0.8):
                prob = 0.
            else:
                prob = 0.
        elif dataset == 'default':
            if x[0] <= np.quantile(x_train[:,0], 0.3) and x[-1] <= np.quantile(x_train[:,1], 0.3):
                prob = 0.8 
            elif x[0] <= np.quantile(x_train[:,0], 0.3) and x[-1] > np.quantile(x_train[:,1], 0.3):
                prob = 0.6
            elif x[0] > np.quantile(x_train[:,0], 0.3) and x[-1] <= np.quantile(x_train[:,1], 0.3):
                prob = 0.4
            elif x[0] > np.quantile(x_train[:,0], 0.3) and x[-1] > np.quantile(x_train[:,1], 0.3):
                prob = 0.
        elif dataset == 'fraud':
            # x0 purchase value 
            # x-1 time difference
            if x[0] > np.quantile(x_train[:,0], 0.7) and x[-1] <= np.quantile(x_train[:,-1], 0.3):
                prob = 0.8 
            elif x[0] > np.quantile(x_train[:,0], 0.7) and x[-1] > np.quantile(x_train[:,-1], 0.3):
                prob = 0.4
            elif x[0] <= np.quantile(x_train[:,0], 0.3) and x[-1] <= np.quantile(x_train[:,-1], 0.3):
                prob = 0.2
            elif x[0] <= np.quantile(x_train[:,0], 0.3) and x[-1] > np.quantile(x_train[:,-1], 0.3):
                prob = 0.1
            else:
                prob = 0.2
            # random generate a discretion behavior model
        elif dataset == 'claims':
            if x[0] <= np.quantile(x_train[:,0], 0.5) and x[-4] <= np.quantile(x_train[:,1], 0.5):
                prob = 0.6 
            elif x[0] <= np.quantile(x_train[:,0], 0.5) and x[-4] > np.quantile(x_train[:,1], 0.5):
                prob = 0.8
            elif x[0] > np.quantile(x_train[:,0], 0.5) and x[-4] > np.quantile(x_train[:,1], 0.5):
                prob = 0.4
            elif x[0] > np.quantile(x_train[:,0], 0.5) and x[-4] <= np.quantile(x_train[:,1], 0.5):
                prob = 0.
        # flip a coin to decide whether to label or not
        label = np.random.binomial(1, prob)
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


if dataset == 'gmc':
    hidden = 8 
else:
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

#optimizer = optim.Adam(model.parameters(), lr=args.lr)
#optimizer_hbm = optim.Adam(human_model.parameters(), lr=args.lr)
if 'syn' in dataset:
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer_hbm = optim.SGD(human_model.parameters(), lr=args.lr, momentum=0.9)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer_hbm = optim.Adam(human_model.parameters(), lr=args.lr)

wrapper_hbm = ModelWrapper(model=human_model, criterion=nn.CrossEntropyLoss())

optimizer_init = optim.Adam(init_model.parameters(), lr=args.lr)
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
elif method == 'e_entropy':
    criterion = e_Entropy()
elif method == 'e_bald_ucb':
    criterion = e_BALD_UCB()

# Setup our active learning loop for our experiments
if method == 'e_bald' or method == 'joint_bald' or method == 'joint_naive_bald' or method == 'joint_bald_ucb' or method == 'joint_bald_ts' or method == 'e_entropy' or method == 'e_bald_ucb':
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
        human_get_probabilities = wrapper_hbm.predict_on_dataset,
    )
else:
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
if dataset == 'synthetic' or dataset == 'synthetic_moon' or dataset == 'synthetic_rect' or dataset == 'synthetic_circle' or dataset == 'synthetic_moon2':
    init_num = 50 
else: 
    # gmc results in the main paper init_num = 50
    init_num = 50 

tried_labeling = al_dataset.label_randomly_human_init(init_num, human_labeler) 
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

    #if step == 0:
    #    tried_labeling = al_dataset.label_randomly_human_init(args.qs, human_labeler)

    #    human_need_label = human_dataset._oracle_to_pool_index(tried_labeling)
    #    human_dataset.label(human_need_label)
    #    # train the human behavior model
    #    human_train_loss = wrapper_hbm.train_on_dataset(human_dataset, optimizer=optimizer_hbm, batch_size=args.bs, epoch=args.nepoch, use_cuda=use_cuda, workers=1)

    if method in ['e_bald', 'joint_bald', 'joint_naive_bald', 'joint_bald_ucb', 'joint_bald_ts', 'e_entropy','e_bald_ucb']:
        #newlylabeled = set(list(np.where(al_dataset.labelled)[0])) - set(list(np.where(current_labeled)[0]))
        #current_labeled = al_dataset.labelled
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

        # plot underlying data distribution
        #color = ['r' if i == 0 else 'g' for i in y_train]
        #plt.scatter(x_train[:,0], x_train[:,1], c=color, alpha=0.1)

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

        if method in ['e_bald', 'joint_bald', 'joint_naive_bald', 'joint_bald_ucb', 'joint_bald_ts', 'naive_bald', 'e_entropy', 'e_bald_ucb']:
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

    #res = wrapper.get_metrics("test")
    #acc.append(res['test_accuracy'])
    #auc.append(res['test_auroc'])
    #loss.append(res['test_loss'])

    # predict f1 score and roc_auc_score on the test set
    from sklearn.metrics import f1_score, roc_auc_score, log_loss
    y_pred = wrapper.predict_on_batch(x_test.to(device), iterations=args.num_iter)
    y_pred = to_prob(y_pred.cpu().numpy()).mean(-1)
    
    if dataset == 'synthetic_moon':
        exam_idx = (x_test[:,0] <= 6) 
    elif dataset == 'synthetic_moon2':
        exam_idx = np.ones(len(x_test), dtype=bool)     
    elif dataset == 'synthetic_rect':
        exam_idx = (x_test[:,0] <= 2) & (x_test[:,0] >= 0.) & (x_test[:,1] >= 0)
        exam_idx = ~exam_idx
    elif dataset == 'synthetic_circle':
        exam_idx = (x_test[:,1] >= 0)
        exam_idx = ~exam_idx
        #exam_idx = np.ones(len(x_test), dtype=bool)
    elif dataset == 'law':
        exam_idx = (x_test[:,2] <= np.quantile(x_train[:,1], 0.5)) & (x_test[:,3] <= np.quantile(x_train[:,3], 0.5))
        exam_idx = ~exam_idx
    elif dataset == 'gmc':
        # high debt and low income
        # x[3] > np.quantile(x_train[:,3], 0.3) and x[4] <= np.quantile(x_train[:,4], 0.7)
        #exam_idx = (x_test[:,3] > np.quantile(x_train[:,3], 0.3)) & (x_test[:,4] <= np.quantile(x_train[:,4], 0.7))
        exam_idx = (x_test[:,3] <= np.quantile(x_train[:,3], 0.7)) 
        exam_idx = ~exam_idx
    elif dataset == 'credit':
        exam_idx = (x_test[:,9] <= np.quantile(x_train[:,9], 0.5)) & (x_test[:,10] <= np.quantile(x_train[:,10], 0.5))      
        exam_idx = exam_idx |  (x_test[:,9] > np.quantile(x_train[:,9], 0.5)) & (x_test[:,10] <= np.quantile(x_train[:,10], 0.5))
        exam_idx = ~exam_idx
    elif dataset == 'compas':
        exam_idx = (x_test[:,12] > np. quantile(x_train[:,12], 0.8)) 
        exam_idx = ~exam_idx
    elif dataset == 'default':
        exam_idx = (x_test[:,0] <= np.quantile(x_train[:,0], 0.3)) & (x_test[:,-1] <= np.quantile(x_train[:,-1], 0.3))
        exam_idx = ~exam_idx
    elif dataset == 'claims':
        exam_idx = (x_test[:,0] > np.quantile(x_train[:,0], 0.5)) & (x_test[:,-4] <= np.quantile(x_train[:,-4], 0.5))
        exam_idx = ~exam_idx
    elif dataset == 'gmcuniform':
        exam_idx = np.ones(len(x_test), dtype=bool)     
    
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
    seed_everything(seed)
    flag, tried_labeling = al_loop.step()
    if not flag:
        # We are done labelling! stopping
        break
            
    # now have vector of acc, auc and loss 
    # save them to a file
    #results = pd.DataFrame({'acc': acc, 'auc': auc, 'loss': loss, 'f1': f1, 'aucroc': aucroc, 'seed': args.seed, \
    #                'dataset': dataset, 'method': method, 'bs': args.bs, 'lr': args.lr})
    results = pd.DataFrame({'acc': acc, 'loss': loss, 'f1': f1, 'aucroc': aucroc, 'seed': args.seed, \
                    'total_costs': total_costs, 'examine_costs': examine_costs, 'label_costs': label_costs, \
                    'num_samples': num_samples, 'dataset': dataset, 'method': method, 'bs': args.bs, 'lr': args.lr})
    results['step'] = np.arange(len(acc))
    results.to_csv(f'./log/coldstart_{dataset}_{method}_{seed}_{args.beta}_results.csv', index=False)
