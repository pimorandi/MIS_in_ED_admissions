import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

import treetaggerwrapper as tt

from bs4 import BeautifulSoup
import unicodedata
from unidecode import unidecode
from html import unescape
import re

def get_ascii(title):
    txt = unescape(title)
    txt = unidecode(txt)
    txt = re.sub(r'[^\x20-\x7f\t\n\r]', "_", txt)
    soup = BeautifulSoup(txt, features="html.parser")
    return soup.get_text(separator="\n")

def lemmatize(corpus):
    """
    corpus: series of strings.
    Lemmatizazion performed using TreeTagger library
    """
    tagger = tt.TreeTagger(TAGLANG='it')
    lemmatized_struct = corpus.apply(lambda x: tagger.TagText(x))
    lemmatized_strings = lemmatized_struct.apply(lambda struc: ' '.join([x.split('\t')[-1] for x in struc]))
    return lemmatized_strings

def read_pkl(filename):
    with open(filename, 'rb') as f:
        obj = pkl.load(f)
    return obj

def write_pkl(obj, file_name, overwrite=False):
    if os.path.exists(file_name) & (overwrite==False):
        print(f"File {file_name} already exists. Set 'overwrite' option to True")
    else:
        with open(file_name, 'wb') as f:
            pkl.dump(obj, f)
            
def plot_roc(r, figsize=(8,5)):
    rocs = r[1].groupby('fpr').agg(['mean', 'std']).swaplevel(1,0, axis = 1)[1]
    train_AUC_mean = np.round(r[0].groupby('metric').agg(['mean', 'std']).loc['roc_auc_ovr', 'train'].values[0], 3)
    train_AUC_std = np.round(r[0].groupby('metric').agg(['mean', 'std']).loc['roc_auc_ovr', 'train'].values[1], 3)
    test_AUC_mean = np.round(r[0].groupby('metric').agg(['mean', 'std']).loc['roc_auc_ovr', 'test'].values[0], 3)
    test_AUC_std = np.round(r[0].groupby('metric').agg(['mean', 'std']).loc['roc_auc_ovr', 'test'].values[1], 3)
    
    f, ax = plt.subplots(figsize=figsize)

    ax = rocs['train']['mean'].plot(label=f'Train (AUC: {train_AUC_mean} $\pm$ {train_AUC_std})')
    _ = ax.fill_between(
        rocs.index.tolist(),
        rocs['train']['mean'] - rocs['train']['std'],
        rocs['train']['mean'] + rocs['train']['std'],
        color="grey",
        alpha=0.2,
    )
    _ = rocs['test']['mean'].plot(ax=ax, label=f'Test (AUC: {test_AUC_mean} $\pm$ {test_AUC_std})')
    _ = ax.fill_between(
        rocs.index.tolist(),
        rocs['test']['mean'] - rocs['test']['std'],
        rocs['test']['mean'] + rocs['test']['std'],
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    _ = ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", label="Chance", alpha=0.8)
    _ = ax.legend()
    _ = ax.grid(alpha=0.3)
    _ = ax.set_title('Receiver Operating Curve')
    _ = ax.set_xlabel('False Positive rate')
    _ = ax.set_ylabel('True Positive rate')
    return ax

def colorCode_performance(df, r):
    df['TRUE'] = r[2].true.values
    df['PRED'] = r[2].prediction.values
    cc_true = df.groupby('Priority').TRUE.apply(list)
    cc_pred = df.groupby('Priority').PRED.apply(list)
    
    cols = ['Precision','Recall','F1','Support']
    CC_prest = pd.DataFrame(columns=cols)
    perfs = precision_recall_fscore_support(cc_true.loc[1], cc_pred.loc[1])
    
    _c = [('Precision', '0'), ('Precision', '1'),
         ('Recall', '0'), ('Recall', '1'),
         ('F1', '0'), ('F1', '1'),
         ('Support', '0'), ('Support', '1'),]
    c = pd.MultiIndex.from_tuples(_c, names=["Metric", "Class"])
    CC_prest = pd.DataFrame(columns=c)

    for i in range(1,5):
        perfs = precision_recall_fscore_support(cc_true.loc[i], cc_pred.loc[i])
        CC_prest.loc[i] = np.concatenate(perfs)

    return CC_prest

def plot_demography(df):
    f, axs = plt.subplots(1,2,figsize=(14,5))

    control_values = df.query("Target==0").Gender.value_counts() / df.query("Target==0").Gender.value_counts().sum()
    mis = df.query("Target==1").Gender.value_counts() / df.query("Target==1").Gender.value_counts().sum()
    x = np.array([0, 1])
    w = 0.3
    _ = axs[0].bar(x=x-w/2, height=control_values, width=w, label='Control', alpha=0.5)
    _ = axs[0].bar(x=x+w/2, height=mis, width=w, label='MIS', alpha=0.5)
    _ = axs[0].set_xticks(x)
    _ = axs[0].set_xticklabels(['M','F'])
    _ = axs[0].legend(loc='lower right')

    b = np.linspace(0, 100, 25)
    _ = df.query("Target==0").Age.hist(bins=b, ax=axs[1], density=True, alpha=0.5, label='Control')
    _ = df.query("Target==1").Age.hist(bins=b, ax=axs[1], density=True, alpha=0.5, label='MIS')
    _ = axs[1].set_title('Age Distribution')
    _ = axs[1].grid(alpha=0.3)
    _ = axs[1].legend()

    _ = f.suptitle('Demographics', fontsize=18)
    _ = axs[0].set_title('Gender Proportions')
    _ = axs[1].set_title('Age Distribution')
    
    return f, axs