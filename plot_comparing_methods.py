import argparse
import csv
import os
import numpy as np
import itertools


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from helper_files.utils import subplots_centered

parser = argparse.ArgumentParser()
# Which folder are the CSV results in?
parser.add_argument('-scorefolder', required=True)
# How many columns of plots? -> 5 in the paper
parser.add_argument('-c', required=True)
# How many rows of plots? -> 5 in the paper
parser.add_argument('-r', required=True)

datasets = {'ALOI': r'$\textsc{ALOI}$', 
        'Annthyroid': r'$\textsc{Annthy}$',
        'Arrhythmia': r'$\textsc{Arrhy}$',
        'Cardiotocography': r'$\textsc{Cardio}$',
        'Glass': r'$\textsc{Glass}$',
        'HeartDisease': r'$\textsc{Heart}$',
        'Hepatitis': r'$\textsc{Hepa}$',
        'Ionosphere': r'$\textsc{Iono}$',
        'KDDCup99': r'$\textsc{KDD}$',
        'PageBlocks': r'$\textsc{Page}$',
        'Parkinson': r'$\textsc{Parkin}$',
        'PenDigits': r'$\textsc{Pen}$',
        'Pima': r'$\textsc{Pima}$',
        'Shuttle': r'$\textsc{Shuttle}$',
        'SpamBase': r'$\textsc{Spam}$',
        'Stamps': r'$\textsc{Stamps}$',
        'Waveform': r'$\textsc{Wave}$',
        'WBC': r'$\textsc{WBC}$',
        'WDBC': r'$\textsc{WDBC}$',
        'Wilt': r'$\textsc{Wilt}$',
        'WPBC': r'$\textsc{WPBC}$'
        }

args = parser.parse_args()

files = sorted(os.listdir(args.scorefolder))
c = int(args.c)
r = int(args.r)

# The number of columns/rows should be enough to cover all of the datasets
assert r * c >= len(files), 'Not enough subplots'
# There needs to be an even number of empty slots in the bottom row
assert (r * c - len(files)) // 2 == (r * c - len(files)) / 2, 'Subplots would not align'

figure, axis = subplots_centered(nrows=r, ncols=c, nfigs=len(files), figsize=(25, 25))
figure.subplots_adjust(wspace=0.05, hspace=0.05)

# Graphical stuff
len_lrow = len(files) - (r - 1)*c
index_xticks = np.arange(1, len_lrow + 1, step=1)
index_xticks = np.concatenate((index_xticks, np.arange(index_xticks[-1] + 1, index_xticks[-1] + (c - len_lrow)/2 + 1, step=1)))
index_xticks = len(files) - np.concatenate((index_xticks, np.arange(index_xticks[-1] + len_lrow + 1, index_xticks[-1] + len_lrow + (c - len_lrow)/2 + 1, step=1)))

index_yticks = np.arange(0, len(files), step=c)

for j, f in enumerate(files):
    axis[j].grid(alpha=0.4)
    axis[j].margins(0)

    if j in index_xticks:
        axis[j].set_xticks(np.arange(10, 60, step=10))
        axis[j].set_xticklabels(np.arange(10, 60, step=10), fontsize=40)
        
    else:
        axis[j].set_xticks(np.arange(10, 60, step=10))
        axis[j].set_xticklabels([])

    if j in [20]:
        axis[j].set_xlabel('\% of labels', fontsize=50, labelpad=10)
    
    if j in [5]:
        axis[j].set_ylabel('AUROC', fontsize=50, labelpad=10, y=-0.08)

    if j in index_yticks:
        axis[j].set_yticks(np.linspace(6, 9, 4)/10)
        axis[j].set_yticklabels(np.linspace(6, 9, 4)/10, fontsize=40)
        
    else:
        axis[j].set_yticks(np.linspace(6, 9, 4)/10)
        axis[j].set_yticklabels([])

    axis[j].set_title(datasets[f.split('/')[-1].split('_')[0]],
        fontdict={
            'fontsize': 40,
            'alpha': 0.7
        },
        bbox={
            'boxstyle':'round',
            'alpha': 0.2
        },
        pad = 20,
        y=0.02)

    

    with open('{}/{}'.format(args.scorefolder, f), 'r') as file:
        models = next(itertools.islice(csv.reader(file), 6, None))
        csvreader = csv.reader(file)
        
        x_labels = list(range(0, 60 + 1, 5))

        lines = [{
            'color': 'r',
            'linewidth': 6,
            'linestyle': '-',
            'marker': '^',
            'markersize': 2
        }, {
            'color': 'darkblue',
            'linewidth': 4,
            'linestyle': '--',
            'marker': '*',
            'markersize': 2
        }, {
            'color': 'darkorange',
            'linewidth': 4,
            'linestyle': '--',
            'marker': '*',
            'markersize': 2
        }, {
            'color': 'limegreen',
            'linewidth': 4,
            'linestyle': '--',
            'marker': '*',
            'markersize': 2
        }, {
            'color': 'deepskyblue',
            'linewidth': 4,
            'linestyle': '--',
            'marker': '*',
            'markersize': 2
        }]
        names = {
            'SLADe': r'$\textsc{SLADe}$',
            'SSDO': r'$\textsc{SSDO}$',
            'P-SVM': r'$\textsc{P-SVM}$',
            'HIF': r'$\textsc{HIF}$',
            'GP': r'$\textsc{GP}$'
        }

        models_mean = {
            m: np.zeros(len(x_labels)) for m in models
        }

        lines_models = {
            m: lines[i] for (i, m) in enumerate(models)
        }
        
        nb_models = len(models)
        
        row_count = 0
        for row in csvreader:
            try:
                row = [float(x) for x in row]
            except ValueError:
                continue

            row_count += 1
            ha_auc = float(row[0])
            models_performance = {
                m: [float(x) for x in row[i + 1:len(row):nb_models]] for (i, m) in enumerate(models_mean)
            }

            for m in models_performance:
                models_mean[m] += models_performance[m]

        
        for m in models_mean:
            axis[j].plot(x_labels, models_mean[m]/row_count, color=lines_models[m]['color'], linewidth=lines_models[m]['linewidth'], linestyle=lines_models[m]['linestyle'], label=names[m], marker=lines_models[m]['marker'], markersize=8)
                
        axis[j].axis(ymin=0.5, ymax=1)

handles, labels = axis[j].get_legend_handles_labels()
order = [4, 0, 1, 3, 2]
figure.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper center', ncol=len(labels), fontsize=40)
figure.subplots_adjust(wspace=0.05, hspace=0.02, top=.93, bottom=0, left=0, right=1)

if not os.path.isdir('plots/comparing_methods/'):
    os.makedirs('plots/comparing_methods/')
figure.savefig('plots/comparing_methods/{}.pdf'.format('_'.join(args.scorefolder.split('/')[2:])), dpi=300, bbox_inches='tight', format='pdf')






