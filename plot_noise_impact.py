import csv
import os
import itertools
import argparse
from matplotlib import patches


import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']


score_folders = ['scores/noise_comparison/0.0', 'scores/noise_comparison/0.1', 'scores/noise_comparison/0.2']

x_ticks = {'ALOI': r'$\textsc{ALOI}$', 
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

x_ticks = np.array([x_ticks[x.split('_')[0]] for x in sorted(os.listdir(score_folders[0]))])

methods_in_plot = ['SLADe', 'SSDO', 'P-SVM', 'HIF', 'GP']

result = {x.split('/')[2]: {m: [] for m in methods_in_plot} for x in score_folders}

for folder in score_folders:
    noise_level = folder.split('/')[2]
    files = sorted(os.listdir(folder))

    for file in files:
        with open(folder + '/' + file, 'r') as f:
            csvreader = csv.reader(f)
            methods = next(itertools.islice(csvreader, 6, None))
            
            mean_performance = {method: [] for method in methods}

            for row in csvreader:
                try:
                    row = [float(x) for x in row]
                except ValueError:
                    continue
                
                for (i, m) in enumerate(methods):
                    mean_performance[m].append([x for x in row[i + 1:len(row):len(methods)]])

            for m in methods:
                if m in methods_in_plot:
                    result[noise_level][m].append(np.average(mean_performance[m]))


ind = np.arange(len(result['0.0'][methods_in_plot[0]]))
width = 0.15

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

colors = {
    'SLADe': 'r',
    'SSDO': 'darkblue',
    'P-SVM': 'darkorange',
    'GP': 'limegreen',
    'HIF': 'deepskyblue'
}

labels = {
            'SLADe': r'$\textsc{SLADe}$',
            'SSDO': r'$\textsc{SSDO}$',
            'P-SVM': r'$\textsc{P-SVM}$',
            'HIF': r'$\textsc{HIF}$',
            'GP': r'$\textsc{GP}$'
        }

proxy = []
l = []
for i, m in enumerate(methods_in_plot):
    
    ax1.bar(ind + i*width, bottom=np.array(result['0.0'][m]), height=-np.array(result['0.0'][m]) + np.array(result['0.1'][m]), width=width, color=colors[m], alpha=0.5)
    ax1.bar((ind + 0.5 + 2*width)[:-1], bottom=np.zeros(len(ind))[:-1], height=np.ones(len(ind))[:-1], width=0.03, color='grey', alpha=0.3)
    ax1.scatter(ind + i*width, np.array(result['0.0'][m]), marker='*', color=colors[m], edgecolors='k', linewidths=0.5, label=labels[m], s=36)
    ax1.scatter(ind + i*width, np.array(result['0.1'][m]), marker='X', color=colors[m], edgecolors='k',  linewidths=0.5, s=36)
    ax2.bar(ind + i*width, bottom=np.array(result['0.0'][m]), height=-np.array(result['0.0'][m]) + np.array(result['0.2'][m]), width=width, color=colors[m], alpha=0.5)
    ax2.bar((ind + 0.5 + 2*width)[:-1], bottom=np.zeros(len(ind))[:-1], height=np.ones(len(ind))[:-1], width=0.03, color='grey', alpha=0.3)
    ax2.scatter(ind + i*width, np.array(result['0.0'][m]), marker='*', color=colors[m], edgecolors='k',  linewidths=0.5, s=36)
    ax2.scatter(ind + i*width, np.array(result['0.2'][m]), marker='X', color=colors[m], edgecolors='k',  linewidths=0.5, s=36)
    proxy.append(patches.Patch(color=colors[m]))
    l.append(labels[m])



ax1.set_xticks([])
ax1.set_xticklabels([])
ax2.set_xticks(ind+2*width)
ax2.set_xticklabels(x_ticks, rotation=90, fontsize=16)
ax1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax1.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=16)
ax2.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax2.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'], fontsize=16)
ax1.set_xlim(ind[0] - 0.5 + 2*width, np.amax(ind) + 0.5 + 2*width)
ax1.set_ylim(0.45, 1)
ax2.set_xlim(ind[0] - 0.5 + 2*width, np.amax(ind) + 0.5 + 2*width)
ax2.set_ylim(0.45, 1)
ax1.set_title('10\% noise',
        fontdict={
            'fontsize': 16,
            'alpha': 1,
            'color': 'white'
        },
        bbox={
            'boxstyle':'round',
            'alpha': 1
        },
        pad = 20,
        y=0.77, x=0.01, loc='left')
ax2.set_title('20\% noise',
        fontdict={
            'fontsize': 16,
            'alpha': 1,
            'color': 'white'
        },
        bbox={
            'boxstyle':'round',
            'alpha': 1
        },
        pad = 20,
        y=0.77, x=0.01, loc='left')

fig.subplots_adjust(wspace=0, hspace=0.02)
fig.legend(proxy, l, loc='upper center', bbox_to_anchor=(0.515, 1), ncol=5, fontsize=18, )
fig.text(0.08, 0.40, 'AUROC', ha='center', rotation='vertical', fontsize=26)

if not os.path.isdir('plots/noise_impact/'):
    os.makedirs('plots/noise_impact/')
fig.savefig('plots/noise_impact/noise_impact_plot.pdf', dpi=300, bbox_inches='tight', format='pdf')