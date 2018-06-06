import matplotlib as mpl
mpl.use('agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from catalog import Source, Binary, Catalog, Pink

def get_shape(book):
    max_shape = 0

    for i in book.keys():
        curr_shape = i[0]*i[1]
        if curr_shape > max_shape:
            max_shape = curr_shape
            shape = i

    return shape

def label_plot(book, shape, save=None, xtick_rotation=None, 
                color_map='gnuplot2', title=None, weights=None, figsize=(6,6),
                literal_path=False, count_text=False):
    '''Isolated function to plot the attribute histogram if the data is labelled in 
    nature

    book - dict
        A dictionary whose keys are the location on the heatmap, and values
        are the list of values of sources who most belonged to that grid
    shape - tuple
        The shape of the grid. Should attempt to get this from the keys or
        possible recreate it like in self.attribute_heatmap() 
    save - None or Str
        If None, show the figure on screen. Otherwise save to the path in save
    xtick_rotation - None or float
        Will rotate the xlabel by rotation
    color_map - str
        The name of the matplotlib.colormap that will be passed directly to matplotlib.pyplot.get_map()
    title - None of str
        A simple title strng passed to fig.suptitle()
    weights - None or dict
        If not None, the dict will have keys corresponding to the labels, and contain the total
        set of counts from the Binary file/book object. This will be used to `weigh` the contribution
        per neuron, to instead be a fraction of dataset type of statistic. 
    figsize - tuple of int
        Size of the figure to produce. Passed directly to plt.subplots
    literal_path - bool
        If true, take the path and do not modify it. If False, prepend the project_dir path
    count_label - bool
        If true, put as an anotation the counts of items in that neuron plot
    '''
    # Need access to the Normalise and ColorbarBase objects
    import matplotlib as mpl
    # Step one, get unique items and their counts
    from collections import Counter
    
    unique_labels = []
    for k, v in book.items():
        v = [i for items in v for i in items]    
        unique_labels += v
    unique_labels = list(set(unique_labels))
    unique_labels.sort()
    
    max_val = 0
    plt_book = {}
    
    for k, v in book.items():
        v = [i for items in v for i in items]
        c = Counter(v)

        # Guard agaisnt empty most similar neuron
        if len(v) > 0:
            if weights is not None:
                plt_book[k] = { label: c[label] / weights[label] for label in unique_labels }
            else:
                plt_book[k] = { label: c[label] / len(v) for label in unique_labels }
                
            mv = max(plt_book[k].values())
            max_val = mv if mv > max_val else max_val
        else:
            plt_book[k] = None

    if weights is not None:
        cb_label = 'Fraction of Dataset'
        norm = mpl.colors.Normalize(vmin=0, vmax=max_val)
    else:
        cb_label = 'Fraction per Neuron'
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

    cmap = plt.get_cmap(color_map)
    
    fig, ax = plt.subplots(nrows=shape[0]+1, ncols=shape[1]+1, figsize=figsize)

    # Set empty axis labels for everything
    for a in ax.flatten():
        a.set(xticklabels=[], yticklabels=[])

    for k, v in plt_book.items():
        if v is None:
            continue

        color = cmap(norm(list(v.values())))

        ax[k].bar(np.arange(len(unique_labels)),
                 [1]*len(unique_labels),
                 color=color,
                 align='center',
                 tick_label=unique_labels)

        ax[k].set(ylim=[0,1])

        if k[1] != -1: # disable this for now.
            ax[k].set(yticklabels=[])
        if k[0] != shape[1]:
            ax[k].set(xticklabels=[])
        else:
            if xtick_rotation is not None:
                ax[k].tick_params(axis='x', rotation=xtick_rotation)
                for item in ax[k].get_xticklabels():
                    item.set_fontsize(7.5)
        
        if count_text:
            v = [i for items in book[k] for i in items]
            bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
            if isinstance(count_text, tuple):
                ax[k].annotate(f'{len(v)}', xy=count_text, xycoords='axes fraction', bbox=bbox_props)
            else:
                ax[k].annotate(f'{len(v)}', xy=(0.1,0.65), xycoords='axes fraction', bbox=bbox_props)
                
        
    fig.subplots_adjust(right=0.83)
    cax = fig.add_axes([0.85, 0.10, 0.03, 0.8])
     
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, label=cb_label)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

def FIRST_Fraction(CHANNELS=[['FIRST']],
                   PROJECT_DIR = 'Script_Experiments_Fraction',
                   TRIAL=0):
    REALISATIONS = 1000
    BINARY_OPTS = [{'fraction':0.7, 'norm':False, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':'NoNorm_NoLog_NoSig'},
                   {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':f'Norm_NoLog_NoSig'},
                   {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':3., 'convex':False, 'project_dir':f'Norm_NoLog_3'},
                   {'fraction':0.7, 'norm':True, 'log10': [True,False], 'sigma':3., 'convex':False, 'project_dir':f'Norm_Log_3'},
                   {'fraction':0.7, 'norm':True, 'log10': [True, False], 'sigma':3., 'convex':True, 'project_dir':f'Norm_Log_3_Convex'}]

    PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':10},
                 {'som-width':7, 'som-height':7, 'num-iter':10},
                 {'som-width':10, 'som-height':10, 'num-iter':10},
                 {'som-width':12, 'som-height':12, 'num-iter':10}]

    
    for bin_opts, pink_opts, channels in product(BINARY_OPTS, PINK_OPTS, CHANNELS):
        # print(bin_opts, pink_opts)

        # Make sure there are enough channels to do the mask with
        if len(CHANNELS[0]) == 1 and bin_opts['convex']:
            print(f'Skipping this option set. CHANNELS is {CHANNELS[0]}, nothing to apply convex hull masking to. ')
            continue

        chan_name = '_'.join(channels)
        out_dir = f"{PROJECT_DIR}/{chan_name}_{bin_opts['project_dir']}_{pink_opts['som-width']}x{pink_opts['som-height']}_Trial{TRIAL}"
        results = []
        
        if not os.path.exists(f'{out_dir}/trained.pink'):
            print(f'PINK not found: {out_dir}')
        else:
            print(f'\n\nLoading in saved Pink instance: {out_dir}')
            pink = Pink.loader(f'{out_dir}/trained.pink')

            print('\tSOM Channel 1')
            pink.show_som(channel=0, mode=i)
            pink.show_som(channel=0, mode=i, plt_mode='split')
            pink.show_som(channel=0, mode=i, plt_mode='grid')
            print('\tSOM Channel 1')
            pink.show_som(channel=1, mode=i)
            pink.show_som(channel=1, mode=i, plt_mode='split')
            pink.show_som(channel=1, mode=i, plt_mode='grid')

            print('\tAttribute heatmaps')
            book, counts = pink.attribute_heatmap(mode='train', plot=False, realisations=100)
            label_plot(book, get_shape(book), color_map='Blues', weights=counts, xtick_rotation=90, count_text=True, figsize=(10,10),
                       save=f'{pink.project_dir}/Label_Dist_Weighted_100.png')
            label_plot(book, get_shape(book), color_map='Blues', xtick_rotation=90, count_text=True, figsize=(10,10),
                       save=f'{pink.project_dir}/Label_Dist_100.png')
            
        # for i, t in enumerate(pink.binary):
        #     try:
        #         pink.map(mode=i)   
        #         pink.map(mode='validate', SOM_mode=i)   
                
        #         pink.save('trained.pink')

        #         pink.show_som(channel=0, mode=i)
        #         pink.show_som(channel=0, mode=i, plt_mode='split')
        #         pink.show_som(channel=0, mode=i, plt_mode='grid')
        #         pink.show_som(channel=1, mode=i)
        #         pink.show_som(channel=1, mode=i, plt_mode='split')
        #         pink.show_som(channel=1, mode=i, plt_mode='grid')
        #         pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i)
        #         pink.attribute_heatmap(save=f'train_{i}_MC{REALISATIONS}_labels_dist.pdf', mode=i, realisations=REALISATIONS)

        #         pink.count_map(mode=i, save=f'train_{i}_count_map.pdf')
        #         pink.count_map(mode='validate', SOM_mode=i, save=f'validate_{i}_count_map.pdf')

        #         validation_res = pink.validator(SOM_mode=i)

        #         results.append(validation_res)


            # except Exception as e:
            #     print('Try caught something')
            #     print(e)

                # import traceback
                # traceback.print_exc()

            plt.close('all')

            # df = pd.DataFrame(results)
            # df.to_json(f'{pink.project_dir}/FIRST_Results.json')
              

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()

    batch1 = [i for i in range(0,10)]
    
    SEGS_DIR = 'Script_Experiments_Segments_Trials'
    FRAC_DIR = 'Script_Experiments_Fractions_Trials'
    
    # ----------------------------------------------------------
    for i in batch1:
        FIRST_Fraction(TRIAL=i, PROJECT_DIR=FRAC_DIR)
        FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
                        TRIAL=i, PROJECT_DIR=FRAC_DIR)
    


