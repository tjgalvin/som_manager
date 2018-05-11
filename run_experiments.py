import matplotlib as mpl
mpl.use('agg')

import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from catalog import Source, Binary, Catalog, Pink

def FIRST_Fraction(CHANNELS=[['FIRST']],
                   PROJECT_DIR = 'Script_Experiments_Fraction',
                   TRIAL=0):
    REALISATIONS = 1000
    BINARY_OPTS = [{'fraction':0.7, 'norm':False, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':'NoNorm_NoLog_NoSig'},
                   {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':f'Norm_NoLog_NoSig'},
                   {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':3., 'convex':False, 'project_dir':f'Norm_NoLog_3'},
                   {'fraction':0.7, 'norm':True, 'log10': [True, False], 'sigma':3., 'convex':True, 'project_dir':f'Norm_Log_3_Convex'}]

    PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':10},
                 {'som-width':7, 'som-height':7, 'num-iter':10},
                 {'som-width':10, 'som-height':10, 'num-iter':10},
                 {'som-width':13, 'som-height':13, 'num-iter':10},
                 {'som-width':16, 'som-height':16, 'num-iter':10}]

    rgz_dir = 'rgz_rcnn_data'
    cat = Catalog(rgz_dir=rgz_dir)

    print('\nValidating sources...')
    cat.collect_valid_sources()

    results = []
    
    for bin_opts, pink_opts, channels in product(BINARY_OPTS, PINK_OPTS, CHANNELS):
        print(bin_opts, pink_opts)

        # Make sure there are enough channels to do the mask with
        if len(CHANNELS) == 1 and bin_opts['convex']:
            print(f'Skipping this option set. CHANNELS is {CHANNELS}, nothing to apply convex hull masking to. ')
            continue

        chan_name = '_'.join(channels)
        out_dir = f"{PROJECT_DIR}/{chan_name}_{bin_opts['project_dir']}_{pink_opts['som-width']}x{pink_opts['som-height']}_Trial{TRIAL}"

        if not os.path.exists(f'{out_dir}/trained.pink'):
            # This is painful, but since product() is returning a reference to a dict, we cant
            # edit the project_dir in place to build up the folder name, as this gets carried
            # through to later iterations. Hence, we can't **bin_opts below
            bins = cat.dump_binary('source.binary', channels=channels, project_dir=out_dir,
                                norm=bin_opts['norm'], sigma=bin_opts['sigma'], log10=bin_opts['log10'],
                                convex=bin_opts['convex'], fraction=bin_opts['fraction'])

            train_bin, validate_bin = bins
            
            pink = Pink(train_bin, 
                        pink_args=pink_opts,
                        validate_binary=validate_bin) 
            pink.train()
        else:
            print('Loading in saved Pink instance')
            pink = Pink.loader(f'{out_dir}/trained.pink')

        for i, t in enumerate(pink.binary):
            try:
                pink.map(mode=i)   
                pink.map(mode='validate', SOM_mode=i)   
                
                pink.save('trained.pink')

                pink.show_som(channel=0, mode=i)
                pink.show_som(channel=0, mode=i, plt_mode='split')
                pink.show_som(channel=0, mode=i, plt_mode='grid')
                pink.show_som(channel=1, mode=i)
                pink.show_som(channel=1, mode=i, plt_mode='split')
                pink.show_som(channel=1, mode=i, plt_mode='grid')
                pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i)
                pink.attribute_heatmap(save=f'train_{i}_MC{REALISATIONS}_labels_dist.pdf', mode=i, realisations=REALISATIONS)

                pink.count_map(mode=i, save=f'train_{i}_count_map.pdf')
                pink.count_map(mode='validate', SOM_mode=i, save=f'validate_{i}_count_map.pdf')

                validation_res = pink.validator(SOM_mode=i)

                results.append(validation_res)


            except Exception as e:
                print('Try caught something')
                print(e)

                import traceback
                traceback.print_exc()

            plt.close('all')

            df = pd.DataFrame(results)
            df.to_json(f'{pink.project_dir}/FIRST_Results.json')
              
def FIRST_Segments(CHANNELS=[['FIRST']],
                   PROJECT_DIR = 'Script_Experiments_Segments',
                   TRIAL=0):
    REALISATIONS = 1000

    BINARY_OPTS = [{'segments':4, 'norm':False, 'log10': False, 'sigma':False, 'convex':False, 'project_dirs':'NoNorm_NoLog_NoSig'},
                   {'segments':4, 'norm':True, 'log10': False, 'sigma':False,'convex':False,  'project_dirs':f'Norm_NoLog_NoSig'},
                   {'segments':4, 'norm':True, 'log10': False, 'sigma':3., 'convex':False, 'project_dirs':f'Norm_NoLog_3'},
                   {'segments':4, 'norm':True, 'log10': [True, False], 'sigma':3., 'convex':True, 'project_dirs':f'Norm_Log_3_Convex'}]

    PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':10},
                 {'som-width':7, 'som-height':7, 'num-iter':10},
                 {'som-width':10, 'som-height':10, 'num-iter':10},
                 {'som-width':13, 'som-height':13, 'num-iter':10},
                 {'som-width':16, 'som-height':16, 'num-iter':10}]

    rgz_dir = 'rgz_rcnn_data'
    cat = Catalog(rgz_dir=rgz_dir)

    print('\nValidating sources...')
    cat.collect_valid_sources()

    results = []
    
    for bin_opts, pink_opts, channels in product(BINARY_OPTS, PINK_OPTS, CHANNELS):
        print(bin_opts, pink_opts)

        # Make sure there are enough channels to do the mask with
        if len(CHANNELS) == 1 and bin_opts['convex']:
            print(f'Skipping this option set. CHANNELS is {CHANNELS}, nothing to apply convex hull masking to. ')
            continue

        chan_name = '_'.join(channels)
        out_dir = f"{PROJECT_DIR}/{chan_name}_{bin_opts['project_dirs']}_{pink_opts['som-width']}x{pink_opts['som-height']}_Trial{TRIAL}"

        if not os.path.exists(f'{out_dir}/trained.pink'):

            # This is painful, but since product() is returning a reference to a dict, we cant
            # edit the project_dir in place to build up the folder name, as this gets carried
            # through to later iterations. Hence, we can't **bin_opts below
            bins = cat.dump_binary('source.binary', channels=channels, project_dir=out_dir,
                                norm=bin_opts['norm'], sigma=bin_opts['sigma'], log10=bin_opts['log10'],
                                convex=bin_opts['convex'], segments=bin_opts['segments'])

            train_bin, validate_bin = bins
            
            pink = Pink(train_bin, 
                        pink_args=pink_opts,
                        validate_binary=validate_bin) 
            pink.train()
        else:
            pink = Pink.loader(f'{out_dir}/trained.pink')

        for i, t in enumerate(pink.binary):
            try:
                pink.map(mode=i)   
                pink.map(mode='validate', SOM_mode=i)   
                
                pink.save('trained.pink')
                
                pink.show_som(channel=0, mode=i)
                pink.show_som(channel=0, mode=i, plt_mode='split')
                pink.show_som(channel=0, mode=i, plt_mode='grid')
                pink.show_som(channel=1, mode=i)
                pink.show_som(channel=1, mode=i, plt_mode='split')
                pink.show_som(channel=1, mode=i, plt_mode='grid')
                pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i)
                pink.attribute_heatmap(save=f'train_{i}_MC{REALISATIONS}_labels_dist.pdf', mode=i, realisations=REALISATIONS)
                
                pink.count_map(mode=i, save=f'train_{i}_count_map.pdf')
                pink.count_map(mode='validate', SOM_mode=i, save=f'validate_{i}_count_map.pdf')

                validation_res = pink.validator(SOM_mode=i)

                results.append(validation_res)

            except Exception as e:
                print('Try pass captured something')
                print(e)

                import traceback
                traceback.print_exc()

            plt.close('all')

            df = pd.DataFrame(results)
            df.to_json(f'{pink.project_dir}/FIRST_Results.json')

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()

    batch1 = [i for i in range(0,5)]
    batch2 = [i for i in range(5,10)]

    SEGS_DIR = 'Script_Experiments_Segments_Trials'
    FRAC_DIR = 'Script_Experiments_Fractions_Trials'
    
    # ----------------------------------------------------------
    if 'bd-client-01' in hostname:
        for i in batch1:
            FIRST_Segments(TRIAL=i, PROJECT_DIR=SEGS_DIR)

        for i in batch2:
            FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
                           TRIAL=i, PROJECT_DIR=FRAC_DIR)

    # ----------------------------------------------------------
    elif 'bd-client-02' in hostname:
        for i in batch1:
            FIRST_Fraction(TRIAL=i, PROJECT_DIR=FRAC_DIR)

        for i in batch2:
            FIRST_Segments(CHANNELS=[['FIRST','WISE_W1']],
                           TRIAL=i, PROJECT_DIR=SEGS_DIR)
    
    # ----------------------------------------------------------
    elif 'bd-client-03' in hostname:
        for i in batch1:
            FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
                           TRIAL=i, PROJECT_DIR=FRAC_DIR)

        for i in batch2:
            FIRST_Segments(TRIAL=i, PROJECT_DIR=SEGS_DIR)

    # ----------------------------------------------------------
    elif 'bd-client-04' in hostname:
        for i in batch1:
            FIRST_Segments(CHANNELS=[['FIRST','WISE_W1']],
                           TRIAL=i, PROJECT_DIR=SEGS_DIR)

        for i in batch2:
            FIRST_Fraction(TRIAL=i, PROJECT_DIR=FRAC_DIR)
            
    else:
        print('No matching hostname...')