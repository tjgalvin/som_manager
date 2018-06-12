import matplotlib as mpl
mpl.use('agg')

import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from catalog import Source, Binary, Catalog, Pink
from experiment_options import BINARY_OPTS, PINK_OPTS, LEARNING_MODES

def FIRST_Fraction(CHANNELS=[['FIRST']],
                   PROJECT_DIR = 'Script_Experiments_Fraction_Learning',
                   TRIAL=0):

    rgz_dir = 'rgz_rcnn_data'
    cat = Catalog(rgz_dir=rgz_dir)

    print('\nValidating sources...')
    cat.collect_valid_sources()
    
    for bin_opts, pink_opts, channels, learning in product(BINARY_OPTS, PINK_OPTS, CHANNELS, LEARNING_MODES):
        print(bin_opts, pink_opts)

        # Make sure there are enough channels to do the mask with
        if len(CHANNELS[0]) == 1 and bin_opts['convex']:
            print(f'Skipping this option set. CHANNELS is {CHANNELS[0]}, nothing to apply convex hull masking to. ')
            continue

        chan_name = '_'.join(channels)
        out_dir = f"{PROJECT_DIR}/{chan_name}_{bin_opts['project_dir']}_{pink_opts['som-width']}x{pink_opts['som-height']}_Trial{TRIAL}"
        results = []
        
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
            pink.train(learning=learning)
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
                
            except Exception as e:
                print('Try caught something')
                print(e)

                import traceback
                traceback.print_exc()

            plt.close('all')

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()

    batch1 = [i for i in range(0,5)]
    batch2 = [i for i in range(5,10)]

    FRAC_DIR = 'Script_Experiments_Fractions_Trials_Learning'
    
    # ----------------------------------------------------------
    if 'bd-client-01' in hostname:
        for i in batch1:
            FIRST_Fraction(TRIAL=i, PROJECT_DIR=FRAC_DIR)
        FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
                        TRIAL=4, PROJECT_DIR=FRAC_DIR)
    # ----------------------------------------------------------
    elif 'bd-client-02' in hostname:
        for i in batch2:
            FIRST_Fraction(TRIAL=i, PROJECT_DIR=FRAC_DIR)
        # FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
        #             TRIAL=9, PROJECT_DIR=FRAC_DIR)
    # ----------------------------------------------------------
    elif 'bd-client-03' in hostname:
        for i in batch1:
            FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
                           TRIAL=i, PROJECT_DIR=FRAC_DIR)

    # ----------------------------------------------------------
    elif 'bd-client-04' in hostname:
        for i in batch2:
            FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
                           TRIAL=i, PROJECT_DIR=FRAC_DIR)
            
    else:
        print('No matching hostname...')


# OLD MODE OF PROCESS
    # # ----------------------------------------------------------
    # if 'bd-client-01' in hostname:
    #     for i in batch1:
    #         FIRST_Segments(TRIAL=i, PROJECT_DIR=SEGS_DIR)

    #     for i in batch2:
    #         FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
    #                        TRIAL=i, PROJECT_DIR=FRAC_DIR)

    # # ----------------------------------------------------------
    # elif 'bd-client-02' in hostname:
    #     for i in batch1:
    #         FIRST_Fraction(TRIAL=i, PROJECT_DIR=FRAC_DIR)

    #     for i in batch2:
    #         FIRST_Segments(CHANNELS=[['FIRST','WISE_W1']],
    #                        TRIAL=i, PROJECT_DIR=SEGS_DIR)
    
    # # ----------------------------------------------------------
    # elif 'bd-client-03' in hostname:
    #     for i in batch1:
    #         FIRST_Fraction(CHANNELS=[['FIRST','WISE_W1']],
    #                        TRIAL=i, PROJECT_DIR=FRAC_DIR)

    #     for i in batch2:
    #         FIRST_Segments(TRIAL=i, PROJECT_DIR=SEGS_DIR)

    # # ----------------------------------------------------------
    # elif 'bd-client-04' in hostname:
    #     for i in batch1:
    #         FIRST_Segments(CHANNELS=[['FIRST','WISE_W1']],
    #                        TRIAL=i, PROJECT_DIR=SEGS_DIR)

    #     for i in batch2:
    #         FIRST_Fraction(TRIAL=i, PROJECT_DIR=FRAC_DIR)
            
    # else:
    #     print('No matching hostname...')