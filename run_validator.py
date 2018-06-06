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

            for i, t in enumerate(pink.binary):
                
                print('\tRunning Validator')
                pink.weight_test(SOM_mode=i, realisations=100)
                result = pink.validator(SOM_mode=i, realisations=100, weights=True, pack=True)                
                print('\t', result['total_accuracy'])
                results.append(result)

                result = pink.validator(SOM_mode=i, realisations=100, pack=True)                
                print('\t', result['total_accuracy'])
                results.append(result)

                result = pink.validator(SOM_mode=i, pack=True)                
                print('\t', result['total_accuracy'], result['total_correct']+result['total_wrong'])
                results.append(result)

                result = pink.prob_validator(SOM_mode=i, realisations=100, weights=True, pack=True)                
                print('\t', result['total_accuracy'], result['total_correct']+result['total_wrong'])
                results.append(result)
                
                # result = pink.validator(SOM_mode=i, realisations=100)                
                # print(result['total_accuracy'], result['total_correct']+result['total_wrong'])

                # result = pink.validator(SOM_mode=i, realisations=100, weights=True)                
                # print(result['total_accuracy'], result['total_correct']+result['total_wrong'])

            df = pd.DataFrame(results)
            df.to_json(f'{pink.project_dir}/Weighted_Results.json')

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
    


