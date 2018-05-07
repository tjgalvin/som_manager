import pandas as pd
from catalog import Source, Binary, Catalog, Pink
from itertools import product
import matplotlib.pyplot as plt


def FIRST_Fraction():
    PROJECT_DIR = 'Script_Experiments_Fraction'
    CHANNELS = [['FIRST'], ['FIRST', 'WISE_W1']]
    # CHANNELS = [['FIRST']]
    BINARY_OPTS = [{'fraction':0.7, 'norm':False, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':'NoNorm_NoLog_NoSig'},
                   {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':f'Norm_NoLog_NoSig'},
                   {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':3., 'convex':False, 'project_dir':f'Norm_NoLog_3'},
                   {'fraction':0.7, 'norm':True, 'log10': [True, False], 'sigma':3., 'convex':True, 'project_dir':f'Norm_Log_3'}]

    # BINARY_OPTS = [{'fraction':0.8, 'norm':True, 'log10': False, 'sigma':3., 'project_dir':f'Norm_NoLog_3'},
    #                {'fraction':0.8, 'norm':True, 'log10': True, 'sigma':3., 'project_dir':f'Norm_Log_3'}]


    PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':1},
                 {'som-width':7, 'som-height':7, 'num-iter':1},
                 {'som-width':10, 'som-height':10, 'num-iter':1},
                 {'som-width':13, 'som-height':13, 'num-iter':1}]

    # PINK_OPTS = [{'som-width':2, 'som-height':2, 'num-iter':1},
    #              {'som-width':3, 'som-height':3, 'num-iter':1}]


    rgz_dir = 'rgz_rcnn_data'
    cat = Catalog(rgz_dir=rgz_dir)

    # Commenting out to let me ctrl+C without killing things
    # cat.save_sources()

    print('\nValidating sources...')
    cat.collect_valid_sources()

    results = []
    
    for bin_opts, pink_opts, channels in product(BINARY_OPTS, PINK_OPTS, CHANNELS):
        print(bin_opts, pink_opts)

        chan_name = '_'.join(channels)
        out_dir = f"{PROJECT_DIR}/{chan_name}_{bin_opts['project_dir']}_{pink_opts['som-width']}x{pink_opts['som-height']}"

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
        for i, t in enumerate(pink.binary):
            pink.map(mode=i)   
            pink.map(mode='validate', SOM_mode=i)   
            
            pink.show_som(channel=0, mode=i)
            pink.show_som(channel=0, mode=i, plt_mode='split')
            pink.show_som(channel=0, mode=i, plt_mode='grid')
            pink.show_som(channel=1, mode=i)
            pink.show_som(channel=1, mode=i, plt_mode='split')
            pink.show_som(channel=1, mode=i, plt_mode='grid')
            pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i)
            pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i, realisations=1000)
            
            validation_res = pink.validator(SOM_mode=i)

            results.append(validation_res)

            pink.save('trained.pink')

            plt.close('all')

            df = pd.DataFrame(results)
            df.to_json(f'{pink.project_dir}/FIRST_Results.json')
        
        
def FIRST_Segments():
    PROJECT_DIR = 'Script_Experiments'
    CHANNELS = [['FIRST'], ['FIRST', 'WISE_W1']]
    # CHANNELS = [['FIRST']]
    BINARY_OPTS = [{'segments':4, 'norm':False, 'log10': False, 'sigma':False, 'convex':False, 'project_dirs':'NoNorm_NoLog_NoSig'},
                   {'segments':4, 'norm':True, 'log10': False, 'sigma':False,'convex':False,  'project_dirs':f'Norm_NoLog_NoSig'},
                   {'segments':4, 'norm':True, 'log10': False, 'sigma':3., 'convex':False, 'project_dirs':f'Norm_NoLog_3'},
                   {'segments':4, 'norm':True, 'log10': [True, False], 'sigma':3., 'convex':True, 'project_dirs':f'Norm_Log_3'}]

    # BINARY_OPTS = [{'segments':4, 'norm':True, 'log10': False, 'sigma':3., 'project_dir':f'Norm_NoLog_3'},
    #                {'segments':4, 'norm':True, 'log10': True, 'sigma':3., 'project_dir':f'Norm_Log_3'}]


    PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':1},
                 {'som-width':7, 'som-height':7, 'num-iter':1},
                 {'som-width':10, 'som-height':10, 'num-iter':1},
                 {'som-width':13, 'som-height':13, 'num-iter':1}]

    # PINK_OPTS = [{'som-width':2, 'som-height':2, 'num-iter':1},
    #              {'som-width':3, 'som-height':3, 'num-iter':1}]


    rgz_dir = 'rgz_rcnn_data'
    cat = Catalog(rgz_dir=rgz_dir)

    # Commenting out to let me ctrl+C without killing things
    # cat.save_sources()

    print('\nValidating sources...')
    cat.collect_valid_sources()

    results = []
    
    for bin_opts, pink_opts, channels in product(BINARY_OPTS, PINK_OPTS, CHANNELS):
        print(bin_opts, pink_opts)

        chan_name = '_'.join(channels)
        out_dir = f"{PROJECT_DIR}/{chan_name}_{bin_opts['project_dirs']}_{pink_opts['som-width']}x{pink_opts['som-height']}"

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
        for i, t in enumerate(pink.binary):
            
            pink.map(mode=i)   
            pink.map(mode='validate', SOM_mode=i)   
            
            pink.show_som(channel=0, mode=i)
            pink.show_som(channel=0, mode=i, plt_mode='split')
            pink.show_som(channel=0, mode=i, plt_mode='grid')
            pink.show_som(channel=1, mode=i)
            pink.show_som(channel=1, mode=i, plt_mode='split')
            pink.show_som(channel=1, mode=i, plt_mode='grid')
            pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i)
            pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i, realisations=1000)
            
            pink.count_map(mode=i, save=f'train_{i}_count_map.pdf')
            pink.count_map(mode='validate', SOM_mode=i, save=f'validate_{i}_count_map.pdf')

            validation_res = pink.validator(SOM_mode=i)

            results.append(validation_res)

            pink.save('trained.pink')

            plt.close('all')

            df = pd.DataFrame(results)
            df.to_json(f'{pink.project_dir}/FIRST_Results.json')

if __name__ == '__main__':
    FIRST_Fraction()
    # FIRST_Segments()