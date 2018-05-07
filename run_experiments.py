import pandas as pd
from catalog import Source, Binary, Catalog, Pink
from itertools import product
import matplotlib.pyplot as plt


def FIRST():
    PROJECT_DIR = 'Script_Experiments'
    CHANNELS = [['FIRST'], ['FIRST', 'WISE_W1']]
    CHANNELS = [['FIRST']]
    BINARY_OPTS = [{'segments':4, 'norm':False, 'log10': False, 'sigma':False, 'project_dir':'NoNorm_NoLog_NoSig'},
                   {'segments':4, 'norm':True, 'log10': False, 'sigma':False, 'project_dir':f'Norm_NoLog_NoSig'},
                   {'segments':4, 'norm':True, 'log10': False, 'sigma':3., 'project_dir':f'Norm_NoLog_3'},
                   {'segments':4, 'norm':True, 'log10': True, 'sigma':3., 'project_dir':f'Norm_Log_3'}]

    BINARY_OPTS = [{'segments':4, 'norm':True, 'log10': False, 'sigma':3., 'project_dir':f'Norm_NoLog_3'},
                   {'segments':4, 'norm':True, 'log10': True, 'sigma':3., 'project_dir':f'Norm_Log_3'}]


    PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':1},
                 {'som-width':7, 'som-height':7, 'num-iter':1},
                 {'som-width':10, 'som-height':10, 'num-iter':1},
                 {'som-width':13, 'som-height':13, 'num-iter':1}]

    PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':1},
                 {'som-width':7, 'som-height':7, 'num-iter':1}]


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
        project_dir = f"{PROJECT_DIR}/{chan_name}_{bin_opts['project_dir']}_{pink_opts['som-width']}x{pink_opts['som-height']}"

        print(project_dir)

        bins = cat.dump_binary('source.binary', channels=channels,
                               project_dir=project_dir,
                               segments=4)

        train_bin, validate_bin = bins
        
        pink = Pink(train_bin, 
                    pink_args=pink_opts,
                    validate_binary=validate_bin) 
        pink.train()
        for i, t in enumerate(pink.binary):
            
            # Bug here somewhere. Not each train being compared agaisnt validate
            # letting run to completion to see for other bugs.. Since the map from
            # the vlidation has already been made on train 0, subsequent trains 1,2,3
            # cant be made since the file already exists
            pink.map(mode=i)   
            pink.map(mode='validate', SOM_mode=i)   
            # ------------------------------------------------------------------

            # pink.show_som(channel=0, mode=i)
            # pink.show_som(channel=0, mode=i, plt_mode='split')
            # pink.show_som(channel=0, mode=i, plt_mode='grid')
            # pink.show_som(channel=1, mode=i)
            # pink.show_som(channel=1, mode=i, plt_mode='split')
            # pink.show_som(channel=1, mode=i, plt_mode='grid')
            # pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i)
            # pink.attribute_heatmap(save=f'train_{i}_labels_dist.pdf', mode=i, realisations=1000)
            
            validation_res = pink.validator(SOM_mode=i)

            results.append(validation_res)

            plt.close('all')

        df = pd.DataFrame(results)
        df.to_json('FIRST_Results.json')

    # rgz_dir = 'rgz_rcnn_data'
    # cat = Catalog(rgz_dir=rgz_dir)

    # # Commenting out to let me ctrl+C without killing things
    # # cat.save_sources()

    # print('\nValidating sources...')
    # cat.collect_valid_sources()


if __name__ == '__main__':
    FIRST()