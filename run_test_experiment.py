import matplotlib as mpl
mpl.use('agg')

import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from catalog import Source, Binary, Catalog, Pink

def main():
    test_dir = 'Test_Experiment'
    proj_dir = 'FIRST_Norm_Log_3'
    pink_opts = {'som-width':10, 'som-height':10, 'num-iter':1, 'inter-store':'keep'}

    rgz_dir = 'rgz_rcnn_data'
    cat = Catalog(rgz_dir=rgz_dir)

    print('\nValidating sources...')
    cat.collect_valid_sources()

    out_dir = f"{test_dir}/FIRST_{proj_dir}_{pink_opts['som-width']}x{pink_opts['som-height']}_10_iter"
    if not os.path.exists(f'{out_dir}/trained.pink'):
        bins = cat.dump_binary('source.binary', channels=['FIRST'], project_dir=out_dir,
                               norm=True, sigma=3., log10=True, fraction=0.75)

        train_bin, validate_bin = bins

        pink = Pink(train_bin, 
                    pink_args=pink_opts,
                    validate_binary=validate_bin) 
        pink.train(learning=[('gaussian','3.0','0.1'),
                             ('gaussian','1.4','0.05'),
                             ('gaussian','0.3','0.01')])
        pink.map()
        pink.map(mode='validate', SOM_mode='train')
        pink.save('trained.pink')

    pink_opts = {'som-width':15, 'som-height':15, 'num-iter':1, 'inter-store':'keep'}
    
    out_dir = f"{test_dir}/FIRST_{proj_dir}_{pink_opts['som-width']}x{pink_opts['som-height']}_20_iter"
    if not os.path.exists(f'{out_dir}/trained.pink'):
        bins = cat.dump_binary('source.binary', channels=['FIRST'], project_dir=out_dir,
                        norm=True, sigma=2., log10=True, fraction=0.75)

        train_bin, validate_bin = bins

        pink = Pink(train_bin, 
                    pink_args=pink_opts,
                    validate_binary=validate_bin) 
        pink.train(learning=[('gaussian','3.0','0.1'),
                             ('gaussian','1.4','0.05'),
                             ('gaussian','0.3','0.01')])
        pink.map()
        pink.map(mode='validate', SOM_mode='train')
        pink.save('trained.pink')

    # pink_opts = {'som-width':10, 'som-height':10, 'num-iter':1, 'inter-store':'keep'}
    
    # out_dir = f"{test_dir}/FIRST_{proj_dir}_{pink_opts['som-width']}x{pink_opts['som-height']}_30_iter"
    # if not os.path.exists(f'{out_dir}/trained.pink'):
    #     bins = cat.dump_binary('source.binary', channels=['FIRST'], project_dir=out_dir,
    #                     norm=True, sigma=2., log10=True, fraction=0.75)

    #     train_bin, validate_bin = bins

    #     pink = Pink(train_bin, 
    #                 pink_args=pink_opts,
    #                 validate_binary=validate_bin) 
    #     pink.train(learning=[('gaussian','3.0','0.1'),
    #                          ('gaussian','1.4','0.05'),
    #                          ('gaussian','0.3','0.01')])
    #     pink.map()
    #     pink.map(mode='validate', SOM_mode='train')
    #     pink.save('trained.pink')


# ---------------------------------------------------------
    proj_dir = 'FIRST_WISE_Norm_Log_3'

    pink_opts = {'som-width':10, 'som-height':10, 'num-iter':1, 'progress':0.05, 'inter-store':'keep'}

    out_dir = f"{test_dir}/{proj_dir}_{pink_opts['som-width']}x{pink_opts['som-height']}_10_iter"
    if not os.path.exists(f'{out_dir}/trained.pink'):
        bins = cat.dump_binary('source.binary', channels=['FIRST', 'WISE_W1'], project_dir=out_dir,
                               norm=True, sigma=3., log10=True, fraction=0.75)

        train_bin, validate_bin = bins

        pink = Pink(train_bin, 
                    pink_args=pink_opts,
                    validate_binary=validate_bin) 
        pink.train(learning=[('gaussian','3.0','0.1'),
                             ('gaussian','1.4','0.05'),
                             ('gaussian','0.3','0.01')])
        pink.map()
        pink.map(mode='validate', SOM_mode='train')
        pink.save('trained.pink')


    pink_opts = {'som-width':15, 'som-height':15, 'num-iter':1, 'progress':0.05, 'inter-store':'keep'}
    
    out_dir = f"{test_dir}/{proj_dir}_{pink_opts['som-width']}x{pink_opts['som-height']}_20_iter"
    if not os.path.exists(f'{out_dir}/trained.pink'):
        bins = cat.dump_binary('source.binary', channels=['FIRST', 'WISE_W1'], project_dir=out_dir,
                        norm=True, sigma=2., log10=True, fraction=0.75)

        train_bin, validate_bin = bins

        pink = Pink(train_bin, 
                    pink_args=pink_opts,
                    validate_binary=validate_bin) 
        pink.train(learning=[('gaussian','3.0','0.1'),
                             ('gaussian','1.4','0.05'),
                             ('gaussian','0.3','0.01')])
        pink.map()
        pink.map(mode='validate', SOM_mode='train')
        pink.save('trained.pink')

    # pink_opts = {'som-width':10, 'som-height':10, 'num-iter':1, 'progress':0.05, 'inter-store':'keep'}
    
    # out_dir = f"{test_dir}/{proj_dir}_{pink_opts['som-width']}x{pink_opts['som-height']}_30_iter"
    # if not os.path.exists(f'{out_dir}/trained.pink'):
    #     bins = cat.dump_binary('source.binary', channels=['FIRST', 'WISE_W1'], project_dir=out_dir,
    #                     norm=True, sigma=3., log10=True, fraction=0.75)

    #     train_bin, validate_bin = bins

    #     pink = Pink(train_bin, 
    #                 pink_args=pink_opts,
    #                 validate_binary=validate_bin) 
    #     pink.train(learning=[('gaussian','3.0','0.1'),
    #                          ('gaussian','1.4','0.05'),
    #                          ('gaussian','0.3','0.01')])
    #     pink.map()
    #     pink.map(mode='validate', SOM_mode='train')
    #     pink.save('trained.pink')



if __name__ == '__main__':
    main()