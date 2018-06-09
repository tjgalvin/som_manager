BINARY_OPTS = [{'fraction':0.7, 'norm':False, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':'NoNorm_NoLog_NoSig'},
                {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':False, 'convex':False, 'project_dir':f'Norm_NoLog_NoSig'},
                {'fraction':0.7, 'norm':True, 'log10': False, 'sigma':3., 'convex':False, 'project_dir':f'Norm_NoLog_3'},
                {'fraction':0.7, 'norm':True, 'log10': [True,False], 'sigma':3., 'convex':False, 'project_dir':f'Norm_Log_3'},
                {'fraction':0.7, 'norm':True, 'log10': [True, False], 'sigma':3., 'convex':True, 'project_dir':f'Norm_Log_3_Convex'}]

PINK_OPTS = [{'som-width':3, 'som-height':3, 'num-iter':1, 'inter-store':'keep', 'progress':0.05},
                {'som-width':7, 'som-height':7, 'num-iter':1, 'inter-store':'keep', 'progress':0.05},
                {'som-width':10, 'som-height':10, 'num-iter':1, 'inter-store':'keep', 'progress':0.05},
                {'som-width':12, 'som-height':12, 'num-iter':1, 'inter-store':'keep', 'progress':0.05}]

LEARNING_MODES = [  [('gaussian','3.0','0.1'),
                     ('gaussian','1.4','0.05'),
                     ('gaussian','0.3','0.01')]
                ]
