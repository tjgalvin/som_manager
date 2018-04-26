'''A simple script to test to see if the RGZ_RCNN matches one to one with the images downloaded from the down_images script
'''

import glob
import os

# Get a list of objects from the rgz_rcnn directory
rgz_dir = 'rgz_rcnn_data'

scan_str = f'{rgz_dir}/RGZdevkit2017/RGZ2017/PNGImages/*_logminmax.png'
print(f'Globbing for files with: {scan_str}')
files = glob.glob(scan_str)

print(f'Located {len(files)} files...')

first_names = [ i.split('/')[-1].split('_')[0].replace('FIRST','') for i in files]

stamps_dir = 'image_data/wise_reprojected'
found = []
not_found = []
with open('mismatched_files.txt','w') as out_file:
    for i in first_names:
        j = i[:7]+i[9:]
        file_path = f'{stamps_dir}/{j}.fits'
        if os.path.exists(file_path):
            found.append(file_path)
        else:
            not_found.append(file_path)
            out_file.write(f'{i}\n')

print(f'Found matching files: {len(found)}')    
print(f'Missing files: {len(not_found)}')    
print(f'Located {len(files)} files from RGZ_RCNN')

