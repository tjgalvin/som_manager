import io
from tqdm import tqdm
import numpy as np
from glob import glob
from astropy import units as u
import requests
import reproject as rp
from astropy.io import fits as pyfits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
import os
import shutil as sh

def make_dir(d):
    '''Helper function to make a directory if it does not exist
    
    Its wrapped in a try incase there is some race condition in a 
    multi-process environment
    '''
    try:
        if not os.path.exists(d):
            print(f'Making {d}')
            os.makedirs(d)
    except OSError as e:
        pass

def down_first_cata():
    url = 'http://sundog.stsci.edu/first/catalogs/first_14dec17.fits.gz'
    fn = url.split('/')[-1]
    
    if not os.path.exists(fn):
        with open(fn,'wb') as out_file:
            print(f'Downloading {url}...')
            response = requests.get(url, stream=True)
            for data in tqdm(response.iter_content()):
                out_file.write(data)

def _first(pos, first_dir, cutout_size=5., pad=0.):
        '''Download the postage stamp from the FIRST catalog service

        pos - Astropy.angle.SkyCoord
            The astropy object for the source position to download
        first_dir - str
            The directory to put FIRST images to
        cutout_size - float
            The cutout size of the postage stamp to download
        pad - float
            Additional area to download for each postage stamp
        '''
        size = cutout_size + pad

        pos_str = pos.to_string('hmsdms', sep=' ')
        data = {'RA': ' '.join(pos_str.split()),
               'ImageType': 'FITS File',
               'ImageSize':size,
               'Equinox':'J2000'}

        try:
            res = requests.post('https://third.ucllnl.org/cgi-bin/firstcutout', data=data, stream=False)
            fn = res.headers['content-disposition'].split('filename=')[-1].replace('"','')
                
            filename = f'{first_dir}/{fn}'
            with open(filename, 'wb') as out:
                for i in res:
                    out.write(i)

            pack = fn
            success = True
        except Exception as e:
            pack = e
            success = False

        return pack, success

def _wise(pos, fn, wise_dir, cutout_size=5., pad=3., band=1):
        '''Download the postage stamp from the IRSA WISE service

        pos - Astropy.angle.SkyCoord
            The SkyCoord Object to download
        fn - str
            The filename to write out to
        wise_dir - str
            The Directory to write the fits images out to
        cutout_size - float
            Size of the postage stamp to download
        pad - float
            Additional area in arcminutes to download for each postage stamp
        band - int
            The WISE band to download
        '''
        band = int(band)
        size = cutout_size + pad

        try:
            data = {'POS':pos.to_string().replace(' ',',')}
            res = requests.get('https://irsa.ipac.caltech.edu/ibe/search/wise/allsky/4band_p3am_cdd',
                            params=data)
            t = Table.read(io.BytesIO(res.content), format='ascii.ipac')

            coadd_id = t['coadd_id'][0]
            coaddgrp, coadd_ra = coadd_id[:2], coadd_id[:4]
            url = f'https://irsa.ipac.caltech.edu/ibe/data/wise/allsky/4band_p3am_cdd/{coaddgrp:s}/{coadd_ra:s}/{coadd_id:s}/{coadd_id:s}-w{band:1d}-int-3.fits'

            params = {'center':pos.to_string().replace(' ',','),
                'size':f'{size}arcmin', 'gzip':'false'}

            res = requests.get(url, params=params)
            filename = f'{wise_dir}/{fn}'
            with open(f'{filename}', 'wb') as out:
                for i in res:
                    out.write(i)

            pack = fn
            success = True
        except Exception as e:
            pack = e
            success = False

        return pack, success

def reproject(master, slave, out_fits):
        '''Function to load images and reproject them onto the pixel grid of 
        the image that belongs to the master FITS image. Will automatically write
        files to disk and save the path to the file. 
        
        master - str
              The filename of the master pixel grid that will be common between the
              two files
        slave - str
              The filename of the FITS image that we will force to match the master
              pixel grid
        out_fits - str
              The filename of the new FITS image to create of the slave image on 
              the master grid
        '''

        # Open the master fits file and delete the unneeded header fields
        with pyfits.open(master, memmap=True) as master_fits, pyfits.open(slave, memmap=True) as slave_fits:
            
            # Reproject freaks out if the number of dimensions is not the 
            # same between fits images. The FIRST general as two extra 
            # dimensions of length one for freq and stokes
            for fits in [master_fits, slave_fits]:
                for i in [j for j in fits[0].header if j[0] == 'C' and j[-1] in ['3','4']]:
                    fits[0].header.pop(i)
            
            master_cp = master_fits[0].header
            
            for item in ['BUNIT', 'MAGZP', 'MAGZPUNC']:
                master_cp[item] = slave_fits[0].header[item]
            for item in ['DATAMAX','DATAMIN']:
                master_cp.pop(item)
                

            
            # Perform the interpolation onto the new grid
            common = rp.reproject_interp(slave_fits, master_fits[0].header)
            pyfits.writeto(out_fits, 
                           common[0].astype(np.float), 
                           master_cp, 
                           overwrite=True)
                        
def download(row):
    '''Download the images from FIRST and WISE, and reproject them onto the grid of FIRST
    '''
    pos = SkyCoord(ra=row['RA']*u.deg, dec=row['DEC']*u.deg)
    fn, first_success = _first(pos, f_dir)
    if not first_success:
        return 'First_Failed'
    
    fn, wise_success = _wise(pos, fn, w_dir, pad=1)
    if not wise_success:
        return 'WISE_Failed'
        
    reproject(f'{f_dir}/{fn}', f'{w_dir}/{fn}', f'{r_dir}/{fn}')
    return fn

if __name__ == '__main__':
    f_dir = 'Images/first'
    w_dir = 'Images/wise'
    r_dir = 'Images/wise_reprojected'

    make_dir(f_dir)
    make_dir(w_dir)
    make_dir(r_dir)

    down_first_cata()
    df = Table.read('first_14dec17.fits.gz').to_pandas()
    sd = dd.from_pandas(df, npartitions=100)

    sd['filename'] = sd.apply(lambda x: download(x), axis=1, meta=('filename', str))
    with ProgressBar():
        a = sd.compute()

    a.to_csv('FIRST_Cata_Images.csv')
    a.to_json('FIRST_Cata_Images.json')
    a.to_pickle('FIRST_Cata_Images.pkl')   