'''
Tookit to help manage the SOM sources, including downloading images, reprojecting
them onto a common grid, producing the binary file, and interacting with PINK. 
'''
import io
import os
import sys
import glob
import requests
import numpy as np
import reproject as rp
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits as pyfits

def make_dir(d):
    '''Helper function to make a directory if it does not exist
    
    Its wrapped in a try incase there is some race condition in a 
    multi-process environment
    '''
    try:
        if not os.path.exists(d):
            os.makedirs(d)
    except OSError as e:
        pass

class Source(object):
    '''Object class to handle a single object from a catalogue, downloading
    images, saving images, reprojecting them onto a common grid, and dumping
    them into a binary file
    '''
    def __str__(self):
        '''Make the string output meaningful
        '''
        out = f'{self.pos}\n'

        if len(self.images) > 0:
            out += f'{len(self.images)} downloaded...\n'

        return out

    def __init__(self, position, out_dir, cutout_size=5., info=None):
        '''Accept the parameters of this particular source

        position - astropy.coordinates.SkyCoord
                The RA/DEC position of a source
        out_dir - str
                The root output directory used for writing to
        cutout_size - float
               The size of postage stamps in arcminutes
        info - any
             A dictionary like object for future bookkeeping
        '''
        self.pos = position
        self.out = out_dir
        self.cutout_size = cutout_size
        self.info = info
        self.filename = None
        self.valid = False # By default, empty source should not be valid
        self.masterstring = None

        self.images = {}
        self.downloaded_images = False
        self.common_images = {}
        self.common_shape = (0,0)

    def _first(self, pad=0.):
        '''Download the postage stamp from the FIRST catalog service

        pad - float
            Additional area to download for each postage stamp
        '''
        d = f'{self.out}/FIRST'
        size = self.cutout_size + pad

        make_dir(d)

        pos = self.pos.to_string('hmsdms', sep=' ')
        data = {'RA': ' '.join(pos.split()),
               'ImageType': 'FITS File',
               'ImageSize':size,
               'Equinox':'J2000'}

        try:
            res = requests.post('https://third.ucllnl.org/cgi-bin/firstcutout', data=data, stream=False)
            
            if self.filename is None:
                fn = res.headers['content-disposition'].split('filename=')[-1].replace('"','')
                self.filename = fn

            filename = f'{d}/{fn}'
            with open(filename, 'wb') as out:
                for i in res:
                    out.write(i)

            pack = filename
        except:
            pack = 'ERROR'

        self.images['FIRST'] = pack

    def _wise(self, pad=3., band=1):
        '''Download the postage stamp from the IRSA WISE service

        pad - float
            Additional area in arcminutes to download for each postage stamp
        band - int
            The WISE band to download
        '''
        band = int(band)
        d = f'{self.out}/WISE_W{band}'
        size = self.cutout_size + pad

        make_dir(d)

        try:
            data = {'POS':self.pos.to_string().replace(' ',',')}
            res = requests.get('https://irsa.ipac.caltech.edu/ibe/search/wise/allsky/4band_p3am_cdd',
                            params=data)
            t = Table.read(io.BytesIO(res.content), format='ascii.ipac')

            coadd_id = t['coadd_id'][0]
            coaddgrp, coadd_ra = coadd_id[:2], coadd_id[:4]
            url = f'https://irsa.ipac.caltech.edu/ibe/data/wise/allsky/4band_p3am_cdd/{coaddgrp:s}/{coadd_ra:s}/{coadd_id:s}/{coadd_id:s}-w{band:1d}-int-3.fits'

            params = {'center':self.pos.to_string().replace(' ',','),
                'size':f'{size}arcmin', 'gzip':'false'}

            res = requests.get(url, params=params)
            filename = f'{d}/{self.filename}'
            with open(f'{filename}', 'wb') as out:
                for i in res:
                    out.write(i)

            pack = filename
        
        except:
            pack = 'ERROR'

        self.images[f'WISE_W{band}'] = pack

    def download_images(self, report=False, force=False):
        '''
        Routines to download images from enabled sources

        report - bool
               Print an optional report on the downloaded images
        force - bool
               By default, if images have already been downloaded, don't
               redo them. This will turn off that option
        '''
        if not force and self.downloaded_images:
            return

        self._first()
        self._wise()
        # self._wise(band=2)
        # self._wise(band=3)
        self.downloaded_images = True
        if report:
            print('Images downloaded')
            for i in self.images:
                print(i, self.images[i])

    def reproject(self, master='FIRST', force=False):
        '''Function to load images and reproject them onto the pixel grid of 
        the image that belongs to the master keyword. Will automatically write
        files to disk and save the path to the file. This is to prevent memory
        issues when a number of source class types are created. 
        
        master - str
              The keyword of the master image to use for the reprojection. It must
              exist in the self.images dictionary
        force - bool
              By default, this reprojection function will only be performed if the 
              common_images attribute is empty. Force will ignore this.
        '''
        # Empty dictionaries evaluate to False
        if not force and self.common_images:
            return

        if master not in self.images.keys():
            print('RETURNING')
            return

        # Open the master fits file and delete the unneeded header fields
        with pyfits.open(self.images[master], memmap=True) as master_fits:
            for i in [j for j in master_fits[0].header if j[0] == 'C' and j[-1] in ['3','4']]:
                master_fits[0].header.pop(i)

            # Set the master file in there
            self.masterstring = master_fits[0].header.tostring()
            self.common_images[master] = master_fits[0].data
            self.common_shape = master_fits[0].data.shape

        for key in [k for k in self.images.keys() if k != master]:
            if self.images[key] == 'ERROR':
                self.common_images[key] = np.zeros((2,2))
            else:
                common = rp.reproject_interp(self.images[key], master_fits[0].header)
                self.common_images[key] = common[0]

        # Attempt to write things to disk to save space in memory
        for k, v in self.common_images.items():
            out_dir = f"{self.out}/{k}_Common"
            path = f"{out_dir}/{self.filename}".replace('.fits','.npy')
            make_dir(out_dir)

            with open(path, 'wb') as out:
                np.save(out, v.astype('f'))
                self.common_images[k] = path

    def _valid(self, nan_fail=True, shape_fail=True, zero_fail=True):
        '''Internal method to actually run the validation checker and 
        set the attribute appropriately. Do it in a lazy manner. i.e,
        the moment it fails a check, return

        nan_fail - bool
                If nan is present in the data, image is not valid
        shape_fail - bool
              Check to see if same number of pixels +/- 5 in both directions
        zero_fail - bool
              Check to see how many zero pixels there are. If to many, than 
              assume that there was clipping of the field
        '''
        self.valid = False
        for img in self.images.values():
            if img == 'ERROR':
                return

            with pyfits.open(img) as fits:
                data = fits[0].data.squeeze()

                if nan_fail and np.isnan(data).any():
                    return
                if shape_fail and np.any(np.abs(np.array(data.shape) - data.shape[0] > 5)):
                    # Useful for WISE data, which will clip the image if the source
                    # falls on the edge of a mosaic image
                    return
                if zero_fail and np.sum(data == 0.) > 100:
                    # Useful for FIRST data, which will pad the image with zeros to
                    # make it the correct size
                    return

        self.valid = True

    def is_valid(self):
        '''Function to assess whether this source is a valid source that 
        should be included into the binary dump for the SOM
        '''
        self._valid()
        return self.valid

    def sigma_clip(self, data, std=3.):
        '''Perform sigma clipping of the data

        data - numpy.ndarray
             The input image data that will be clipped
        std - float
             The clipping level to be used
        '''
        mask = data < std*data.std()
        data[mask] = 0.
        return data

    def normalise(self, data):
        '''Normalise the input data in some manner

        data - numpy.ndarray
             The image that will be normalised to some scale
        '''
        pass

    def dump(self, of, order=None):
        '''Dump the contents of the common_images to a binary file 
        provided by the of file handler

        of - file
           File to write the binary images to

        order - list
             Order of the keys to write out to
        '''
        if order is None:
            for img in self.common_images.values():
                with open(img, 'rb') as in_file:
                    of.write(in_file.read())
        else:
            for item in order:
                if item not in self.common_images.keys():
                    raise ValueError
                with open(self.common_images[item], 'rb') as in_file:
                    of.write(in_file.read())

    def show_reprojected(self):
        '''Quick function to look at each of the 
        '''
        if len(self.common_images) == 0:
            return

        for k, v in self.common_images.items():
            arr = np.load(v)

            fig, ax = plt.subplots(1,1)
            print('Plotting ', k)
            print('\tArray size: ', arr.shape)
            ax.imshow(arr)
            ax.set(title=k)
            fig.show()

if __name__ == '__main__':
    pos = SkyCoord(ra=111.892869323*u.deg, dec=64.6832779684*u.deg)

    a = Source(pos, 'Images')
    print(a)

    a.download_images(report=True)
    a.reproject()

    print(a)
    print('Is a valid: ', a.is_valid())

    a.show_reprojected()

    if a.valid:
        with open('test.dumpy', 'wb') as of:
            a.dump(of)