'''A manager class to accept a catalogue and generate the sources
'''
import io
import os
import sys
import glob
import shutil
import pickle
import struct
import hashlib
import requests
import subprocess
import numpy as np
import reproject as rp
import astropy.units as u
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from astropy.table import Table
from astropy.io import fits as pyfits
from astropy.coordinates import SkyCoord

from dask import delayed
from dask.diagnostics import ProgressBar

# This will hide the WARNINGs produced by fits.open and an ill
# crafted FITS header field
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

def chunk_sources(l, chunk_length):
    '''Function to yield a smaller subset of the passed iterable `l`
    so that Dask does not take forever building the work graph
    
    l - list or list like
          The list to return in chunks
    chunk_length - int
          Length of elements in each chunk
    '''
    for i in range(0, len(l), chunk_length):
        yield l[i:i+chunk_length]

def get_hash(f, blocksize=56332):
    '''Will return the SHA1 hash of the input file

    blocksize - int
          Number of bytes to read in at once to save memory
    '''
    sha1 = hashlib.sha1()

    with open(f, 'rb') as in_file:
        while True:
            data = in_file.read(blocksize)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()

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

class Base(object):
    '''Base class to inherit from. 
    '''
    def save(self, out_path):
        '''A function to save this class instance to disk.

        out_path - str
              The path to write the pickled output to
        '''
        with open(out_path, 'wb') as out_file:
            pickle.dump(self.__dict__, out_file)

    @classmethod
    def loader(cls, in_path):
        '''Load in a saved reference of the class pickle produced by save()

        in_path - str
              Path to the saved Binary class
        '''
        with open(in_path, 'rb') as in_file:
            attributes = pickle.load(in_file)

        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)

        return obj

class Source(Base):
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

    def __init__(self, out_dir, rgz_path=None, position=None, cutout_size=5., info=None):
        '''Accept the parameters of this particular source

        out_dir - str
                The root output directory used for writing to
        rgz_path - str
                The path to the file to load in if data are being using from the RGZ dataset
                downloaded using the rgz_cnn download_data.py script from Chen Wu (UWA)
        position - astropy.coordinates.SkyCoord or None
                The RA/DEC position of a source to be used if data are to be
                dowloaded
        cutout_size - float
               The size of postage stamps in arcminutes if data are to be downloaded
        info - any
             A dictionary like object for future bookkeeping
        '''
        if rgz_path is None and position is None:
            raise ValueError('No valid Source mode has been set')

        self.rgz_path = rgz_path
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

        if self.rgz_path is not None:
            self.load_rgz_images()

    # ----------------------------------------------------------------
    # Functions to load and reproject images
    # ----------------------------------------------------------------    
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
                if np.any(np.isnan(common[0])):
                    self.valid = False

        # Attempt to write things to disk to save space in memory
        for k, v in self.common_images.items():
            out_dir = f"{self.out}/{k}_Common"
            path = f"{out_dir}/{self.filename}".replace('.fits','.npy')
            make_dir(out_dir)

            with open(path, 'wb') as out:
                np.save(out, v.astype('f'))
                self.common_images[k] = path
    # ----------------------------------------------------------------    

    # ----------------------------------------------------------------    
    # Function to load in data from RGZ
    # ----------------------------------------------------------------    
    def load_rgz_images(self):
        '''Given the image path from the intialisation of this Source instance
        attempt to load in the saved RGZ images using the data directory structure
        from the download_data.py script. 

        Since these images are (1) PNGs, and (2) already reprojected, we will load them
        in, convert them to greyscale, and place them into the COMMON images folder. Since
        we are following a `hardcoded` directory structure, the keys to the self.common_images
        dict will be `FIRST` and `WISE_W1`.
        '''
        self.valid = False
        # Load in the FIRST Dataset
        first_path = self.rgz_path
        self.filename = first_path.split('/')[-1]
        
        out_dir = f"{self.out}/FIRST_Common"
        path = f"{out_dir}/{self.filename}".replace('.png','.npy')
        make_dir(out_dir)

        # Convert image to grey scale and dump it in the _Common path
        img = np.array(Image.open(first_path).convert('L'))
        s1 = img.shape
        with open(path, 'wb') as out:
            np.save(out, img.astype('f'))
            self.common_images['FIRST'] = path

        wise_path = first_path.replace('_logminmax','_infraredct')
        if not os.path.exists(wise_path):
            self.common_images['WISE_W1'] = 'ERROR'
            return

        out_dir = f"{self.out}/WISE_W1_Common"
        path = f"{out_dir}/{self.filename}".replace('.png','.npy')
        make_dir(out_dir)

        # Convert image to grey scale and dump it in the _Common path
        img = np.array(Image.open(wise_path).convert('L'))
        s2 = img.shape
        with open(path, 'wb') as out:
            np.save(out, img.astype('f'))
            self.common_images['WISE_W1'] = path

        # The images do not have the same sizes...
        if s1 != s2:
            return

        self.common_shape = s1
        self.valid = True
    # ----------------------------------------------------------------    

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
            if self.filename is None:
                return

            with pyfits.open(img) as fits:
                data = fits[0].data.squeeze()

                if nan_fail and np.isnan(data).any():
                    return
                if shape_fail and np.any(np.abs(np.array(data.shape) - data.shape[0]) > 5):
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
        return 1. * data / data.max()

    def dump(self, of, order=None, sigma=False, norm=False):
        '''Dump the contents of the common_images to a binary file 
        provided by the of file handler

        of - file or None
           File to write the binary images to. If none, return data, otherwise
           write it to the file
        order - list
             Order of the keys to write out to
        sigma - False or float
             If False, there will be no sigma clipping performed on the 
             data. If float, then 
        '''
        if order is None:
            for img in self.common_images.values():
                with open(img, 'rb') as in_file:
                    data = np.load(in_file)
                    if isinstance(sigma, float):
                        data = self.sigma_clip(data, std=sigma)
                    if norm:
                        data = self.normalise(data)

                    if of is None:
                        return data.astype('float')
                    else:
                        data.astype('f').tofile(of)
        else:
            for item in order:
                if item not in self.common_images.keys():
                    raise ValueError
                with open(self.common_images[item], 'rb') as in_file:
                    data = np.load(in_file)
                    if isinstance(sigma, float):
                        data = self.sigma_clip(data, std=sigma)
                    if norm:
                        data = self.normalise(data)

                    if of is None:
                        return data
                    else:
                        data.astype('f').tofile(of)

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

class Binary(Base):
    '''Wrap up a Binary file into a class to `attach` metadata associated 
    with it, including the valid sources, the channels and their order, 
    a hash of the binary, the path and anything else I can think of
    '''
    def __str__(self):
        '''Return a neat printed statement of the object instance
        '''
        out = f'Binary path: {self.binary_path}\n'
        out+= f'Number of sources {len(self.sources)}\n'
        out+= f'The binary SHA1: {self.binary_hash}\n'

        return out

    def __repr__(self):
        '''Print a neat represenation of the object. Return __str__
        '''
        return self.__str__()

    def __init__(self, binary_path, sources, sigma, norm, channels=''):
        '''Create and track the meta-information of the binary

        binary_path - str
             The path to the binary file
        source - list
             A list of the Source objects packed into the binary_path file
        sigma - False of float
            The option passed to each of the Source.dump() methods
        norm - bool
            The option passed to each of the Source.dump() methods
        channels - list
            A list of the channels/surveys packed into the binary_path file
        '''
        self.sources = sources
        self.binary_path = binary_path
        self.binary_hash = get_hash(self.binary_path)
        self.channels = channels
        self.sigma = sigma
        self.norm = norm

    def get_image(self, index, channel=0):
        '''Return the index-th image that was dumped to the binary image file that
        is managed by this instance of Binary
        
        index - int
            The source image to return
        channel - int
            The channel number of the image to return
        '''
        with open(self.binary_path, 'rb') as in_file:
            numberOfImages, numberOfChannels, width, height = struct.unpack('i' * 4, in_file.read(4 * 4))
            if index > numberOfImages:
                return None
            if channel > numberOfChannels:
                return None

            size = width * height
            in_file.seek((index*numberOfChannels + channel) * size*4, 1)
            array = np.array(struct.unpack('f' * size, in_file.read(size*4)))
            data = np.ndarray([width,height], 'float', array)

            return data

    def get_data(self, label):
        '''Function to return some property from each of the attached self.sources
        objects

        label - str
             Name of the field in the self.sources object to retreive
        '''
        return [ s.info[label] for s in self.sources ]

class Catalog(Base):
    '''A class object to manage a catalogue and spawn corresponding Source
    classes
    '''
    def __str__(self):
        '''A neat string output for this class
        '''
        out  = f'Catalog file {self.catalog}\n'
        out += f'\tNumber of Sources {len(self.sources)}'

        return out

    def __repr__(self):
        '''Return neat representation of the object. Defaults to __str__
        '''
        return self.__str__()

    def __init__(self, rgz_dir=None, catalog=None, out_dir='Images', 
                       sources_dir='Sources', scan_sources=True, step=1000):
        '''Accept the name of a catalogue and read it in

        rgz_dir - str or None
                Directory of data downloaded from the RGZ using Chen Wu dowload_data.py script
                https://github.com/chenwuperth/rgz_rcnn/blob/master/tools/download_data.py#L30
        catalog - str
               The name of the catalogue file to attempt to load in
        out_dir - str
               The name of the output directory to save outputs to
        sources_dir - str
               The name of the folder that Source class instances will be pickled to
        scan_sources - bool
               By default, attempt to scan in pickled Source class instances
        step - int
               Reduce the size of the input catalog by stepping every `step` rows
        '''
        self.out_dir = out_dir
        self.sources_dir = sources_dir
        # Make the Source objects
        self.sources = []
        self.valid_sources = None

        # Three options:
        #     - Scan the Sources folder (Images/Sources) assuming images are downloaded
        #     - No Sources to load and need to download them
        #     - Data has been downloaded from Chen Wu rgz_rcnn package and needs to be processed
        # Option one would be most preferred to avoid the downloading, so lets attempt
        # to load those in, and then form the catalog and valid sources, otherwise 
        # attempt to load in the catalog again
        if scan_sources and os.path.exists(f'{self.out_dir}/{sources_dir}'):
            src_files = glob.glob(f'{self.out_dir}/{sources_dir}/*pkl')
            print('\nLoading in previous Source pickles')
            for s in tqdm(src_files):
                with open(s, 'rb') as in_file:
                    self.sources.append(pickle.load(in_file))
            self.catalog = 'Generated from Source pickles'

        elif catalog is not None:
            self.catalog = catalog
            self.tab = Table.read(catalog)[::step].to_pandas()
            self._gen_sources()

        elif rgz_dir is not None and os.path.exists(rgz_dir):
            self.catalog = rgz_dir
            self.tab = None
            self._scan_rgz()

        else:
            raise ValueError

    def _scan_rgz(self):
        '''Function to attempt to read in data downloaded from RGZ, assuming the 
        folder structure from the rgz_rcnn download_data.py script from Chen Wu
        '''
        rgz_dir = self.catalog
        print(f'The RGZ directory is: {rgz_dir}')

        scan_str = f'{rgz_dir}/RGZdevkit2017/RGZ2017/PNGImages/*_logminmax.png'
        print(f'Globbing for files with: {scan_str}')
        files = glob.glob(scan_str)

        print(f'Located {len(files)} files...')

        for f in tqdm(files):
            self.sources.append(Source(self.out_dir, rgz_path=f))

    def _gen_sources(self):
        '''Generate the Source objects
        '''
        print('Generating the Sources....')
        for index, r in tqdm(self.tab.iterrows()):
            self.sources.append(Source(SkyCoord(ra=r['RA']*u.deg, dec=r['DEC']*u.deg), self.out_dir, info=r) )

    def download_validate_images(self, chunk_length=3000):
        '''Kick off the download_images and is_valid method of each of the 
        Source objects

        chunk_length - int
             To avoid overhead of Dask builing the graph, which appear to scale to N**2, 
             break up the input list into smaller chunks
        '''
        @delayed
        def down(s):
            s.download_images()
            return s
        
        @delayed
        def val(s):
            s.is_valid()
            return s
        
        @delayed
        def reduce(s):
            return s

        results = []

        print(f'\nDownloading and validating {len(self.sources)} sources...') 
        for sub_sources in chunk_sources(self.sources, chunk_length):
            sub_sources = [down(s) for s in sub_sources]
            sub_sources = [val(s) for s in sub_sources]
            with ProgressBar():
                results += reduce(sub_sources).compute(num_workers=20)

        self.sources = results

    def collect_valid_sources(self):
        '''Make a new list of just the valid sources
        '''
        self.valid_sources = [s for s in self.sources if s.valid == True]
        print(f'\nThe number of valid sources is: {len(self.valid_sources)}')

    def save_sources(self, out_dir='Sources'):
        '''Attempt to save the sources to a directory to read in later and
        avoid having to redownload
        '''
        path = f'{self.out_dir}/{out_dir}'
        make_dir(path)

        print(f'\nSaving sources to {path}')

        for s in tqdm(self.sources):
            if s.filename is not None:
                # Dirty dirty str replace
                with open(f"{path}/{s.filename.replace('.png','.pkl').replace('.fits','.pkl')}", 'wb') as out_file:
                    pickle.dump(s, out_file, protocol=3)

    def reproject_valid_sources(self, master='FIRST', chunk_length=3000):
        '''Proceed to reproject each of the images onto the common pixel grid
        of the elected master image

        master - str
              The dictionary key/image survey of the master FITS image to match to
        '''

        @delayed
        def reproject(s):
            s.reproject(master=master)
            return s
        
        @delayed
        def reduce(s):
            return s

        results = []
        print(f'Reprojecting {len(self.valid_sources)} sources...')
        for sub_sources in chunk_sources(self.valid_sources, chunk_length):
            valid_sources = [reproject(s) for s in sub_sources]

            with ProgressBar():
                results += reduce(valid_sources).compute(num_workers=4)

        self.valid_sources = results

    def dump_binary(self, binary_out, channels=['FIRST'], sigma=False, norm=False):
        '''This function produces the binary file that is expected by PINK. It contains
        the total number of images to use, the number of chanels and the dimension in and y 
        axis

        binary_out - str
             The name of the output binary file to create
        channels - list or str
             The channels to write out to the dumped binary file. Either the name of
             the single survey, or a list of surveys to dump
        sigma - Bool or Float
             Passed directly to the Source.dump() method. If False, no sigma clipping. If
             a float, then sigma clipping is performed.
        norm - bool
             Sets whether normalisation on each plane will be performed. At the moment this
             is an independent, meaning each plane is normalised independently from one
             another
        '''
        if isinstance(channels, str):
            channels = [channels]

        # Lets find the total set of image sizes
        img_sizes = list(set([s.common_shape for s in self.valid_sources]))        

        # TODO: Will want to eventually get the smallest number in each direction
        # to use as the image dimensions. For the moment ignore the problem
        assert len(img_sizes) == 1, 'Different image sizes, not sure what to do yet'
        x_dim, y_dim = img_sizes[0][0], img_sizes[0][1]

        print(f'The number of images to dump: {len(self.valid_sources)}')
        print(f'The number of channels to dump: {len(channels)}')
        print(f'The x and y dimensions: {x_dim}, {y_dim} ')
        with open(binary_out, 'wb') as out_file:
            out_file.write(struct.pack('i', len(self.valid_sources)))
            out_file.write(struct.pack('i', len(channels)))
            out_file.write(struct.pack('i', x_dim))
            out_file.write(struct.pack('i', y_dim))

            for s in self.valid_sources:
                s.dump(out_file, order=channels, sigma=sigma, norm=norm)

        return Binary(binary_out, self.valid_sources, sigma, norm, channels=channels)

class Pink(Base):
    '''Manage a single Pink training session, including the arguments used to run
    it, the binary file used to train it, and interacting with the results afterwards
    '''
    def __str__(self):
        '''Neatly print the contents of the instance
        '''
        out = f'The binary file attached: {self.binary.binary_path}\n'
        out+= f'Contains {len(self.binary.sources)} sources\n'
        out+= f'Channels are {self.binary.channels}\n'
        if self.trained:
            out+= f'SOM is trained: {self.SOM_path}\n'
            out+= f'SOM weight hash is {self.SOM_hash}\n'
        else:
            out+= 'SOM is not trained'

        return out

    def __repr__(self):
        '''Print a neat representation of the object. Defaults to __str__
        '''
        return self.__str__()

    def __init__(self, binary, pink_args = {}):
        '''Create the instance of the Pink object and set appropriate parameters
        
        binary - Binary
             An instance of the Binary class that will be used to train Pink
        pink_args - dict
             The arguments to supply to Pink
        '''
        if not isinstance(binary, Binary):
            raise TypeError('binary is expected to be instance of Binary class')

        self.trained = False
        self.binary = binary
        # Items to generate the SOM
        self.SOM_path = f'{self.binary.binary_path}.Trained_SOM'
        self.SOM_hash = ''
        self.exec_str = ''

        if pink_args:
            self.pink_args = pink_args
        else:
            self.pink_args = {'som-width':10,
                              'som-height':10}

        # Items for the Heatmap
        self.heat_path = f'{self.binary.binary_path}.heat'
        self.heat_hash = ''
        self.src_heatmap = None

    def update_pink_args(self, **kwargs):
        '''Helper function to update pink arguments that may not have been included
        when creating the class instance
    
        Use the kwargs and simply update the pink_args attribute
        '''
        self.pink_args.update(kwargs)

    def train(self):
        '''Train the SOM with PINK using the supplied options and Binary file
        '''
        if self.trained:
            print('The SOM has been trained already')
            return
        
        if not os.path.exists(self.binary.binary_path):
            raise ValueError(f'Unable to locate {self.binary.binary_path}')
        
        if self.binary.binary_hash != get_hash(self.binary.binary_path):
            raise ValueError(f'The hash checked failed for {self.binary.binary_path}')

        pink_avail = True if shutil.which('Pink') is not None else False
        exec_str  = f'Pink --train {self.binary.binary_path} {self.SOM_path} '
        exec_str += ' '.join(f'--{k}={v}' for k,v in self.pink_args.items())

        if pink_avail:
            self.exec_str = exec_str
            self.pink_process = subprocess.run(self.exec_str.split())
            self.trained = True
            self.SOM_hash = get_hash(self.SOM_path)
        else:
            print('PINK can not be found on this system...')

    def retrieve_som_data(self, channel=0):
        '''If trained, this function will return the SOM data from some desired
        channel

        channel - int or None
             The channel from the some to retrieve. If a negative number or None
             return the entire structure, otherwise return that channel number
        '''
        if not self.trained:
            return None
        
        if get_hash(self.SOM_path) != self.SOM_hash:
            return None

        with open(self.SOM_path, 'rb') as som:
            # Unpack the header information
            numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = struct.unpack('i' * 6, som.read(4*6))
            SOM_size = np.prod([SOM_width, SOM_height, SOM_depth])

            # Check to ensure that the request channel exists. Remeber we are comparing
            # the index
            if channel > numberOfChannels - 1:
                return None

            dataSize = numberOfChannels * SOM_size * neuron_width * neuron_height
            array = np.array(struct.unpack('f' * dataSize, som.read(dataSize * 4)))

            image_width = SOM_width * neuron_width
            image_height = SOM_depth * SOM_height * neuron_height
            data = np.ndarray([SOM_width, SOM_height, SOM_depth, numberOfChannels, neuron_width, neuron_height], 'float', array)
            data = np.swapaxes(data, 0, 5) # neuron_height, SOM_height, SOM_depth, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 0, 2) # SOM_depth, SOM_height, neuron_height, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 4, 5) # SOM_depth, SOM_height, neuron_height, numberOfChannels, SOM_width, neuron_width
            data = np.reshape(data, (image_height, numberOfChannels, image_width))

            if channel < 0 or channel is None:
                # Leave data as is and return
                pass
            else:
                data = data[:,channel,:]

            return (data, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height)

    def show_som(self, channel=0):
        '''Method to plot the trained SOM, and associated plotting options

        channel - int
             The channel from the SOM to plot. Defaults to the first (zero-index) channel
        '''
        params = self.retrieve_som_data(channel=channel)
        if params is None:
            return
        (data, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height) = params

        fig, ax = plt.subplots(SOM_width, SOM_height, figsize=(16,16), 
                            gridspec_kw={'hspace':0.001,'wspace':0.001,
                                        'left':0.001, 'right':0.999,
                                        'bottom':0.001, 'top':0.9})

        for x in range(SOM_width):
            for y in range(SOM_height):
                d = data[x*neuron_width:(x+1)*neuron_width, 
                         y*neuron_width:(y+1)*neuron_width]
                ax[x,y].imshow(d, cmap=plt.get_cmap('coolwarm'))
                ax[x,y].get_xaxis().set_ticks([])
                ax[x,y].get_yaxis().set_ticks([])

        fig.suptitle((f'Images: {len(self.binary.sources)} - Sigma: {self.binary.sigma} - Norm: {self.binary.norm}\n'
                        f'Channel: {channel}'))
        fig.savefig(f'{self.SOM_path}-ch_{channel}.pdf')

    def _process_heatmap(self, image_number=0, plot=False, channel=0, binary=None):
        '''Function to process the heatmap file produced by the `--map`
        option in the Pink utility

        image_number - int
               Index of the image to open and display the map for
        plot - bool
               Plot the weight map to inspect quickly
        channel - int
               The channel/plan of the SOM to extract. This is only used for plotting
               and has no impact on the actual heatmap
        binary - Binary or None
               An instance of the Binary class to manage a binary image file
               from which a source image will be pulled from. if none, revert
               to the one contained in this instance
        '''
        if binary is None:
            binary = self.binary
            
        with open(self.heat_path, 'rb') as in_file:
            numberOfImages, SOM_width, SOM_height, SOM_depth = struct.unpack('i' * 4, in_file.read(4*4))
            
            size = SOM_width * SOM_height * SOM_depth
            image_width = SOM_width
            image_height = SOM_depth * SOM_height
            in_file.seek(image_number * size * 4, 1)
            array = np.array(struct.unpack('f' * size, in_file.read(size * 4)))
            data = np.ndarray([SOM_width, SOM_height, SOM_depth], 'float', array)
            data = np.swapaxes(data, 0, 2)
            data = np.reshape(data, (image_height, image_width))
            # data = data[::-1] # Apprently the first axis is out of order when you
                              # plot it agaisnt the proper SOM and an the corresponding
                              # source
            data /= data.sum()
            # Simple diagnostic plot
            if plot:
                fig, (ax1, ax2, ax3) = plt.subplots(1,3)
                params = self.retrieve_som_data(channel=channel)
                if params is None:
                    return
                (som, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height) = params

                source_img = binary.get_image(image_number)

                ax1.imshow(data)
                ax2.imshow(som)
                ax3.imshow(source_img)

                # plt.show() will block, but fig.show() wont
                plt.show()
            
            return data

    def _apply_heatmap(self):
        '''Function to loop through the Pink map output (i.e. self.heatmap) and 
        read in as a list each source heatmap
        '''
        result = []
        for index, src in enumerate(self.binary.sources):
            result.append(self._process_heatmap(image_number=index))
        
        self.src_heatmap = result

    def heatmap(self, binary=None, plot=False, apply=False, **kwargs):
        '''Using Pink, produce a heatmap of the input Binary instance. 
        Note that by default the Binary instance attached to self.binary will be used. 

        binary - Binary or None
             An instance of the Binary class with sources to match to the SOM. If None, 
             than use the Binary instance attached to this Pink class instance
        plot - bool
             Make an initial diagnostic plot of the SOM and the correponding heatmap.
             This will show the first source of the binary object
        apply - bool
             Add an attribute to the class instance with the list of heatmaps
        kwargs - dict
             Additional parameters passed directly to _process_heatmap()
        '''
        if binary is None:
            binary = self.binary
        if not self.trained:
            return
        if self.SOM_hash != get_hash(self.SOM_path):
            raise ValueError(f'The hash checked failed for {self.SOM_path}')        
        if binary.binary_hash != get_hash(binary.binary_path):
            raise ValueError(f'The hash checked failed for {self.binary.binary_path}')

        pink_avail = True if shutil.which('Pink') is not None else False        
        exec_str = f'Pink --map {self.binary.binary_path} {self.heat_path} {self.SOM_path} '
        exec_str += ' '.join(f'--{k}={v}' for k,v in self.pink_args.items())
        
        if pink_avail:
            if not os.path.exists(self.heat_path):
                subprocess.run(exec_str.split())
                self.heat_hash = get_hash(self.heat_path)
            self._process_heatmap(plot=plot, binary=binary, **kwargs)
            if apply:
                self._apply_heatmap()
        else:
            print('PINK can not be found on this system...')

    def attribute_plot(self, book, shape):
        '''Produce a grid of histograms based on the list of items inside it
        
        book - dict
            A dictionary whose keys are the location on the heatmap, and values
            are the list of values of sources who most belonged to that grid
        shape - tuple
            The shape of the grid. Should attempt to get this from the keys or
            possible recreate it like in self.attribute_heatmap()
        '''
        # Step one, get range
        vmin, vmax = np.inf, 0
        for k, v in book.items():
            if min(v) < vmin:
                vmin = min(v)
            if max(v) > vmax:
                vmax = max(v)
        bins = np.linspace(vmin, vmax, 10)

        fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1])

        for k, v in book.items():
            print(k, len(v))
            ax[k].hist(v, bins=bins)

        plt.show()

    def attribute_heatmap(self, label='SIDEPROB', plot=True):
        '''Based on the most likely grid/best match in the heatmap for each source
        build up a distibution plot of each some label/parameters

        label - str
             The label/value to extract from each source
        plot - bool
             Plot the attribute heatmap/distribution
        '''
        shape = self.src_heatmap[0].shape
        items = self.binary.get_data(label)

        book = defaultdict(list)

        for heat, item in zip(self.src_heatmap, items):
            loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)
            book[loc].append(item)

        if plot:
            self.attribute_plot(book, shape)

if __name__ == '__main__':

    if '-r' in sys.argv:
        rgz_dir = 'rgz_rcnn_data'

        cat = Catalog(rgz_dir=rgz_dir)

        cat.save_sources()
        cat.collect_valid_sources()

        test_bin = cat.dump_binary('TEST.binary', norm=True)
        # test_bin = cat.dump_binary('TEST.binary', channels=['FIRST','WISE_W1'])

        print(test_bin)

        pink = Pink(test_bin, pink_args={'som-width':6,
                                         'som-height':6}) 

        pink.train()
        pink.save('TEST.pink')
        pink.heatmap(plot=True, image_number=0, apply=True)
        pink.heatmap(plot=True, image_number=500, apply=True)
        

    elif '-m' in sys.argv:
        pink_file = 'default_sig_chan.Pink'
        print(f'Loading in {pink_file}')
        load_pink = Pink.loader(pink_file)

        print(load_pink)

        load_pink.heatmap(plot=True, image_number=0, apply=True)
        load_pink.attribute_heatmap(plot=True, label='RMS')

    elif '-p' in sys.argv:
        for i in ['default_sig_norm_chan.binary', 'default_sig_chan.binary', 'default.binary', 'default_sig.binary', 'default_norm.binary', 'default_sig_norm.binary']:
        # for i in ['default_sig_norm.binary']:
            load_binary = Binary.loader(i)

            print('Printing the loaded binary...')
            print(load_binary)

            pink = Pink(load_binary, pink_args={'som-width':12,
                                                'som-height':12})  
            print(pink)
            print('\n')

            pink.train()

            print('\n')
            print(pink)
            # pink.show_som(channel=0)
            # pink.show_som(channel=1)
            pink.save(i.replace('binary', 'Pink'))

    elif '-c' in sys.argv:
        cat = Catalog(catalog='./first_14dec17.fits.gz', step=250)
        print(cat)

        cat.download_validate_images()
        cat.collect_valid_sources()
        cat.save_sources()
        cat.reproject_valid_sources()
        cat.collect_valid_sources()
        print('\n')
        binary = cat.dump_binary('default.dump')
        binary.save('default.binary')

        binary_sig_chan = cat.dump_binary('default_sig_chan.dump' ,sigma=3., channels=['FIRST','WISE_W1'])
        binary_sig_chan.save('default_sig_chan.binary')

        binary_sig_chan = cat.dump_binary('default_sig_norm_chan.dump', norm=True ,sigma=3., channels=['FIRST','WISE_W1'])
        binary_sig_chan.save('default_sig_norm_chan.binary')

        binary_sig = cat.dump_binary('default_sig.dump' ,sigma=3.)
        binary_sig.save('default_sig.binary')

        binary_norm = cat.dump_binary('default_norm.dump', norm=True)
        binary_norm.save('default_norm.binary')

        binary_sig_norm = cat.dump_binary('default_norm_sig.dump', sigma=3., norm=True)
        binary_sig_norm.save('default_sig_norm.binary')

        print('\n', binary)

        load_binary = Binary.loader('default.binary')

        print('Printing the loaded binary...')
        print(load_binary)

    elif '-s' in sys.argv:
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
    
    else:
        print('Options:')
        print(' -r : Run test code to scan in RGZ image data')
        print(' -s : Run test code for Source class')
        print(' -c : Run test code for Catalogue class')
        print(' -p : Run test code for the Pink class')
        print(' -m : Run the heatmap code for a trained Pink model')