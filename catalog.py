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

from tqdm import tqdm
# from source import Source
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
        '''A function to save this Binary instance to disk, wrapping a reference to 
        the sources, channels that produced the PINK binary, and making a hash to

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

        of - file
           File to write the binary images to
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

    def __init__(self, catalog=None, out_dir='Images', sources_dir='Sources',
                       scan_sources=True, step=1000):
        '''Accept the name of a catalogue and read it in

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

        # Two options:
        #     - Scan the Sources folder (Images/Sources) assuming images are downloaded
        #     - No Sources to load and need to download them
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

        else:
            raise ValueError

    def _gen_sources(self):
        '''Generate the Source objects
        '''
        print('Generating the Sources....')
        for index, r in tqdm(self.tab.iterrows()):
            self.sources.append(Source(SkyCoord(ra=r['RA']*u.deg, dec=r['DEC']*u.deg), self.out_dir, info=r) )

    def download_validate_images(self):
        '''Kick off the download_images and is_valid method of each of the 
        Source objects
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

        print('\nDownloading and validating the images...')    
        sources = [down(s) for s in self.sources]
        sources = [val(s) for s in sources]
        with ProgressBar():
            self.sources = reduce(sources).compute(num_workers=75)

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
            with open(f"{path}/{s.filename.replace('.fits','.pkl')}", 'wb') as out_file:
                pickle.dump(s, out_file, protocol=3)

    def reproject_valid_sources(self, master='FIRST'):
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

        print(f'Reprojecting {len(self.valid_sources)} sources...')
        valid_sources = [reproject(s) for s in self.valid_sources]

        with ProgressBar():
            self.valid_sources = reduce(valid_sources).compute(num_workers=4)

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
        self.SOM_path = f'{self.binary.binary_path}.Trained_SOM'
        self.SOM_hash = ''
        self.exec_str = ''
        if pink_args:
            self.pink_args = pink_args
        else:
            self.pink_args = {'som-width':10,
                              'som-height':10}

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

    def show_som(self, channel=0):
        '''Method to plot the trained SOM, and associated plotting options

        channel - int
             The channel from the SOM to plot. Defaults to the first (zero-index) channel
        '''
        if not self.trained:
            return
        
        if get_hash(self.SOM_path) != self.SOM_hash:
            return

        with open(self.SOM_path, 'rb') as som:
            # Unpack the header information
            numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = struct.unpack('i' * 6, som.read(4*6))
            SOM_size = np.prod([SOM_width, SOM_height, SOM_depth])

            # Check to ensure that the request channel exists. Remeber we are comparing
            # the index
            if channel > numberOfChannels - 1:
                return

            dataSize = numberOfChannels * SOM_size * neuron_width * neuron_height
            array = np.array(struct.unpack('f' * dataSize, som.read(dataSize * 4)))

            image_width = SOM_width * neuron_width
            image_height = SOM_depth * SOM_height * neuron_height
            data = np.ndarray([SOM_width, SOM_height, SOM_depth, numberOfChannels, neuron_width, neuron_height], 'float', array)
            data = np.swapaxes(data, 0, 5) # neuron_height, SOM_height, SOM_depth, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 0, 2) # SOM_depth, SOM_height, neuron_height, numberOfChannels, neuron_width, SOM_width
            data = np.swapaxes(data, 4, 5) # SOM_depth, SOM_height, neuron_height, numberOfChannels, SOM_width, neuron_width
            data = np.reshape(data, (image_height, numberOfChannels, image_width))

            data_1 = data

            fig, ax = plt.subplots(SOM_width, SOM_height, figsize=(16,16), 
                                gridspec_kw={'hspace':0.001,'wspace':0.001,
                                            'left':0.001, 'right':0.999,
                                            'bottom':0.001, 'top':0.9})

            for x in range(SOM_width):
                for y in range(SOM_height):
                    d = data_1[x*neuron_width:(x+1)*neuron_width, 
                               int(channel),
                               y*neuron_width:(y+1)*neuron_width]
                    ax[x,y].imshow(d, cmap=plt.get_cmap('coolwarm'))
                    ax[x,y].get_xaxis().set_ticks([])
                    ax[x,y].get_yaxis().set_ticks([])

            fig.suptitle((f'Images: {len(self.binary.sources)} - Sigma: {self.binary.sigma} - Norm: {self.binary.norm}\n'
                          f'Channel: {channel}'))
            # fig.subplots_adjust(top=0.90)
            fig.savefig(f'{self.SOM_path}-ch_{channel}.pdf')

if __name__ == '__main__':

    if '-p' in sys.argv:
        for i in ['default_sig_norm_chan.binary', 'default_sig_chan.binary', 'default.binary', 'default_sig.binary', 'default_norm.binary', 'default_sig_norm.binary']:
            load_binary = Binary.loader(i)

            print('Printing the loaded binary...')
            print(load_binary)

            pink = Pink(load_binary, pink_args={'som-width':6,
                                                'som-height':6})  
            print(pink)
            print('\n')

            pink.train()

            print('\n')
            print(pink)
            pink.show_som(channel=0)
            pink.show_som(channel=1)

    elif '-c' in sys.argv:
        cat = Catalog(catalog='./first_14dec17.fits.gz', step=250)
        print(cat)

        cat.download_validate_images()
        cat.collect_valid_sources()
        cat.save_sources()
        cat.reproject_valid_sources()
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
        print(' -s : Run test code for Source class')
        print(' -c : Run test code for Catalogue class')
        print(' -p : Run test code for the Pink class')
