'''A manager class to accept a catalogue and generate the sources
'''
import io
import os
import sys
import glob
import shutil
import pickle
import struct
import xmltodict as xd
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
    # Function to load in data from RGZ
    # ----------------------------------------------------------------    
    def load_rgz_images_pngs(self):
        '''Given the image path from the intialisation of this Source instance
        attempt to load in the saved RGZ images using the data directory structure
        from the download_data.py script. 

        Since these images are (1) PNGs, and (2) already reprojected, we will load them
        in, convert them to greyscale, and place them into the COMMON images folder. Since
        we are following a `hardcoded` directory structure, the keys to the self.common_images
        dict will be `FIRST` and `WISE_W1`.

        Attach the annotation file to the self.info structure for the moment. 
        '''
        self.valid = False
        # Load in the FIRST Dataset
        first_path = self.rgz_path
        self.filename = first_path.split('/')[-1]
        
        xml = first_path.replace('PNGImages', 'Annotations').replace('.png', '.xml')
        if os.path.exists(xml):
            with open(xml, 'r') as xml_in:
                self.info = xd.parse(xml_in.read())

        out_dir = f"{self.out}/FIRST_Common"
        path = f"{out_dir}/{self.filename}".replace('.png','.npy')
        make_dir(out_dir)

        # Convert image to grey scale and dump it in the _Common path
        img = np.array(Image.open(first_path).convert('L'))
        img = img[3:-3, 3:-3] # Clip out pixels that may have used to pad reprojection
        s1 = img.shape
        if np.any(~np.isfinite(img)):
            print(f'Dropping RGZ FIRST image {self.filename}')
            return

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
        img = img[3:-3, 3:-3] # Clip out pixels that may have used to pad reprojection
        s2 = img.shape
        if np.any(~np.isfinite(img)):
            print(f'Dropping RGZ WISE image {self.filename}')
            return
        
        with open(path, 'wb') as out:
            np.save(out, img.astype('f'))
            self.common_images['WISE_W1'] = path

        # The images do not have the same sizes...
        if s1 != s2:
            return
        
        self.common_shape = s1
        self.valid = True
    
    def load_rgz_images(self, down_dir='image_data'):
        '''Given the image path from the intialisation of this Source instance
        attempt to load in the saved RGZ images using the data directory structure
        from the download_data.py script. Use the filenames from that directory to 
        open the FITS images downloaded from the download_images.py scrip from this repo.

        Since these images are (1) already reprojected, we will load them in and place them into 
        the COMMON images folder. Since we are following a `hardcoded` directory structure, the keys to the self.common_images
        dict will be `FIRST` and `WISE_W1`.

        Attach the annotation file to the self.info structure for the moment.

        down_dir - str
              The directory of the FITS data that has been downloaded from download_images.py
        '''
        self.valid = False
        # Load in the FIRST Dataset
        first_path = self.rgz_path
        self.filename = first_path.split('/')[-1]
                
        # The filename from FIRST server is slightly different to the name used by rgz_rcnn
        down_name = self.filename.replace('_logminmax.png','').replace('FIRST','')        
        down_name = down_name[:7]+down_name[9:]
        # print(self.filename, down_name)

        xml = first_path.replace('PNGImages', 'Annotations').replace('.png', '.xml')
        if os.path.exists(xml):
            with open(xml, 'r') as xml_in:
                self.info = xd.parse(xml_in.read())

        out_dir = f"{self.out}/FIRST_Common"
        make_dir(out_dir)

        first_file = f'{down_dir}/first/{down_name}.fits'
        if os.path.exists(first_file):
            with pyfits.open(first_file, memmap=True) as in_file:
                data = in_file[0].data
                # if np.any(~np.isfinite(data)):
                #     print(f'Dropping RGZ FIRST image {self.filename}')
                #     return

                first_path = f'{out_dir}/{down_name}.npy'
                np.save(first_path, data.astype('f'))

                s1 = data.shape
                self.common_images['FIRST'] = first_path
        # else:
        #     print('FIRST FILE', down_name, first_file)

        wise_file = first_file.replace('first','wise_reprojected')
        if not os.path.exists(wise_file):
            # print('WISE FILE', wise_file)
            self.common_images['WISE_W1'] = 'ERROR'
            return

        out_dir = f"{self.out}/WISE_W1_Common"
        path = f"{out_dir}/{down_name}".replace('.fits','.npy')
        make_dir(out_dir)

        with pyfits.open(wise_file, memmap=True) as in_file:
            data = in_file[0].data
            # if np.any(~np.isfinite(data)):
            #     print(f'Dropping RGZ FIRST image {self.filename}')
            #     return

            wise_path = f'{out_dir}/{down_name}.npy'
            np.save(wise_path, data.astype('f'))

            s2 = data.shape
            self.common_images['WISE_W1'] = wise_path
        
        # The images do not have the same sizes...
        if s1 != s2:
            return
        
        self.common_shape = s1
        self.valid = True
    
    def rgz_annotations(self):
        '''Return the object annotation information from the self.info
        structure. if this is None, than the corresponding XML file was
        not found when running self.load_rgz_image(). Return, as a list, 
        the names of each of the components for now. 
        '''
        if self.info is None:
            return None

        return self.info['annotation']
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
        for img in self.common_images.values():
            if img == 'ERROR':
                return
            if self.filename is None:
                return

            with open(img, 'rb') as fits:
                data = np.load(fits)

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
        # Flag out NaNs and other badness...
        mask = np.isfinite(data)
        data[~mask] = 0.

        # ... so that this works. Otherwise the SOM will
        # come out empty
        mask = data < std*data.std()
        data[mask] = 0.
        return data

    def normalise(self, data):
        '''Normalise the input data in some manner

        data - numpy.ndarray
             The image that will be normalised to some scale
        '''
        return 1. * data / data.max()

    def log10(self, data):
        '''Apply a log transform onto the data array

        data - numpy.ndarray
              The image that will be transformed onto a log scale
        '''

        return numpy.log10(data)

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
            order = self.common_images.keys()

        for item in order:
            if item not in self.common_images.keys():
                raise ValueError
            img = self.common_images[item]
            with open(img, 'rb') as in_file:
                data = np.load(in_file)
                if norm:
                    data = self.normalise(data)
                if isinstance(sigma, float):
                    data = self.sigma_clip(data, std=sigma)
                
                if of is None:
                    return data.astype('float')
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

    def get_data(self, label=None, func=None):
        '''Function to return some property from each of the attached self.sources
        objects

        label - str
             Name of the field in the self.sources object to retreive
        func - Function
            Some type of python callable to apply to an instance of Source
        '''
        if label is None and func is None:
            raise ValueError('Label and Func can not both be None')

        if func is None:
            return [ s.info[label] for s in self.sources ]
        else:
            return [ func(s) for s in self.sources ] 

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
            self.sources.append(Source(self.out_dir, position=SkyCoord(ra=r['RA']*u.deg, dec=r['DEC']*u.deg), info=r) )

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
        for s in tqdm(self.sources):
            s.is_valid()

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

    def show_som(self, channel=0, split=False):
        '''Method to plot the trained SOM, and associated plotting options

        channel - int
             The channel from the SOM to plot. Defaults to the first (zero-index) channel
        split - Bool
             Slice the neurons into their own subplot axes objects from the returned data
             matrix from self.retrieve_som_data(). Otherwise just plot it on screen. 
        '''
        params = self.retrieve_som_data(channel=channel)
        if params is None:
            return
        (data, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height) = params

        if split:
            fig, ax = plt.subplots(SOM_width, SOM_height, figsize=(16,16), 
                                gridspec_kw={'hspace':0.001,'wspace':0.001,
                                            'left':0.001, 'right':0.999,
                                            'bottom':0.001, 'top':0.9})

            for x in range(SOM_width):
                for y in range(SOM_height):
                    d = data[x*neuron_width:(x+1)*neuron_width, 
                            y*neuron_width:(y+1)*neuron_width]
                    ax[x,y].imshow(d)
                    ax[x,y].get_xaxis().set_ticks([])
                    ax[x,y].get_yaxis().set_ticks([])

            fig.suptitle((f'Images: {len(self.binary.sources)} - Sigma: {self.binary.sigma} - Norm: {self.binary.norm}\n'
                            f'Channel: {channel}'))

        else:
            fig, ax = plt.subplots(1,1)

            im = ax.imshow(data)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])            
            fig.colorbar(im, label='Intensity')

        fig.savefig(f'{self.SOM_path}-ch_{channel}.pdf')

    def _process_heatmap(self, image_number=0, plot=False, channel=0, binary=None, save=None):
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
        save - None or str
               If None plot the window, otherwise save the figure to the path
               set by save
        '''
        if binary is None:
            binary = self.binary
        
        # I have modified the original version of PINK to also output
        # the transform information. Extract here if it exisists
        transform = f'{self.heat_path}.transform'
        if os.path.exists(transform):
            with open(transform, 'rb') as in_file:
                (numberOfImages, width, height, depth) = struct.unpack('i'*4, in_file.read(4*4))
                som_size = width * height * depth
                image_width = width
                image_height = height * depth

                in_file.seek(image_number * 8 * som_size + 4*4)
                transform_map = struct.unpack('fi' * som_size, in_file.read(4*2*som_size))
                angle = np.array(transform_map[::2])
                flipped = np.array(transform_map[1::2])
                angle = np.ndarray([width, height, depth], 'float', angle)
                angle = np.swapaxes(angle, 0, 2)
                angle = np.reshape(angle, (image_height, image_width))
                flipped = np.ndarray([width, height, depth], 'int', flipped)
                flipped = np.swapaxes(flipped, 0, 2)
                flipped = np.reshape(flipped, (image_height, image_width))

        else:
            angle, flipped = None, None

        with open(self.heat_path, 'rb') as in_file:
            numberOfImages, SOM_width, SOM_height, SOM_depth = struct.unpack('i' * 4, in_file.read(4*4))
            
            size = SOM_width * SOM_height * SOM_depth
            image_width = SOM_width
            image_height = SOM_depth * SOM_height

            # Seek the image number here
            in_file.seek(image_number * size * 4, 1)
            array = np.array(struct.unpack('f' * size, in_file.read(size * 4)))
            data = np.ndarray([SOM_width, SOM_height, SOM_depth], 'float', array)
            data = np.swapaxes(data, 0, 2)
            data = np.reshape(data, (image_height, image_width))

            # data /= data.sum()
            # Simple diagnostic plot
            if plot:
                # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
                fig, ax = plt.subplots(3,3)
                params = self.retrieve_som_data(channel=channel)
                if params is None:
                    return
                (som, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height) = params

                src_img = binary.get_image(image_number)
                loc = np.unravel_index(np.argmin(data, axis=None), data.shape)
                
                where = np.where(data==data.min())
                loc2 = (where[0][0], where[1][0])

                ax[0,0].imshow(data)
                ax[0,0].plot(loc2[0], loc2[1],'ro')
                ax[0,0].plot(loc2[1], loc2[0],'b^')
                ax[0,0].set(title='Euclidean Distance (Heatmap)')

                ax[0,1].imshow(som)
                ax[0,1].set(title='Trained SOM')

                ax[0,2].imshow(src_img)
                ax[0,2].set(title='Source Image')
            
                if angle is not None:
                    ang = np.rad2deg(angle[loc[0], loc[1]])
                    flip = flipped[loc[0], loc[1]]
                    
                    # TODO: This will fail if the data have not been
                    # normalised...
                    img = Image.fromarray((src_img*255).astype(np.uint8), mode='L')
                    
                    img = img.rotate(-ang)
                    if flip == 1:
                        # print('\tFlipping')
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        # img = img.transpose(Image.FLIP_LEFT_RIGHT)

                    ax[1,2].imshow(np.array(img))
                    ax[1,2].set(title=f'Rotation: {ang:.1f}, Flipped: {flip}')

                    x, y = loc
                    winning_neuron = som[y*neuron_width:(y+1)*neuron_width,
                                         x*neuron_width:(x+1)*neuron_width]
                    ax[1,1].imshow(winning_neuron)
                    ax[1,1].set(title='Red Marker')

                    y,x = x,y 
                    winning_neuron = som[y*neuron_width:(y+1)*neuron_width,
                                         x*neuron_width:(x+1)*neuron_width]
                    ax[1,0].imshow(winning_neuron)
                    ax[1,0].set(title='Blue Marker')

                    ax[2,0].imshow(np.rad2deg(angle))
                    ax[2,0].plot(loc[0], loc[1],'ro')
                    ax[2,0].plot(loc[1], loc[0],'b^')
                    ax[2,0].set(title='Rotation')

                    ax[2,1].imshow(flipped)
                    ax[2,1].plot(loc[0], loc[1],'ro')
                    ax[2,1].plot(loc[1], loc[0],'b^')
                    ax[2,1].set(title='Flipped Bit')

                    # print(angle.shape, flipped.shape, data.shape)
                    # print(angle[loc2[1],loc2[0]], np.rad2deg(angle[loc2[1],loc2[0]]), loc2 )

                else:
                    fig.delaxes(ax[1,2])
                    fig.delaxes(ax[1,1])
                    fig.delaxes(ax[1,0])
                
                fig.delaxes(ax[2,2])
                # plt.show() will block, but fig.show() wont
                if save is None:
                    plt.show()
                else:
                    plt.savefig(save)

                plt.close(fig)
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
        # exec_str = f'Pink --cuda-off --map {self.binary.binary_path} {self.heat_path} {self.SOM_path} '
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

    def _numeric_plot(self, book, shape, save=None):
        '''Isolated function to plot the attribute histogram if the data is 
        numeric in nature

        book - dict
            A dictionary whose keys are the location on the heatmap, and values
            are the list of values of sources who most belonged to that grid
        shape - tuple
            The shape of the grid. Should attempt to get this from the keys or
            possible recreate it like in self.attribute_heatmap()
        save - None or Str
            If None, show the figure on screen. Otherwise save to the path in save       
        '''
        # Step one, get range
        vmin, vmax = np.inf, 0
        for k, v in book.items():
            # Flatten the list out. The RGZ annotations can be comprised of 
            # multiple objects. For the moment, flatten. them. out. 
            if min(v) < vmin:
                vmin = min(v)
            if max(v) > vmax:
                vmax = max(v)
        bins = np.linspace(vmin, vmax, 10)

        fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1])

        for k, v in book.items():
            ax[k].hist(v, bins=bins)

        if save is None:
            plt.show()
        else:
            plt.savefig(save)

    def _label_plot(self, book, shape, save=None, xtick_rotation=None):
        '''Isolated function to plot the attribute histogram if the data is labelled in 
        nature

        book - dict
            A dictionary whose keys are the location on the heatmap, and values
            are the list of values of sources who most belonged to that grid
        shape - tuple
            The shape of the grid. Should attempt to get this from the keys or
            possible recreate it like in self.attribute_heatmap() 
        save - None or Str
            If None, show the figure on screen. Otherwise save to the path in save
        xtick_rotation - None or float
            Will rotate the xlabel by rotation
        '''
        # Step one, get unique items and their counts
        from collections import Counter
        unique_labels = []
        max_val = 0
        for k, v in book.items():
            v = [i for items in v for i in items]
            c = Counter(v)
            unique_labels.append(c.keys())
            mv = max(c.values())
            max_val = mv if mv > max_val else max_val

        # Work out the unique labels and sort them so that each 
        # sub pplot may be consistent. Calling Counter object 
        # with a key that doesnt exits returns 0. Exploit this 
        # when plotting the bars
        unique_labels = list(set([u for labels in unique_labels for u in labels]))
        unique_labels.sort()

        fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1])

        for k, v in book.items():
            v = [i for items in v for i in items]
            c = Counter(v)
            s = sum(c.values())

            ax[k].bar(np.arange(len(unique_labels)),
                      [c[l]/s for l in unique_labels],
                      align='center',
                      tick_label=unique_labels)
            ax[k].set(ylim=[0,1])
            if k[1] != 0:
                ax[k].set(yticklabels=[])
            if k[0] != shape[1]-1:
                ax[k].set(xticklabels=[])
            else:
                if xtick_rotation is not None:
                    ax[k].tick_params(axis='x', rotation=xtick_rotation)

        # fig.tight_layout()
        if save is None:
            plt.show()
        else:
            plt.savefig(save)

    def attribute_plot(self, book, shape, *args, **kwargs):
        '''Produce a grid of histograms based on the list of items inside it
        
        book - dict
            A dictionary whose keys are the location on the heatmap, and values
            are the list of values of sources who most belonged to that grid
        shape - tuple
            The shape of the grid. Should attempt to get this from the keys or
            possible recreate it like in self.attribute_heatmap()
        '''
        # Not covinced this is the best way to do it. Perhaps just a 
        # try: -> except: ?
        if isinstance(book[(0,0)][0], (int, float, complex)):
            self._numeric_plot(book, shape, **kwargs)
        else:
            self._label_plot(book, shape, **kwargs)
            
    def attribute_heatmap(self, label=None, plot=True, func=None, *args, **kwargs):
        '''Based on the most likely grid/best match in the heatmap for each source
        build up a distibution plot of each some label/parameters

        label - str
             The label/value to extract from each source
        plot - bool
             Plot the attribute heatmap/distribution
        func - Function or callable
             Function that may be applied to each of the instances of Source
        '''
        shape = self.src_heatmap[0].shape
        items = self.binary.get_data(label=label, func=func)

        book = defaultdict(list)

        for heat, item in zip(self.src_heatmap, items):
            loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)
            book[loc].append(item)

        if plot:
            self.attribute_plot(book, shape, **kwargs)

    def count_map(self, plot=False, save=None):
        '''Produce a map of the number of images that best match each neuron. This
        will have the same shape as the SOM grid, and the counts in each cell should
        add to the number of images in the Binary file. For now, just use the heatmap
        attached to this Pink instance.

        plot - Bool
            Produce a figure of the counts per neuron
        save - None or Str
            If None, show the figure onscreen. Otherwise save it to the filename in 
            save
        '''
        shape = self.src_heatmap[0].shape
        book = np.zeros(shape)

        for heat in self.src_heatmap:
            loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)
            book[loc] += 1           

        # Diagnostic plot. Not meant to be final...
        if plot:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            params = self.retrieve_som_data(channel=0)
            if params is None:
                return
            (data, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height) = params


            fig, ax = plt.subplots(1,2)

            im = ax[0].imshow(book)
            ax[0].set(title='Counts per Neuron')
            ax[0].xaxis.set(ticklabels=[])
            ax[0].yaxis.set(ticklabels=[])
            
            divider = make_axes_locatable(ax[0])
            cax0 = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax0, label='Counts')
            # cax0.set(label='Counts')

            im2 = ax[1].imshow(data)
            ax[1].set(title='Trained SOM')
            ax[1].xaxis.set(ticklabels=[])
            ax[1].yaxis.set(ticklabels=[])
            
            divider = make_axes_locatable(ax[1])
            cax1 = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax1, label='Intensity')
            
            fig.tight_layout()
            # fig.show()
            if save is None:
                plt.show()
            else:
                plt.savefig(save)

        return book

if __name__ == '__main__':

    if '-r' in sys.argv:
        rgz_dir = 'rgz_rcnn_data'

        cat = Catalog(rgz_dir=rgz_dir)

        cat.save_sources()

        print('\nValidating sources...')
        cat.collect_valid_sources()

        test_bin = cat.dump_binary('TEST.binary', norm=True, sigma=3.)

        print(test_bin)

        pink = Pink(test_bin, pink_args={'som-width':8,
                                         'som-height':8}) 

        pink.train()
        pink.save('TEST.pink')

        # ------------------

        test_bin = cat.dump_binary('TEST_chan.binary', norm=True, channels=['FIRST','WISE_W1'])

        print(test_bin)

        pink = Pink(test_bin, pink_args={'som-width':8,
                                         'som-height':8}) 

        pink.train()        
        pink.save('TEST2.pink')
        # pink.heatmap(plot=True, image_number=0, apply=False)
        # pink.heatmap(plot=True, image_number=500, apply=False)
        
    elif '-t' in sys.argv:
        pink = Pink.loader('TEST2.pink')
        src = len(pink.binary.sources)

        pink.show_som(channel=0)
        pink.show_som(channel=1)

        plot_dir = 'Source_Heatmaps'
        make_dir(plot_dir)

        pink.heatmap(plot=False, apply=True)

        def source_rgz(s):
            # If there is only one object, its returned as dict. Test and list it if needed            
            a = s.rgz_annotations()
            if a is None:
                return ''
            else:
                a = a['object']
                if not isinstance(a, list):
                    a = [a]
                return str(len(a))  
        pink.attribute_heatmap(func=source_rgz, save='example_chan_number_counts.pdf')

        def source_rgz(s):
            # If there is only one object, its returned as dict. Test and list it if needed
            a = s.rgz_annotations()
            if a is None:
                return ''
            else:
                a = a['object']
                if not isinstance(a, list):
                    a = [a]
                return [ i['name'] for i in a ]
        pink.attribute_heatmap(func=source_rgz, xtick_rotation=45, save='example_chan_component_counts.pdf')

        pink.count_map(plot=True, save='example_chan_count_map.pdf')

        # for i in tqdm(range(10, 100)):
        #     pink.heatmap(plot=True, image_number=i, apply=False)#, save=f'{plot_dir}/{i}_heatmap.pdf')
            

        # pink.heatmap(plot=True, image_number=60, apply=False)
        # pink.heatmap(plot=True, image_number=61, apply=False)
        # pink.heatmap(plot=True, image_number=62, apply=False)
        # pink.heatmap(plot=True, image_number=63, apply=False)
        # pink.heatmap(plot=True, image_number=64, apply=False)
        # pink.heatmap(plot=True, image_number=65, apply=False)

        # pink.heatmap(plot=True, image_number=src-1, apply=False)
        # pink.heatmap(plot=True, image_number=500, apply=False)
        # pink.heatmap(plot=True, image_number=100, apply=False)
        # pink.heatmap(plot=True, image_number=300, apply=False)
        # pink.heatmap(plot=True, image_number=400, apply=False)
        # pink.heatmap(plot=True, image_number=450, apply=False)
        
    else:
        print('Options:')
        print(' -r : Run test code to scan in RGZ image data')
        print(' -t : Run test code for the Transform outputs and heatmap')
