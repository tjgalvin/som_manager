'''A set of management class to interact with PINK. This includes reading in 
and numpy objects, creating binary files, training/mapping PINK and producing
results
'''
import matplotlib as mpl
mpl.use('agg')

import io
import os
import sys
import glob
import shutil
import pickle
import struct
import random
import subprocess
import pandas as pd
import xmltodict as xd
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from astropy.table import Table
from astropy.io import fits as pyfits
from skimage import morphology as skm

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
    # Added this to allow dry run testing of the training learning
    # modes parameter
    if not os.path.exists(f):
        return 'NO_EXISTING_FILE'

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
    def _path_build(self, out_file):
        '''Helper function to make a output file path if the self.project_dir
        attribute exists
        '''
        if 'project_dir' in dir(self):
            out_file = f'{self.project_dir}/{out_file}'

        return out_file

    def save(self, out_path, disable_project_dir=False):
        '''A function to save this class instance to disk.

        out_path - str
              The path to write the pickled output to
        disable_projet_dir - False
              By default this function will look for a project_dir attribute
              to save to. This will disable that behaviour
        '''
        if not disable_project_dir:
            if 'project_dir' in dir(self):
                make_dir(self.project_dir)
                out_path = f'{self.project_dir}/{out_path}'

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

    def __repr__(self):
        return self.__str__()

class Source(Base):
    '''Object class to handle a single object from a catalogue, downloading
    images, saving images, reprojecting them onto a common grid, and dumping
    them into a binary file
    '''
    def __str__(self):
        '''Make the string output meaningful
        '''
        out = self.rgz_path
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
        return (data - data.min()) / (data.max() - data.min())
        # return 1. * data / data.max()

    def log10(self, data):
        '''Apply a log transform onto the data array

        data - numpy.ndarray
              The image that will be transformed onto a log scale
        '''
        # This will almost certainly throw a runtime warning on non-normalised
        # images. You cant log a negative number. It should be safe to ignore. 
        data = np.log10(data)
        mask = np.isfinite(data)
        
        data[~mask] = np.nanmin(data[mask].flatten())

        return data

    def convex_hull(self, data, limit=0.):
        '''Derive the convex hull mask of the data array, where values are above
        the `limit` value

        data - numpy.ndarray
            Work out the convex hull of the data array 
        '''
        return skm.convex_hull.convex_hull_image(data>limit)

    def dump(self, of, order=None, sigma=False, norm=False, log10=False, convex=False):
        '''Dump the contents of the common_images to a binary file 
        provided by the of file handler

        of - file or None
           File to write the binary images to. If none, return data, otherwise
           write it to the file
        order - list
             Order of the keys to write out to
        sigma - False or float or list
             If False, there will be no sigma clipping performed on the 
             data. If float, then it will be sigma clipped to that level. If list, then 
             each index in the list is used to evaluate the condition for the corresponding
             channel being dumped.
        norm - bool or list
             Normalise the data so the min/max is 0 and 1. If listm then each index
             in the list is used to evaluate the condition for the corresponding 
             channel being dumped
        log10 - bool or list
             Log the data. For values which are negative, so come out as NaN, replace
             with the smallest non-Nan value. If list, then then each index in the list
            is used to evaluate the confition for the corresponding channel being dumped
        convex - bool
            Will apply a convex hull mask to images. The mask is construted using the first
            channel image, and then applied to all subsequent channels. There is no need 
            for convex to be tested for a list or not. 
        '''
        if order is None:
            order = self.common_images.keys()

        hull = None

        for count, item in enumerate(order):
            if item not in self.common_images.keys():
                raise ValueError
            img = self.common_images[item]
            with open(img, 'rb') as in_file:
                data = np.load(in_file)
                
                if isinstance(sigma, (float, int)):
                    data = self.sigma_clip(data, std=sigma)
                elif isinstance(sigma, list) and isinstance(sigma[count], (float, int)) :
                    data = self.sigma_clip(data, std=sigma[count])

                if (isinstance(log10, bool) and log10) or (isinstance(log10, list) and log10[count]):
                    data = self.log10(data)
                if (isinstance(norm, bool) and norm) or (isinstance(norm, list) and norm[count]):
                    data = self.normalise(data)
                
                if convex:
                    if hull is None:
                        hull = self.convex_hull(data)
                    else:
                        data[~hull] = data[hull].mean()

                if of is None:
                    return data.astype('float')
                else:
                    data.astype('f').tofile(of)

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

    def __init__(self, binary_path, sources, sigma, norm, log10, convex, channels='', project_dir='.'):
        '''Create and track the meta-information of the binary

        binary_path - str
             The path to the binary file
        source - list
             A list of the Source objects packed into the binary_path file
        sigma - False of float
            The option passed to each of the Source.dump() methods
        norm - bool
            The option passed to each of the Source.dump() methods
        log10 - bool or list
            The option passed to each of the Source.dump() methods
        convex - bool
            The option passed to each of the Source.dump() methods
        channels - list
            A list of the channels/surveys packed into the binary_path file
        project_dir - str
            The directory to consider as the dumping ground for this 
            binary and associated high level data products
        '''
        self.sources = sources
        self.binary_path = binary_path
        self.binary_hash = get_hash(self.binary_path)
        self.channels = channels
        self.sigma = sigma
        self.norm = norm
        self.log10 = log10
        self.convex = convex
        self.project_dir = project_dir

        if project_dir != '.':
            make_dir(project_dir)

        self.heat_path = {}
        self.heat_hash = {}
        self.src_heatmap = {}

        # At most, a Binary instance can only have a single attached SOM file
        self.SOM_path = f'{self.binary_path}.Trained_SOM'
        self.SOM_hash = ''
        self.trained = False

    @property
    def preprocessor_args(self):
        '''Reterive the pre-processor arguments of this Binary file
        '''
        return {'norm': self.norm, 'log':self.log10, 'sigma':self.sigma,
                'convex':self.convex}

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
        raise ValueError('Creating objects from Catalogue directly not supported')
        print('Generating the Sources....')
        for index, r in tqdm(self.tab.iterrows()):
            self.sources.append(Source(self.out_dir, position=SkyCoord(ra=r['RA']*u.deg, dec=r['DEC']*u.deg), info=r) )

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

    def _write_binary(self, binary_out, channels=['FIRST'], sigma=False, norm=False, log10=False, 
                            convex=False, project_dir='.', sources=None):
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
        log10 - bool
             Log the data. For invalid values, insert the smallest non-NaN value in the data
        convex - bool
             Derive a convex hull mask from the first channel image, and apply the mask to subsequent
             channels that are being dumped
        project_dir - str
            The directory to consider as the dumping ground for this 
            binary and associated high level data products
        sources - None or list of Source()
            If None, use all the sources in the self.valid_sources() attribute as the items to 
            dump. Otherwise, use the objects passed in the sources keyword
        '''
        if isinstance(channels, str):
            channels = [channels]

        if sources is None:
            sources = self.valid_sources

        # Lets find the total set of image sizes
        img_sizes = list(set([s.common_shape for s in sources]))        

        # TODO: Will want to eventually get the smallest number in each direction
        # to use as the image dimensions. For the moment ignore the problem
        assert len(img_sizes) == 1, 'Different image sizes, not sure what to do yet'
        x_dim, y_dim = img_sizes[0][0], img_sizes[0][1]

        if project_dir != '.':
            make_dir(project_dir)
            binary_out = f'{project_dir}/{binary_out}'

        print(f'\nDumping to: {binary_out}')
        print(f'The number of images to dump: {len(sources)}')
        print(f'The number of channels to dump: {len(channels)}')
        print(f'The x and y dimensions: {x_dim}, {y_dim} ')
        print(f'Sigma clipping: {sigma}')
        print(f'Normalised: {norm}')
        print(f'Log10: {log10}')
        print(f'Convex hull masking: {convex}')
        with open(binary_out, 'wb') as out_file:
            out_file.write(struct.pack('i', len(sources)))
            out_file.write(struct.pack('i', len(channels)))
            out_file.write(struct.pack('i', x_dim))
            out_file.write(struct.pack('i', y_dim))

            for s in tqdm(sources):
                s.dump(out_file, order=channels, sigma=sigma, norm=norm, log10=log10, convex=convex)

        return Binary(binary_out, sources, sigma, norm, log10, convex, channels=channels, project_dir=project_dir)

    def dump_binary(self, binary_out, *args, fraction=None, segments=None, **kwargs):
        '''This function wraps around the self._write_binary() function. If we are splitting 
        the data into a training and validation set, do it here, and pass through the relevant
        information to the _write_binary() method, including the sources for each set. Modify
        the name in place as we do this. 

        binary_out - str
               The name of the binary file to write out
        fraction - None or float
              If fraction is None, write out all files to the single binary file. If it is a
              float, between 0 to 1, split it into the training and validation sets, and return
              a Binary instance for each
        segments - None or Int
             The number of data partitions to make from the training data. One will be kept
             for validation. For instance, if segments=5, four will be used for training and
             one will be for validation
        '''
        import copy

        if fraction is None and segments is None:
            return self._write_binary(binary_out, *args, **kwargs)

        if fraction is not None:
            assert 0. <= fraction <= 1., ValueError('Fraction has to be between 0 to 1')

            # First shuffle the list. Make a deep copy to ensure not funny business happens
            # calling dump binary multiple time. I don't think it would happen but I ***think***
            # this helps ensure it
            cp_valid_sources = copy.deepcopy(self.valid_sources)
            random.shuffle(cp_valid_sources)

            # Next calculate the size of the spliting to do
            pivot = int(len(cp_valid_sources)*fraction)
            train = cp_valid_sources[:pivot]
            validate = cp_valid_sources[pivot:]

            print(f'Length of the training set: {len(train)}')
            print(f'Length of the validate set: {len(validate)}')

            train_bin = self._write_binary(f'{binary_out}_train', *args, sources=train, **kwargs)
            validate_bin = self._write_binary(f'{binary_out}_validate', *args, sources=validate, **kwargs)

            return (train_bin, validate_bin)
        
        elif segments is not None:
            # First shuffle the list
            cp_valid_sources = copy.deepcopy(self.valid_sources)
            random.shuffle(cp_valid_sources)

            stride = len(cp_valid_sources) // segments

            train_segments = [ cp_valid_sources[i*stride:i*stride+stride] for i in range(segments-1) ]
            validate = cp_valid_sources[(segments-1)*stride:]

            print('Length of valid_sources: ', len(self.valid_sources))
            print('Number of segments: ', segments)
            for count, i in enumerate(train_segments):
                print(f'Train {count} : {len(i)}')
            print('Valid : ', len(validate))

            train_bin = [ self._write_binary(f'{binary_out}_train_{count}', *args, sources=train, **kwargs) \
                                for count, train in enumerate(train_segments) ]
            validate_bin = self._write_binary(f'{binary_out}_validate', *args, sources=validate, **kwargs)
            
            return (train_bin, validate_bin)

class Pink(Base): 
    '''Manage a single Pink training session, including the arguments used to run
    it, the binary file used to train it, and interacting with the results afterwards
    '''
    def __str__(self):
        '''Neatly print the contents of the instance
        '''
        out = f'{len(self.binary.binary_path)} binary files attached\n'
        
        return out

    def __init__(self, binary, pink_args = {}, validate_binary=None):
        '''Create the instance of the Pink object and set appropriate parameters
        
        binary - Binary
             An instance of the Binary class that will be used to train Pink
        pink_args - dict
             The arguments to supply to Pink
        validate_binary - Binary
             The Binary object that will be used to validate the results of the training agaisnt
        '''
        if not (isinstance(binary, Binary) or isinstance(binary, list)):
            raise TypeError(f'binary is expected to be instance of Binary or list class, not {type(binary)}')

        self.trained = False

        if isinstance(binary, Binary):
            binary = [binary]
        self.binary = binary
        # Lets just assume all will have the same project_dir. 
        self.project_dir = self.binary[0].project_dir

        # Items to generate the SOM
        self.exec_str = ['' for b in self.binary]

        self.validate_binary = validate_binary

        if pink_args:
            self.pink_args = pink_args
        else:
            self.pink_args = {'som-width':10,
                              'som-height':10}

    def _reterive_binary(self, mode):
        '''Helper function to reterive a binary file from. This will return a validation
        binary, or one of the training binaries. At the very least, there will always be
        one training binary

        mode - str or int
             If str, it will return the self.validate_binary if `validate` specified, or return
             then first self.binary item if `train` specified. If an int is provided, then that
             is used as an index to self.binary
        '''
        if not (isinstance(mode, str) or isinstance(mode, int)):
            raise ValueError(f'binary mode {mode} not supported. Supported modes are either `train`, `validate` or an integer index')
        elif mode == 'train':
            return self.binary[0]
        elif mode == 'validate':
            return self.validate_binary
        else:
            return self.binary[mode]

    def update_pink_args(self, **kwargs):
        '''Helper function to update pink arguments that may not have been included
        when creating the class instance
    
        Use the kwargs and simply update the pink_args attribute
        '''
        self.pink_args.update(kwargs)

    def _train(self, binary=None, SOM_path=None, learning=None):
        '''Train the SOM with PINK using the supplied options and Binary file
        
        binary - None or Binary
             If None, use the self.binary attribute. Otherwise, try on the provided
             binary
        SOM_path - None or str
             If None, use the self.SOM_path attribute. Otherwise, try on the provided
             SOM_path value
        learning - None or list of tuples
             If None, use a default learning mode. Otherwise, iterate over the 
             list and perform each specific learning mode, daisy chaining the 
             output of the previous PINK as the input into the next. 
        '''
        if self.trained:
            print('The SOM has been trained already')
            return
        
        if not os.path.exists(binary.binary_path):
            raise ValueError(f'Unable to locate {binary.binary_path}')
        
        if binary.binary_hash != get_hash(binary.binary_path):
            raise ValueError(f'The hash checked failed for {binary.binary_path}')

        if learning is None:
            learning = [('gaussian', '3.' , '0.2')]
        
        inter_stages = []

        pink_avail = True if shutil.which('Pink') is not None else False
        
        for count, mode in enumerate(learning):
            if count == len(learning) - 1:
                out_path = binary.SOM_path
            else:
                out_path = f'{binary.SOM_path}.Stage_{count}'
            
            if count > 0:
                init = f"--init={inter_stages[-1]['path']}"
            else:
                init = '--init=zero'

            exec_str  = f'Pink --train {binary.binary_path} {out_path} '
            exec_str += ' '.join(f'--{k}={v}' for k,v in self.pink_args.items())
            exec_str += f' {init}'
            exec_str += ' --dist-func ' + ' '.join(mode)

            if pink_avail:
                print('\n', exec_str)
                subprocess.run(exec_str.split())

                info = {'path':out_path, 'hash':get_hash(out_path), 'exec_str':exec_str, 'learning':mode}
                inter_stages.append(info)
                print(info['hash'])
                if count == len(learning) - 1:
                    print('Attaching final SOM information')
                    binary.SOM_hash = info['hash']
                    binary.trained = True
                    binary.inter_stages = inter_stages
            else:
                print(exec_str)

    def train(self, **kwargs):
        '''Wrapper around the _train() method that will actual perform the training. Here
        we do the working to handle the case of multiple training binaries, whic is the case
        when the segments option is used. It should be fairly transparent to the calling 
        function. 
        '''
        for count, train in enumerate(self.binary):
            self._train(binary=train, **kwargs)
            train.trained = True

        self.trained = True

    def retrieve_som_data(self, channel=0, count=0, iteration=None):
        '''If trained, this function will return the SOM data from some desired
        channel

        channel - int or None
             The channel from the some to retrieve. If a negative number or None
             return the entire structure, otherwise return that channel number
        count - int
             The trained binary from which to pull the SOM from. This is important
             if cross validation has been used
        iteration - None or int
             The intermidiate  SOM saved by PINK. If None, open the SOM in SOM_path
        '''
        binary = self._reterive_binary(count)
        
        if not binary.trained:
            # print(f'{binary.binary_path} not trained... Returning...')
            return None
        
        if get_hash(binary.SOM_path) != binary.SOM_hash:
            # print(f'{binary.SOM_path} hash not matching... Returning...')
            return None

        path = binary.SOM_path
        if iteration is not None:
            # Assume there is some type of extension for the file type
            name, file_type = path.rsplit('.', 1)
            path = f'{name}_{iteration}.{file_type}'
            print(f'Path is {path}')

        with open(path, 'rb') as som:
            # Unpack the header information
            numberOfChannels, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height = struct.unpack('i' * 6, som.read(4*6))
            SOM_size = np.prod([SOM_width, SOM_height, SOM_depth])

            # Check to ensure that the request channel exists. Remeber we are comparing
            # the index
            if channel > numberOfChannels - 1:
                # print(f'Channel {channel} larger than {numberOfChannels}... Returning...')
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

            return (binary, data, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height)

    def show_som(self, mode='train', channel=0, iteration=None, plt_mode='raw', color_map='bwr'):
        '''Method to plot the trained SOM, and associated plotting options

        mode - str or int
             Type passed through to _reterive_binary()
        channel - int
             The channel from the SOM to plot. Defaults to the first (zero-index) channel
        iteration - None or int
             The intermidiate  SOM saved by PINK. If None, open the SOM in SOM_path
        plt_mode - str
             Mode to print the SOM on. 
             `split` - Slice the neurons into their own subplot axes objects from the returned data
                       matrix from self.retrieve_som_data(). 
             `grid` - Plot each channel on its own subfigure
             `raw`  - Otherwise just plot it on screen. 
        color_map - str
            The name of the matplotlib.colormap that will be passed directly to matplotlib.pyplot.get_map()
        '''
        import matplotlib as mpl

        params = self.retrieve_som_data(channel=channel, count=mode,iteration=iteration)
        if params is None:
            # print('Params is None... Returning...')
            return
        (binary, data, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height) = params

        if plt_mode == 'split':
            fig, ax = plt.subplots(SOM_width, SOM_height, figsize=(16,16), 
                                gridspec_kw={'hspace':0.001,'wspace':0.001,
                                            'left':0.001, 'right':0.999,
                                            'bottom':0.001, 'top':0.95})

            for x in range(SOM_width):
                for y in range(SOM_height):
                    d = data[x*neuron_width:(x+1)*neuron_width, 
                            y*neuron_width:(y+1)*neuron_width]
                    ax[x,y].imshow(d)
                    ax[x,y].get_xaxis().set_ticks([])
                    ax[x,y].get_yaxis().set_ticks([])

            fig.suptitle(f'{binary.channels[channel]}')
            fig.savefig(f'{binary.SOM_path}-ch_{channel}-split.pdf')

        elif plt_mode == 'raw':
            fig, ax = plt.subplots(1,1)

            im = ax.imshow(data, cmap=plt.get_cmap(color_map), norm=mpl.colors.SymLogNorm(0.03))
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])  
            ax.set(title=f'{binary.channels[channel]} Layer')          
            fig.colorbar(im, label='Intensity')
            fig.savefig(f'{binary.SOM_path}-ch_{channel}.pdf')

        elif plt_mode == 'grid':
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            chans = len(binary.channels)
            
            cols = 2 if chans > 1 else 1
            rows = int(chans/cols + 0.5)

            fig, axes = plt.subplots(rows, cols)
            for c, ax in enumerate(fig.axes):
                params = self.retrieve_som_data(channel=c, count=mode)
                if params is None:
                    return
                (binary, data, SOM_width, SOM_height, SOM_depth, neuron_width, neuron_height) = params

                ax.set(title=binary.channels[c])

                im = ax.imshow(data, cmap=plt.get_cmap(color_map))
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])  

                divider = make_axes_locatable(ax)
                cax0 = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax0, label='Intensity')

            fig.tight_layout()
            fig.savefig(f'{binary.SOM_path}-grid.pdf')

    def src_heatmap_plot(self, index=0, mode='train', SOM_mode=None, color_map='bwr'):
        '''Simple function to produce a image of the source passed to PINK, and a 
        corresponding map showing its distance from the neurons

        index - int
             The index of the source to plot
        mode - `train`, `validate` or int
             Mode passed to self._reterive_binary()
        SOM_mode - `train`, `validate` or int
             The name of the binary containing the SOM which the sources have been
             mapped agaisnt
        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        save = self._path_build('TEST.png')

        binary = self._reterive_binary(mode)
        if SOM_mode is None:
            SOM_path = binary.SOM_path
        else:
            SOM_binary = self._reterive_binary(SOM_mode)
            SOM_path = SOM_binary.SOM_path

        src_img_ch0 = binary.get_image(index=index, channel=0)
        src_img_ch1 = binary.get_image(index=index, channel=1)

        fig = plt.figure(figsize=(5,5))

        ax0 = plt.subplot2grid((2,2), (0,0))
        im = ax0.imshow(src_img_ch0, cmap=plt.get_cmap(color_map))
        ax0.set(title=binary.channels[0])
        ax0.get_xaxis().set_ticks([])
        ax0.get_yaxis().set_ticks([])

        divider = make_axes_locatable(ax0)
        cax0 = divider.append_axes('left', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax0, label='Intensity')
        cax0.yaxis.set_ticks_position('left')
        cax0.yaxis.set_label_position('left')

        if src_img_ch1 is not None:
            ax1 = plt.subplot2grid((2,2), (0,1))
            im = ax1.imshow(src_img_ch1, cmap=plt.get_cmap(color_map))
            ax1.set(title=binary.channels[1])
            ax1.get_xaxis().set_ticks([])
            ax1.get_yaxis().set_ticks([])

            divider = make_axes_locatable(ax1)
            cax1 = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax1, label='Intensity')
        
        heatmap = binary.src_heatmap[SOM_path][index]
        ax2 = plt.subplot2grid((2,2), (1,0))
        im = ax2.imshow(heatmap)
        ax2.get_xaxis().set_ticks([])
        ax2.get_yaxis().set_ticks([])
        
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('left', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax2, label='Euclidean Distance')
        cax2.yaxis.set_ticks_position('left')
        cax2.yaxis.set_label_position('left')


        ax3 = plt.subplot2grid((2,2), (1,1))
        prob = 1. / heatmap**10
        prob = prob / prob.sum()

        im = ax3.imshow(prob)
        ax3.get_xaxis().set_ticks([])
        ax3.get_yaxis().set_ticks([])
        
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax3, label='Likelihood')

        fig.tight_layout(h_pad=0.1, w_pad=0.1)
        plt.savefig(save)

    def _process_heatmap(self, heat_key, image_number=0, plot=False, channel=0, binary=None, save=None):
        '''Function to process the heatmap file produced by the `--map`
        option in the Pink utility

        heat_key - str
            The key passed to _process_heatmap() that will open the correct map file
            that corresponds to the distances of SOM.SOM_path
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
            binary = self.binary[0]
        
        if save is not None:
            save = self._path_build(save)

        with open(binary.heat_path[heat_key], 'rb') as in_file:
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

            return data

    def _apply_heatmap(self, binary, SOM):
        '''Function to loop through the Pink map output (i.e. self.heatmap) and 
        read in as a list each source heatmap

        binary - Binary
             An instance of the Binary class. Is influenced by the mode of the calling
             function
        SOM - Binary
            An instance of the Binary class used to provide the trained map

        '''
        result = []
        for index, src in enumerate(binary.sources):
            result.append(self._process_heatmap(SOM.SOM_path, binary=binary, image_number=index))
        
        binary.src_heatmap[SOM.SOM_path] = result

    def map(self, mode='train', plot=False, apply=True, SOM_mode=None, **kwargs):
        '''Using Pink, produce a heatmap of the input Binary instance. 
        Note that by default the first training Binary instance attached to self.binary 
        will be used. 

        mode - `train` or `validate`
             Specify which of the attached Binary instances we should map and process.
             If neither mode is selected, than raise an error
        plot - bool
             Make an initial diagnostic plot of the SOM and the correponding heatmap.
             This will show the first source of the binary object
        apply - bool
             Add an attribute to the class instance with the list of heatmaps
        SOM_mode - str or None
             If None, the path to the SOM to use when mapping will be used from the 
             binary objected returned with the `mode` options. Otherwise, attempt
             to load the SOM from the binary returned with SOM and _reterive_binary()
        kwargs - dict
             Additional parameters passed directly to _process_heatmap()
        '''
        binary = self._reterive_binary(mode)
        
        if SOM_mode is None:
            SOM_binary = binary
            heat_path = f'{binary.binary_path}.{mode}.heat'
            heat_key = binary.SOM_path
            binary.heat_path[heat_key] = heat_path
        else:
            SOM_binary = self._reterive_binary(SOM_mode)
            heat_path = f'{binary.binary_path}.{SOM_mode}.heat'            
            heat_key = SOM_binary.SOM_path
            binary.heat_path[SOM_binary.SOM_path] = heat_path
            
        if not self.trained:
            return
        if SOM_binary.SOM_hash != get_hash(SOM_binary.SOM_path):
            raise ValueError(f'The hash checked failed for {SOM_binary.SOM_path}')        
        if binary.binary_hash != get_hash(binary.binary_path):
            raise ValueError(f'The hash checked failed for {binary.binary_path}')

        pink_avail = True if shutil.which('Pink') is not None else False        
        # exec_str = f'Pink --cuda-off --map {self.binary.binary_path} {self.heat_path} {self.SOM_path} '
        exec_str = f'Pink --map {binary.binary_path} {heat_path} {SOM_binary.SOM_path} '
        exec_str += ' '.join(f'--{k}={v}' for k,v in self.pink_args.items())
        
        if pink_avail:
            if not os.path.exists(heat_path):
                subprocess.run(exec_str.split())
                binary.heat_hash[heat_key] = get_hash(heat_path)
            else:
                print('Not running PINK to map, file exists.\n')
            if plot:
                self._process_heatmap(plot=plot, binary=binary, **kwargs)
            if apply:
                self._apply_heatmap(binary, SOM_binary)
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
        save = self._path_build(save)
                
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

    def _label_plot(self, book, shape, save=None, xtick_rotation=None, 
                    color_map='gnuplot2', title=None, weights=None, figsize=(6,6),
                    literal_path=False):
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
        color_map - str
            The name of the matplotlib.colormap that will be passed directly to matplotlib.pyplot.get_map()
        title - None of str
            A simple title strng passed to fig.suptitle()
        weights - None or dict
            If not None, the dict will have keys corresponding to the labels, and contain the total
            set of counts from the Binary file/book object. This will be used to `weigh` the contribution
            per neuron, to instead be a fraction of dataset type of statistic. 
        figsize - tuple of int
            Size of the figure to produce. Passed directly to plt.subplots
        literal_path - bool
            If true, take the path and do not modify it. If False, prepend the project_dir path
        '''
        if not literal_path:
            save = self._path_build(save)

        # Need access to the Normalise and ColorbarBase objects
        import matplotlib as mpl
        # Step one, get unique items and their counts
        from collections import Counter
        unique_labels = []
        max_val = 0
        for k, v in book.items():
            v = [i for items in v for i in items]
            c = Counter(v)
            unique_labels.append(c.keys())
            
            # Guard agaisnt empty most similar neuron
            if len(v) > 0:
                mv = max(c.values())
                max_val = mv if mv > max_val else max_val

        # Work out the unique labels and sort them so that each 
        # sub pplot may be consistent. Calling Counter object 
        # with a key that doesnt exits returns 0. Exploit this 
        # when plotting the bars
        unique_labels = list(set([u for labels in unique_labels for u in labels]))
        unique_labels.sort()

        cmap = plt.get_cmap(color_map)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)

        fig, ax = plt.subplots(nrows=shape[0], ncols=shape[1], figsize=figsize)

        # Set empty axis labels for everything
        for a in ax.flatten():
            a.set(xticklabels=[], yticklabels=[])

        # Keep track of range for the color bar normalisation
        norm_max = 0  
        for k, v in book.items():
            v = [i for items in v for i in items]
            c = Counter(v)
            s = sum(c.values())

            # Guard agaisnt most similar empty neuron
            if s > 0:
                # If we have weights, use them to weight the value of the labels
                # otherwise just weight by the sum of items in the neuron
                if weights is not None:
                    values = [c[l]/weights[l] for l in unique_labels]
                    norm_max = max(values) if max(values) > norm_max else norm_max

                    color = cmap(values)
                else:
                    norm_max = 1
                    color = cmap([c[l]/s for l in unique_labels])

                ax[k].bar(np.arange(len(unique_labels)),
                         [1]*len(unique_labels),
                         color=color,
                         align='center',
                         tick_label=unique_labels)

            ax[k].set(ylim=[0,1])
            if k[1] != -1: # disable this for now.
                ax[k].set(yticklabels=[])
            if k[0] != shape[1]-1:
                ax[k].set(xticklabels=[])
            else:
                if xtick_rotation is not None:
                    ax[k].tick_params(axis='x', rotation=xtick_rotation)
                    for item in ax[k].get_xticklabels():
                        item.set_fontsize(7.5)

        fig.subplots_adjust(right=0.83)
        cax = fig.add_axes([0.85, 0.10, 0.03, 0.8])

        # Need to calculate the values, then create the color map with the
        # attached norm object, then plot, then this will work correctly. 

        # norm = mpl.colors.Normalize(vmin=0, vmax=norm_max)        
        cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

        if weights is None:
            cb1.set_label('Fraction Contributed per Neuron')
        else:
            cb1.set_label('Fraction of Dataset')
            
        if title is not None:
            fig.suptitle(title)

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
        # if isinstance(book[(0,0)][0], (int, float, complex)):
        #     self._numeric_plot(book, shape, **kwargs)
        # else:
        #     self._label_plot(book, shape, **kwargs)

        try:
            self._numeric_plot(book, shape, **kwargs)
        except:
            self._label_plot(book, shape, **kwargs)
            
    def attribute_heatmap(self, realisations=1, mode='train', label=None, plot=True, 
                                func=None, SOM_mode=None, weights=False, *args, **kwargs):
        '''Based on the most likely grid/best match in the heatmap for each source
        build up a distibution plot of each some label/parameters

        realisations - int
             The number of times to use the heatmap to select a position to put something
        mode - `train` or `validate`
             Specify which of the attached Binary instances we should map and process.
             If neither mode is selected, than raise an error
        label - str
             The label/value to extract from each source
        plot - bool
             Plot the attribute heatmap/distribution
        func - Function or callable
             Function that may be applied to each of the instances of Source
        global_normalise - bool - TODO: Remove this option. Think it does nothing.
             Produce counts of each label across the entire dataset, idea being
             that they are than used to `weight` the counts after the fact. For
             instance, the `1_1` label is more common, so should be downweighted
        SOM_mode - str or int
             The mode keyword that will be used to select the SOOM to distribute the 
             labels with. If mode is `train` or an int (which is a training segement) then
             there will be a single SOM_path in the src_heatmap[] attribute. Otherwise, if
             `mode` is `validate`, then there can be multiple SOM_paths as keys in the src_heatmap[]
             attribute
        weights - bool
             Use the global counts of the labels from the `book` object as a weigh when plotting. 
             TODO: Think about weighing the book object directly. I dont think this is the way to
             do it at this point
        '''
        if SOM_mode is not None:
            raise NotImplementedError('SOM_mode is not implemented in attribute_heatmap')

        binary = self._reterive_binary(mode)

        if binary is None:
            return

        if func is None:
            func = self._source_rgz

        heatmaps = binary.src_heatmap[binary.SOM_path]
        shape = heatmaps[0].shape
        items = binary.get_data(label=label, func=func)

        book = defaultdict(list)
        global_counts = defaultdict(int)
        book_counts = defaultdict(int)

        if realisations == 1:
            for heat, item in zip(heatmaps, items):
                loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)
                book[loc].append(item)
                
                # This is dangerous. Item HAS to be a list. Be aware that this could
                # be a problem later
                for i in item:
                    global_counts[i] += 1
        else:
            pixels = range(np.prod(shape))
            for heat, item in zip(heatmaps, items):                
                prob = 1. / heat**10.
                prob = prob / prob.sum()
                rand_pos = np.random.choice(pixels, size=realisations, p=prob.flatten())
                xd = rand_pos // heat.shape[0]
                yd = rand_pos % heat.shape[1]            
                locs = [(x,y) for x,y in zip(xd, yd)]
                for loc in locs:
                    book[loc].append(item)
                    # This is dangerous. Item HAS to be a list. Be aware that this could
                    # be a problem later
                    for i in item:
                        global_counts[i] += 1

        if plot:
            if realisations == 1:
                title = 'No realisations'
            else:
                title = f'{realisations} realisations Performed'
            if weights:
                title = f'{title} with weights'
                self.attribute_plot(book, shape, title=title, weights=global_counts, **kwargs)
            else:
                self.attribute_plot(book, shape, title=title, **kwargs)
                
        return book, global_counts

    def count_map(self, mode='train', SOM_mode= None, plot=False, save=None, color_map='bwr'):
        '''Produce a map of the number of images that best match each neuron. This
        will have the same shape as the SOM grid, and the counts in each cell should
        add to the number of images in the Binary file. For now, just use the heatmap
        attached to this Pink instance.

        mode - `train` or `validate` or int
             Specify which of the attached Binary instances we should map and process.
             If int return that training segment
        SOM_mode - see above
            Return the binary so that the SOM_path attribute can be obtained and 
            passed to grab the corresponding src_heatmap object that were created
            agaisnt that SOM. If None, use the SOM_path attribute of the binary
            returned by mode.
        plot - Bool
            Produce a figure of the counts per neuron
        save - None or Str
            If None, show the figure onscreen. Otherwise save it to the filename in 
            save
        color_map - str
            The name of the matplotlib.colormap that will be passed directly to matplotlib.pyplot.get_map()
        '''
        import matplotlib as mpl

        binary = self._reterive_binary(mode)

        if binary is None:
            return

        # Reterive the key that will be passed to src_heatmap dict
        if SOM_mode is None:
            SOM_path = binary.SOM_path
        else:
            SOM_binary = self._reterive_binary(SOM_mode)
            SOM_path = SOM_binary.SOM_path

        if save is not None:
            save = self._path_build(save)

        shape = binary.src_heatmap[SOM_path][0].shape
        book = np.zeros(shape)

        for heat in binary.src_heatmap[SOM_path]:
            loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)
            book[loc] += 1           

        # Diagnostic plot. Not meant to be final...
        if plot:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fig, ax = plt.subplots(1,1)

            im = ax.imshow(book, vmin=0, cmap=plt.get_cmap(color_map))
            ax.set(title='Counts per Neuron')
            ax.xaxis.set(ticklabels=[])
            ax.yaxis.set(ticklabels=[])
            
            divider = make_axes_locatable(ax)
            cax0 = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax0, label='Counts')

            fig.tight_layout()
            # fig.show()
            if save is None:
                plt.show()
            else:
                plt.savefig(save)

        return book

    def _source_rgz(self, s):
        '''Default function used to process source labels. 

        s - Source object instance
        '''
        # If there is only one object, its returned as dict. Test and list it if needed
        a = s.rgz_annotations()
        if a is None:
            return ''
        else:
            a = a['object']
            if not isinstance(a, list):
                a = [a]
            return [ i['name'] for i in a ]

    def validator(self, mode='validate', SOM_mode='train' , realisations=1, func=None, pack=True,
                  weights=False):
        '''The code to check the validate data agaisnt the training data. Use the distribution
        of labels from some neuron from the training data to predict the label of the source
        from the validate set

        mode - str or int
             Value passed to _reterive_binary(). This should almost certainly remain set as
             `validate`, although I am leaving it as an option if I have to change things
             down the line
        SOM_mode - str or int
             Value pased to _reterive_binary(). This is the binary that will be seen as ground
             truth and what the validate binary is compared agaisnt
        realisations - int
             The number of times to use the heatmap to select a position to put something
        func - None or Function
             The function used to process the labels. If None, use a default method
        pack - bool
             Back the answer dict object with the number of realisations, the preprocessor arguments
             and the Pink specific arguments
        weights - bool
             Use the counts of a label across the entire dataset as a weight when calculating
             the distribution of labels on a per neuron basis
        '''
        from collections import Counter

        if mode != 'validate':
            print(f'WARNING: validation binary is set to {mode}')

        # Get items
        train = self._reterive_binary(SOM_mode)
        valid = self.validate_binary
        valid = self._reterive_binary(mode)

        # Get the `book` object from the training data with label types
        if func is None:
            func = self._source_rgz

        book, label_counts = self.attribute_heatmap(func=func, mode=SOM_mode, realisations=realisations, plot=False)
        answer_book = defaultdict( lambda : {'correct':0,'wrong':0, 'accuracy':0} )

        # print(label_counts)

        # For speed, loop through the entire `book` object once now and flatten the items
        for key in book:
            v = [i for items in book[key] for i in items]
            book[key] = v

        empty = 0

        for src, heat in zip(valid.sources, valid.src_heatmap[train.SOM_path]):
            loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)

            # Capture an error if we can't increment counters. Most'y happens
            # when the book object doesnt have a loc
            try:
                # Build up the distribution of answers here
                # v = [i for items in book[loc] for i in items]
                c = book[loc]
                if isinstance(c, list):
                    c = Counter(c)
                    book[loc] = c

                if weights:
                    arr = np.array([c[i]/label_counts[i] for i in c.keys()])
                else:
                    arr = np.array([c[i] for i in c.keys()])
                # print(arr)
                items = [i for i in c.keys()]

                # Given the distribution of answers for that Neuron, make the guess.
                guess = items[np.argmax(arr)]
                
                # Using the distribution as a method of selecting generally does pretty poorly...
                # guess = np.random.choice(items, p=arr / np.sum(arr))

                
                predict = [i for i in func(src)]
                if len(predict) == 0:
                    empty += 1
                else:
                    if guess in [i for i in func(src)]:
                        answer_book[guess]['correct'] += 1
                        answer_book['total']['correct'] += 1
                    else:
                        answer_book[guess]['wrong'] += 1
                        answer_book['total']['wrong'] += 1
                    answer_book[guess]['accuracy'] = answer_book[guess]['correct'] / (answer_book[guess]['correct']+answer_book[guess]['wrong'] )
                    answer_book['total']['accuracy'] = answer_book['total']['correct'] / (answer_book['total']['correct']+answer_book['total']['wrong'] )

            except Exception as e:
                empty += 1    
                print(e)
        # Flatten out the dict of dicts into a single dict
        flattened_book = {f'{k}_{k2}': v2 for k, v in answer_book.items() for k2, v2 in v.items()}

        if pack:
            flattened_book['probability_summed'] = False
            flattened_book['weighted'] = weights
            flattened_book['empty_neuron_attempts'] = empty
            flattened_book['train_src_hash'] = train.binary_hash
            flattened_book['train_SOM_hash'] = train.SOM_hash
            flattened_book['valid_src_hash'] = valid.binary_hash
            flattened_book['train_segment'] = SOM_mode
            flattened_book['validate_path'] = valid.binary_path
            flattened_book['trained_SOM'] = train.binary_path
            flattened_book['realisations'] = realisations
            flattened_book['experiment'] = self.project_dir
            flattened_book.update(train.preprocessor_args)
            flattened_book.update(self.pink_args)

        return flattened_book

    def ed_to_prob(self, ed, stretch=10):
        prob = 1. / ed**stretch
        prob = prob / prob.sum()
        
        return prob
    
    def prob_validator(self, mode='validate', SOM_mode='train' , realisations=1, func=None, pack=True,
                             weights=False):
        '''The code to check the validate data agaisnt the training data. Use the distribution
        of labels from some neuron from the training data to predict the label of the source
        from the validate set. Compute the sum by using the probabilty matric of the subject source
        to poll all the neurons for their labels. 

        mode - str or int
            Value passed to _reterive_binary(). This should almost certainly remain set as
            `validate`, although I am leaving it as an option if I have to change things
            down the line
        SOM_mode - str or int
            Value pased to _reterive_binary(). This is the binary that will be seen as ground
            truth and what the validate binary is compared agaisnt
        realisations - int
            The number of times to use the heatmap to select a position to put something
        func - None or Function
            The function used to process the labels. If None, use a default method
        pack - bool
            Back the answer dict object with the number of realisations, the preprocessor arguments
            and the Pink specific arguments
        weights - bool
            Use the counts of a label across the entire dataset as a weight when calculating
            the distribution of labels on a per neuron basis
        '''
        from collections import Counter

        if mode != 'validate':
            print(f'WARNING: validation binary is set to {mode}')

        # Get items
        train = self._reterive_binary(SOM_mode)
        valid = self.validate_binary
        valid = self._reterive_binary(mode)

        # Get the `book` object from the training data with label types
        if func is None:
            func = self._source_rgz

        book, label_counts = self.attribute_heatmap(func=func, mode=SOM_mode, realisations=realisations, plot=False)
        answer_book = defaultdict( lambda : {'correct':0,'wrong':0, 'accuracy':0} )

        # Get unique labels first
        unique_labels = []
        for key in book:
            v = set([i for items in book[key] for i in items])
            for i in v:
                unique_labels.append(i)
        unique_labels = list(set(unique_labels))
        unique_labels.sort()
        
        print(unique_labels)

        # For speed, loop through the entire `book` object once and apply scheme
        for key in book:
            v = [i for items in book[key] for i in items]
            c = Counter(v)
            if weights:
                arr = np.array([c[i]/label_counts[i] for i in unique_labels])
            else:
                arr = np.array([c[i]/sum(c.values()) for i in unique_labels])
            book[key] = arr

        for src, heat in zip(valid.sources, valid.src_heatmap[train.SOM_path]):
            prob = self.ed_to_prob(heat, stretch=10.)
            
            values = defaultdict(float)
            empty = 0

            # Orignal methof using only the max label of each neuron
            #         for key in book:
            #             values[unique_labels[np.argmax(book[key])]] += prob[key] * max(book[key])
            
            # Poll all labels in each neuron
            for count, label in enumerate(unique_labels):
                for key in book:
                    values[label] += prob[key] * book[key][count]
            
            
            import operator
            guess = max(values.items(), key=operator.itemgetter(1))[0]
            prediction = [i for i in func(src)]
            if len(prediction) > 0:
                if guess in [i for i in func(src)]:
                    answer_book[guess]['correct'] += 1
                    answer_book['total']['correct'] += 1
                else:
                    answer_book[guess]['wrong'] += 1
                    answer_book['total']['wrong'] += 1
                answer_book[guess]['accuracy'] = answer_book[guess]['correct'] / (answer_book[guess]['correct']+answer_book[guess]['wrong'] )
                answer_book['total']['accuracy'] = answer_book['total']['correct'] / (answer_book['total']['correct']+answer_book['total']['wrong'] )
            else:
                empty += 1
                
        # Flatten out the dict of dicts into a single dict
        flattened_book = {f'{k}_{k2}': v2 for k, v in answer_book.items() for k2, v2 in v.items()}

        if pack:
            flattened_book['probability_summed'] = True
            flattened_book['weighted'] = weights
            flattened_book['empty_neuron_attempts'] = empty
            flattened_book['train_src_hash'] = train.binary_hash
            flattened_book['train_SOM_hash'] = train.SOM_hash
            flattened_book['valid_src_hash'] = valid.binary_hash
            flattened_book['train_segment'] = SOM_mode
            flattened_book['validate_path'] = valid.binary_path
            flattened_book['trained_SOM'] = train.binary_path
            flattened_book['realisations'] = realisations
            flattened_book['experiment'] = self.project_dir
            flattened_book.update(train.preprocessor_args)
            flattened_book.update(self.pink_args)

        return flattened_book

    def weight_test(self, mode='validate', SOM_mode='train' , realisations=1, func=None, pack=True,
                  weights=False):
        '''Code to test that the weighting option when predicting is doing the correct thing. 

        I was unsure whether they were being weighted by the total counts of the labls in the set correctly.
        This let me hack up a few more sanity checks without compromising the original code.

        mode - str or int
             Value passed to _reterive_binary(). This should almost certainly remain set as
             `validate`, although I am leaving it as an option if I have to change things
             down the line
        SOM_mode - str or int
             Value pased to _reterive_binary(). This is the binary that will be seen as ground
             truth and what the validate binary is compared agaisnt
        realisations - int
             The number of times to use the heatmap to select a position to put something
        func - None or Function
             The function used to process the labels. If None, use a default method
        pack - bool
             Back the answer dict object with the number of realisations, the preprocessor arguments
             and the Pink specific arguments
        weights - bool
             Use the counts of a label across the entire dataset as a weight when calculating
             the distribution of labels on a per neuron basis
        '''
        from collections import Counter

        if mode != 'validate':
            print(f'WARNING: validation binary is set to {mode}')

        # Get items
        train = self._reterive_binary(SOM_mode)
        valid = self.validate_binary
        valid = self._reterive_binary(mode)

        # Get the `book` object from the training data with label types
        if func is None:
            func = self._source_rgz

        book, label_counts = self.attribute_heatmap(func=func, mode=SOM_mode, realisations=realisations, plot=False)
        answer_book = defaultdict( lambda : {'correct':0,'wrong':0, 'accuracy':0} )

        print(label_counts)
        unique_labels = [i for i in label_counts.keys()]
        unique_labels.sort()

        empty = 0

        # For speed, loop through the entire `book` object once now and flatten the items
        for key in book:
            v = [i for items in book[key] for i in items]
            book[key] = v
            
        key = '1_1'
        for key in unique_labels:
            stats = []
            for loc in book:
                c = Counter(book[loc])
                stats.append(c[key] / label_counts[key])

            # print(stats)
            print('\t', key, sum(stats))

        for src, heat in zip(valid.sources, valid.src_heatmap[train.SOM_path]):
            loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)

            c = book[loc]
            if isinstance(c, list):
                c = Counter(c)
                book[loc] = c

            arr = np.array([c[i] / label_counts[i] for i in unique_labels])

            guess = unique_labels[np.argmax(arr)]
            # print(arr)
            # print(guess)

            # print([i for i in func(src)])
            predict = [i for i in func(src)]
            if len(predict) == 0:
                empty += 1
            else:
                if guess in [i for i in func(src)]:
                    answer_book[guess]['correct'] += 1
                    answer_book['total']['correct'] += 1
                else:
                    answer_book[guess]['wrong'] += 1
                    answer_book['total']['wrong'] += 1
                answer_book[guess]['accuracy'] = answer_book[guess]['correct'] / (answer_book[guess]['correct']+answer_book[guess]['wrong'] )
                answer_book['total']['accuracy'] = answer_book['total']['correct'] / (answer_book['total']['correct']+answer_book['total']['wrong'] )

        print('\t', answer_book['total']['accuracy'])

        # for src, heat in zip(valid.sources, valid.src_heatmap[train.SOM_path]):
        #     loc = np.unravel_index(np.argmin(heat, axis=None), heat.shape)

        #     # Capture an error if we can't increment counters. Most'y happens
        #     # when the book object doesnt have a loc
        #     try:
        #         # Build up the distribution of answers here
        #         # v = [i for items in book[loc] for i in items]
        #         v = book[loc]
        #         c = Counter(v)
        #         if weights:
        #             arr = np.array([c[i]/label_counts[i] for i in c.keys()])
        #         else:
        #             arr = np.array([c[i] for i in c.keys()])
        #         # print(arr)
        #         # print(c)
        #         # print(unique_labels)
        #         items = [i for i in c.keys()]

        #         # Given the distribution of answers for that Neuron, make the guess.
        #         guess = items[np.argmax(arr)]
                
        #         # Using the distribution as a method of selecting generally does pretty poorly...
        #         # guess = np.random.choice(items, p=arr / np.sum(arr))

        #         if guess in [i for i in func(src)]:
        #             answer_book[guess]['correct'] += 1
        #             answer_book['total']['correct'] += 1
        #         else:
        #             answer_book[guess]['wrong'] += 1
        #             answer_book['total']['wrong'] += 1
        #         answer_book[guess]['accuracy'] = answer_book[guess]['correct'] / (answer_book[guess]['correct']+answer_book[guess]['wrong'] )
        #         answer_book['total']['accuracy'] = answer_book['total']['correct'] / (answer_book['total']['correct']+answer_book['total']['wrong'] )
        #     except Exception as e:
        #         empty += 1    
        #         print(e)


if __name__ == '__main__':

    FRACTION = 0.8
    PROJECTS_DIR = 'Experiments'
    NUM_ITER = 10
    COLLECTION = [('Experiments/FIRST_WISE_Norm_Log_3_NoSigWise_Convex_Large/TEST8.pink', 'plots')]

    for i in sys.argv[1:]:
   
        if '-r' == i:
            rgz_dir = 'rgz_rcnn_data'

            cat = Catalog(rgz_dir=rgz_dir)

            # Commenting out to let me ctrl+C without killing things
            # cat.save_sources()

            print('\nValidating sources...')
            cat.collect_valid_sources()

            
            bins = cat.dump_binary('TEST_chan_3.binary', norm=True, sigma=[3., False], 
                                        log10=[True, False], convex=True,
                                        channels=['FIRST','WISE_W1'],
                                        project_dir=f'{PROJECTS_DIR}/FIRST_WISE_Norm_Log_3_NoSigWise_Convex_VeryLarge',
                                        fraction=FRACTION)


            train_bin, validate_bin = bins
            print(train_bin)
            print(validate_bin)

            pink = Pink(train_bin, pink_args={'som-width':20,
                                            'som-height':20,
                                            'num-iter':NUM_ITER},
                        validate_binary=validate_bin) 

            pink.train()
            
            pink.map(mode='train', apply=True)
            pink.map(mode='validate', apply=True)

            pink.save('TEST8.pink')
         
        elif '-t' == i:                    
            for pink_file, out_name in COLLECTION:

                    print(f'Loading {pink_file}\n')
                    pink = Pink.loader(pink_file)

                    pink.show_som(channel=0)
                    pink.show_som(channel=0, mode='split')
                    pink.show_som(channel=1)
                    pink.show_som(channel=1, mode='split')
                    pink.show_som(channel=1, mode='grid')

                    def reduce2(s):
                        # If there is only one object, its returned as dict. Test and list it if needed
                        a = s.rgz_annotations()
                        if a is None:
                            return ''
                        else:
                            a = a['object']
                            if not isinstance(a, list):
                                a = [a]
                            return [ i['name'] for i in a ]
                    pink.attribute_heatmap(func=reduce2, xtick_rotation=90, save=f'train_label_counts.pdf',
                                          color_map='Blues', mode='train')
                    pink.attribute_heatmap(func=reduce2, xtick_rotation=90, save=f'valid_label_counts.pdf',
                                          color_map='Blues', mode='validate')
                    pink.attribute_heatmap(func=reduce2, xtick_rotation=90, save=f'train_realisations_label_counts.pdf',
                                          color_map='Blues', mode='train', realisations=10000)
                    pink.attribute_heatmap(func=reduce2, xtick_rotation=90, save=f'valid_realisations_label_counts.pdf',
                                          color_map='Blues', mode='validate', realisations=10000)

                    plt.close('all')

                    # for i in tqdm(range(10, 12)):
                    #     pink.heatmap(plot=True, image_number=i, apply=False, save=f'{i}_heatmap.pdf')

        elif '-v' == i:
            import pandas 

            results = []
            for pink_file, out_name in COLLECTION:
                print(pink_file)
                pink = Pink.loader(pink_file)
                
                answer = pink.validator(realisations=100)
                results.append(answer)
                answer = pink.validator(realisations=1)
                results.append(answer)

            df = pandas.DataFrame(results)
            print(df)
            print('Saving results...')
            df.to_json('Validator_Results.json')
            df.to_csv('Validator_Results.csv')
            df.to_pickle('Validator_Results.pkl')

        elif '-c' == i:
            PROJECTS_DIR = 'Cross_Validation'
    
            rgz_dir = 'rgz_rcnn_data'

            cat = Catalog(rgz_dir=rgz_dir)

            # Commenting out to let me ctrl+C without killing things
            # cat.save_sources()

            print('\nValidating sources...')
            cat.collect_valid_sources()
            bins = cat.dump_binary('TEST_chan.binary', norm=True, sigma=[3., False], log10=[True,False], 
                            channels=['FIRST'],
                            project_dir=f'{PROJECTS_DIR}/FIRST_Norm_Log_3_Cross',
                            segments=4)

            train_bin, validate_bin = bins
            print(train_bin)
            print(validate_bin)

            pink = Pink(train_bin, 
                        pink_args={'som-width':4,
                                   'som-height':4,
                                   'num-iter':1},
                        validate_binary=validate_bin) 

            pink.train()
            results = []
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
        
            df = pd.DataFrame(results)
            print(df)
        
        elif '-g' == i:
            rgz_dir = 'rgz_rcnn_data'
            PROJECTS_DIR = 'Test_Experiments'
            FRACTION = 0.8

            LOAD_PINK  = f'{PROJECTS_DIR}/FIRST_WISE_W1_Norm_Log_3_NoSigWise/Test.pink'
            if not os.path.exists(LOAD_PINK):
                cat = Catalog(rgz_dir=rgz_dir)

                # Commenting out to let me ctrl+C without killing things
                # cat.save_sources()

                print('\nValidating sources...')
                cat.collect_valid_sources()

                bins = cat.dump_binary('TEST.binary', norm=True, sigma=[3., False], 
                                            log10=[True, False], convex=False,
                                            channels=['FIRST', 'WISE_W1'],
                                            project_dir=f'{PROJECTS_DIR}/FIRST_WISE_W1_Norm_Log_3_NoSigWise',
                                            fraction=FRACTION)    

                train_bin, validate_bin = bins
                print(train_bin)
                print(validate_bin)

                pink = Pink(train_bin, pink_args={'som-width':7,
                                                'som-height':7,
                                                'num-iter':5},
                            validate_binary=validate_bin) 

                pink.train()
                pink.map()
                pink.map(mode='validate', SOM_mode='train')
                pink.save('Test.pink')
            else:
                pink = Pink.loader(LOAD_PINK)

            pink.src_heatmap_plot(index=750)

            pink.show_som(channel=0, mode=0)
            # pink.show_som(channel=0, mode=0, plt_mode='split')
            # pink.show_som(channel=0, mode=0, plt_mode='grid')
            # pink.show_som(channel=1, mode=0)
            # pink.show_som(channel=1, mode=0, plt_mode='split')
            # pink.show_som(channel=1, mode=0, plt_mode='grid')

            # pink.attribute_heatmap(plot=True, xtick_rotation=90, color_map='gnuplot',
            #                        save='No_weights.png')
            # pink.attribute_heatmap(plot=True, xtick_rotation=90, color_map='gnuplot', 
            #                        weights=True, save='With_weights.png')
            # print(pink.validator())
            # print(pink.validator(weights=True))

        else:
            print('Options:')
            print(' -r : Run test code to scan in RGZ image data')
            print(' -t : Run test code for the Transform outputs and heatmap')
            print(' -v : Run test code for Validator')
            print(' -c : Run test code for cross-validation')
            print(' -g : Run test code for the global counts weighting for attribute processing/plotting')
            sys.exit(0)