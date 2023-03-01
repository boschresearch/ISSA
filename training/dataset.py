import os
import numpy as np
import PIL.Image
import torch

try:
    import pyspng
except ImportError:
    pyspng = None



def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

def get_dataset_size(path):
    all_files = [os.path.relpath(os.path.join(root, fname), start=path) for root, _dirs, files in
                      os.walk(path, followlinks=True) for fname in files]
    all_files = set(all_files)
    for k in all_files:
        if '.ipynb_checkpoints' in k:
            all_files.remove(k)
    PIL.Image.init()
    image_fnames = sorted(fname for fname in all_files if file_ext(fname) in PIL.Image.EXTENSION)
    raw_shape = len(image_fnames)
    return raw_shape




#----------------------------------------------------------------------------

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path,
                 resolution     =None,
                 img_ratio      =None,      # width // height
                 cropping_mode  ='random',
                 max_size       = None,     # before xflip.
                 use_labels     = False,
                 xflip          = False,
                 random_seed    = 0,
                 inverse_order  = False,
                 use_w          = False,    #read latent or not
                 **super_kwargs,
    ):
        super().__init__()
        self.path = path
        if cropping_mode == 'random':
            self.cropping_func = random_crop
        elif cropping_mode == 'center':
            self.cropping_func = center_crop
        assert os.path.isdir(self.path)

        self.all_files = [os.path.relpath(os.path.join(root, fname), start=self.path) for root, _dirs, files in
                            os.walk(self.path, followlinks=True) for fname in files]
        for k in self.all_files:
            if '.ipynb_checkpoints' in k:
                self.all_files.remove(k)
        self.all_files = set(self.all_files)

        PIL.Image.init()
        self.image_fnames = sorted(fname for fname in self.all_files if file_ext(fname) in PIL.Image.EXTENSION)
        if len(self.image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self.path))[0]
        raw_shape = [len(self.image_fnames)] + list(self.load_test_image(0).shape)

        if resolution is not None:
            assert img_ratio is not None
            raw_shape[3] = resolution
            raw_shape[2] = int(resolution / img_ratio)
            print(f'Setting resolution and image ratio to {raw_shape[2]} x {raw_shape[3]}!')
        else:
            print('Use the default resolution and image ratio in the image foler!')

        self.datasetname = name
        self.raw_shape = list(raw_shape)
        self.use_labels = use_labels
        self.raw_labels = None
        self.label_shape = None
        self.use_w = use_w

        # max_size.
        self.raw_idx = np.arange(self.raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self.raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self.raw_idx)
            if not inverse_order:
                self.raw_idx = np.sort(self.raw_idx[:max_size])
            else:
                self.raw_idx = np.sort(self.raw_idx[-1*max_size:])

        self.xflip = np.zeros(self.raw_idx.size, dtype=np.uint8)
        if xflip:
            self.raw_idx = np.tile(self.raw_idx, 2)
            self.xflip = np.concatenate([self.xflip, np.ones_like(self.xflip)])



    def load_test_image(self, raw_idx):
        fname = self.image_fnames[raw_idx]
        with open(os.path.join(self.path, fname), 'rb') as f:
            if pyspng is not None and file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image


    def load_raw_image(self, raw_idx):
        fname = self.image_fnames[raw_idx]
        with open(os.path.join(self.path, fname), 'rb') as f:
            if pyspng is not None and file_ext(fname) == '.png':
                image = pyspng.load(f.read())
                image = PIL.Image.fromarray(image)
            else:
                image = PIL.Image.open(f)

            width, height = image.size  # Get dimensions

            if height != self.image_shape[-2] or width != self.image_shape[-1]:
                #print('Cropping now!')
                if self.aspect_ratio >=1:
                    new_width = width
                    new_height = int(new_width/self.aspect_ratio)
                else:
                    new_height = height
                    new_width = int(new_height * self.aspect_ratio)
                image = self.cropping_func(image, new_width, new_height)
                image = image.resize((self.image_shape[-1], self.image_shape[-2]), PIL.Image.ANTIALIAS)
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC

        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def load_w(self, raw_idx):
        fname = self.image_fnames[raw_idx][:-4] + '.npz'
        with np.load(os.path.join(self.path, fname)) as f:
            w = f['w']
        return w


    def __getitem__(self, idx):
        image = self.load_raw_image(self.raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self.xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        if self.use_w:
            return image.copy(), self.load_w(self.raw_idx[idx])
        else:
            return image.copy()


    def __len__(self):
        return self.raw_idx.size

    @property
    def name(self):
        return self.datasetname

    @property
    def image_shape(self):
        return list(self.raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[2]

    @property
    def aspect_ratio(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[2] / self.image_shape[1]



#----------------------------------------------------------------------------
def center_crop(img, new_width, new_height):
    width, height = img.size  # Get dimensions
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = left + new_width
    bottom = top + new_height
    img = img.crop((left, top, right, bottom))
    return img

def random_crop(img, new_width, new_height):
    width, height = img.size  # Get dimensions
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    if left <= 0:
        left = 0
    else:
        left = np.random.randint(0, left)
    if top <= 0:
        top = 0
    else:
        top = np.random.randint(0, top)
    right = left + new_width
    bottom = top + new_height
    img = img.crop((left, top, right, bottom))
    return img