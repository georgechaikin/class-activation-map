import tensorflow as tf
import numpy as np

from functools import partial
import os

def get_model(model_path):
    """Returns model using predefined information

    Args:
        model_path (str): path to the Tensorflow model
    """
    model = tf.keras.models.load_model(model_path, compile  = False)
    return model

def load_img(img_path, **kwargs):
    """Loads images using specified arguments
    
    Args:
        img_path (str): path to the image
        **kwargs: additional params
    """
    # Possible kwargs preprocessing
    # channels and img_size are currently used as kwargs
    channels = kwargs.get('channels', 3)
    img_size = kwargs.get('img_size', (224, 224))
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, channels = channels, expand_animations = False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, img_size)
    return img

def get_batch(path_pattern, **kwargs):
    """Loads images into the batch
    
    Arguments:
        path_pattern (str): glob pattern for images
        **kwargs: special arguments for list_files func
    
    Returns:
        numpy.ndarray: numpy array with the shape (b,h,w,c)
    """
    shuffle = kwargs.get('shuffle', None)
    seed = kwargs.get('seed', None)
    save_filepaths = kwargs.get('save_filepaths', False)
    # Define images paths
    images_ds = tf.data.Dataset.list_files(
        path_pattern,
        shuffle = shuffle,
        seed = seed)
    images_ds.cache()
    if save_filepaths:
        filenames_path = os.path.join('tmp', 'filepaths')
        os.makedirs(filenames_path, exist_ok = True)
        images_ds.save(filenames_path)
    # Load images
    partial_load_img = partial(load_img, **kwargs)
    images_ds = images_ds.map(partial_load_img)
    return np.array(list(images_ds.as_numpy_iterator()))