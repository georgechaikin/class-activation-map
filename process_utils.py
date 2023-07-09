import tensorflow as tf
import numpy as np

from functools import partial

def get_model(model_path):
    """Returns the model using predefined information

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
        path_pattern (str, list): glob pattern for images or list of paths
        **kwargs: special arguments for list_files func and for datasets preprocessing
    
    Returns:
        numpy.ndarray: numpy array with the shape (b,h,w,c)
        tuple: tuple with the batch and file paths if save_filepaths is True
    """
    shuffle = False
    seed = None
    save_filepaths = kwargs.get('save_filepaths', False)
    get_dataset = kwargs.get('get_dataset', False)
    # Define images paths
    if type(path_pattern) == list:
        files_ds = tf.data.Dataset.from_generator(lambda: path_pattern, output_types=tf.string)
    else:
        files_ds = tf.data.Dataset.list_files(
            path_pattern,
            shuffle = shuffle,
            seed = seed)
    files_ds.cache()
    # Load images
    partial_load_img = partial(load_img, **kwargs)
    images_ds = files_ds.map(partial_load_img)
    if not get_dataset:
        images_ds = np.array(list(images_ds.as_numpy_iterator()))
        files_ds = list(files_ds.as_numpy_iterator())
    return (images_ds, files_ds) if save_filepaths else images_ds