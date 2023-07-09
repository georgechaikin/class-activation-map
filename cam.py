from process_utils import get_model, get_batch
import tensorflow as tf
import numpy as np
import matplotlib
import os

def get_heatmaps(batch, model, class_index = None, **kwargs):
    """Gets heatmaps for the batch, which correspond to class indices for each image (if the class_index specified) using Tensorflow methods

    Args:
        batch (numpy.ndarray): batch of images with the shape corresponding model input
        model (tensorflow.keras.models.Model): model with GAP layer before the last layer
        class_index (int, list, optional): list of class indices for each image to get CAM heatmaps.
            Default is None, so the class indices will be the most preferable classes for each image
    
    Returns:
        tensorflow.Tensor: batch of heatmaps
    """
    counterfactual = kwargs.get('counterfactual', False)
    batch_size = batch.shape[0]
    if type(model) is tf.keras.models.Sequential:
        cam_layers = model.layers[:-2] + [model.layers[-1]]
        cam_model = tf.keras.models.Sequential(cam_layers)
    else:
        cam_model = tf.keras.models.Model(inputs = model.input, outputs = model.layers[-1](model.layers[-3].output))
    # Preprocess the list of class indices
    preds = class_index
    if class_index is None:
        preds = model.predict(batch)
        preds = tf.argmax(preds, axis = 1)
    if type(class_index) is int:
        preds = tf.repeat(class_index, batch_size)
    # Create heatmaps corresponding the list of class indices (class_index variable)
    heatmaps = cam_model.predict(batch)
    heatmaps = tf.transpose(heatmaps, [0, 3, 1, 2])
    preds = tf.expand_dims(preds, 1)
    heatmaps = tf.gather_nd(heatmaps, preds, batch_dims=1)
    heatmaps = tf.math.maximum(heatmaps, 0)
    heatmaps /= tf.reduce_max(heatmaps, axis = [1,2])[...,tf.newaxis, tf.newaxis]
    # Define conterfactual CAM
    if counterfactual:
      heatmaps*=-1.
    return heatmaps

def get_superimposed_batch(batch, heatmaps, alpha=0.5):
    """Gets the batch of superimposed images
    
    Args:
        batch (tf.Tensor, np.array): batch of images
        heatmaps (tf.Tensor, np.array): batch of CAM heatmaps
        alpha (float): param for blending images and heatmaps
    
    Returns:
        tf.Tensor: batch of images with CAM heatmaps
    """
    # Rescale heatmap to a range 0-255
    heatmaps = np.uint8(255 * heatmaps)
    # Use jet colormap to colorize heatmap
    colormap = matplotlib.colormaps['viridis']
    # Use RGB values of the colormap
    colors = colormap(np.arange(256))[:, :3]
    heatmaps = colors[heatmaps]
    # Create an image with RGB colorized heatmap
    resize_shape = (batch.shape[1], batch.shape[2])
    heatmaps = tf.image.resize(heatmaps, resize_shape).numpy()
    # Superimpose the heatmap on original image
    superimposed_batch = (alpha * heatmaps + (1-alpha)*batch)
    superimposed_batch = tf.math.minimum(superimposed_batch, 1.)
    return superimposed_batch

def save_superimposed_batch(img_dir, img_type, save_dir, **kwargs):
    """Saves images with heatmaps

    Args:
        img_dir (str): the directory where images are stored
        img_type (str): images type
        save_dir: the directory where images with heatmaps should be stored
        **kwargs: additional params for internal functions
    """
    path_pattern = os.path.join(img_dir, f'*.{img_type}')
    batch, filepaths = get_batch(path_pattern, save_filepaths = True, **kwargs)
    model = kwargs['model'] if 'model' in kwargs else get_model(**kwargs)
    heatmaps = get_heatmaps(batch, model, **kwargs)
    superimposed_batch = get_superimposed_batch(batch, heatmaps, alpha=0.5)
    for filepath, cam_img in zip(filepaths, superimposed_batch):
        print(filepath)
        filename = os.path.basename(bytes.decode(filepath))
        save_path = os.path.join(save_dir, filename)
        tf.keras.utils.save_img(save_path, cam_img)