import logging

from scipy.signal import resample


def preprocess(rgb):
    """
    rgb: height x width x channel(RGB)
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    x = resample(resample(gray, 84, axis=1), 110, axis=0)[26:, :]
    return x / 255.0


def log_print(*args, **kwargs):
    if kwargs.get('print', False):
        print(*args, **kwargs)
    logging.info(*args, **kwargs)
