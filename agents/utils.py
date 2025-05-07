import numpy as np
import random
import torch

from collections import namedtuple, deque
from typing import *



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'next_state'))



def channel_l2f(rgb_arr :np.ndarray) -> np.ndarray:
    """
    Move the channel axis of an image or batch of images from the last position to the front.

    This function reorders the axes of an image or batch of images so that the channel dimension
    (assumed to be the last axis) is moved to the first axis (for 3D input) or second axis
    (for 4D input). This is commonly required when converting image data from 'channels last'
    format (H, W, C) or (N, H, W, C) to 'channels first' format (C, H, W) or (N, C, H, W).

    Parameters
    ----------
    rgb_arr : np.ndarray
        A Numpy array representing an image or batch of images.
        - If 3D: shape is (H, W, C)
        - If 4D: shape is (N, H, W, C)
        The last axis is assumed to be the channel axis.

    Returns
    -------
    : np.ndarray
        The input array with the channel axis moved to:
        - Position 0 if input is 3D: (C, H, W)
        - Position 1 if input is 4D: (N, C, H, W)

    Raises
    ------
    NotImplementedError
        - If the input array is not 3D or 4D.
    """
    if rgb_arr.ndim == 3:
        dest = 0
    elif rgb_arr.ndim == 4:
        dest = 1
    else:
        raise NotImplementedError(f"not implemented for {rgb_arr.ndim}-dimensional arrays")
    return np.moveaxis(rgb_arr, -1, dest)



def normalize_img(rgb_arr :np.ndarray) -> np.ndarray:
    """
    Normalize an RGB image array to the range [0.0, 1.0].

    This function converts an input RGB image with pixel values in the range [0, 255]
    to a float32 array with values scaled to the range [0.0, 1.0].

    Parameters
    ----------
    rgb_arr : np.ndarray
        A Numpy array representing the RGB image.
        Expected to have dtype uint8 and values in the range [0, 255].

    Returns
    -------
    : np.ndarray
        The normalized RGB image as a float32 Numpy array with values in the range [0.0, 1.0].
    """
    return rgb_arr.astype(np.float32) / 255



def image_to_torch(rgb_array :np.ndarray) -> torch.Tensor:
    """
    Convert an RGB image or batch of images from NumPy array to a normalized PyTorch tensor.

    This function performs two transformations on the input image(s):
    1. Reorders the channel axis from the last position to the first (channels-first format).
       - For a single image: from (H, W, C) to (C, H, W)
       - For a batch of images: from (N, H, W, C) to (N, C, H, W)
    2. Normalizes the pixel values from [0, 255] to [0.0, 1.0] and converts to a float32 PyTorch tensor.

    Parameters
    ----------
    rgb_array : np.ndarray
        A NumPy array representing an RGB image or a batch of images.
        - If 3D: shape is (H, W, C)
        - If 4D: shape is (N, H, W, C)
        Expected to have dtype uint8 and values in the range [0, 255].

    Returns
    -------
    : torch.Tensor
        A float32 PyTorch tensor with:
        - Shape (C, H, W) if input is 3D
        - Shape (N, C, H, W) if input is 4D
        Values are in the range [0.0, 1.0].

    Raises
    ------
    NotImplementedError
        - If the input array is not 3D or 4D.
    """
    return torch.tensor(normalize_img(channel_l2f(rgb_array)))



def extract_cell(rgb_arr :torch.Tensor, x :int, y :int, cell_size_x :int, cell_size_y :int|None=None) -> torch.Tensor:
    """
    Extract a specific cell from a given RGB image array.

    Parameters
    ----------
    rgb_arr : torch.Tensor of shape (C, Y, X)
        The input RGB image represented as a 3D array with shape (channels, height, width),
        where Y is the height, X is the width, and C is the number of color channels (typically 3 for RGB).
    x : int
        The horizontal index (column) of the top-left corner of the cell to extract.
    y : int
        The vertical index (row) of the top-left corner of the cell to extract.
    cell_size_x : int
        The width of the cell to extract, in pixels.
    cell_size_y : int | None, optional
        The height of the cell to extract, in pixels. If ``None``, the height will be the same as `cell_size_x`.
        Default is ``None``.
        
    Returns
    -------
    : torch.Tensor
        A 3D array representing the extracted cell from the original image. The shape of the array
        will be (C, cell_size_y, cell_size_x), where C is the number of color channels in the original image.
        
    Notes
    -----
    - If ``cell_size_y`` is not provided (None), it will default to the value of ``cell_size_x``, creating a square cell.
    - The coordinates (x=0, y=0) correspond to the top-left corner of the cell within the original image array.
    """
    if cell_size_y is None:
        cell_size_y = cell_size_x
    cell_img = rgb_arr[:, y*cell_size_y:(y+1)*cell_size_y, x*cell_size_x:(x+1)*cell_size_x]
    return cell_img



class Memory():
    """
    This class implements a memory for storing and sampling ``Transition``. 
    """

    def __init__(self, capacity :int=1024):
        """
        Parameters
        ----------
        capacity : int, optional
            Capacity of the memory in number of transitions, by default ``1024``.
        """
        self.memory = deque([], maxlen=capacity)
        return


    def push(self, transition :Transition) -> None:
        """
        Saves a new transition.
        If the memory is full, the oldest transaction is overwritten.

        Parameters
        ----------
        transition : Transition
            New transition to store.
        """
        self.memory.append(transition)
        return None


    def sample(self, batch_size :int=16) -> Transition:
        """
        Draw randomly ``batch_size`` samples from the memory.

        Parameters
        ----------
        batch_size : int
            Number of samples to draw, by default ``16``.

        Returns
        -------
        : Transition
            Transaction samples.
        """
        return random.sample(self.memory, batch_size)


    def __len__(self) -> int:
        return len(self.memory)