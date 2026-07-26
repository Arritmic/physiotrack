import json
import math
import os

import numpy as np
from torch.utils.data import Dataset
from .utils_data import crop_scale

def halpe2h36m(x):
    '''
        Input: x (T x V x C)  
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y
    
def prepare_motion(keypoints, vid_size, scale_range):
    """Normalise a Halpe-ordered keypoint sequence into MotionBERT's input space.

    Args:
        keypoints (np.ndarray): ``(N, J, 3)`` Halpe/AlphaPose keypoints as
            ``(x, y, confidence)``.
        vid_size (tuple[int, int] | None): ``(width, height)`` used to centre and
            scale pixel coordinates. ``None`` skips that step.
        scale_range (list | tuple | None): Target scale range for ``crop_scale``.

    Returns:
        np.ndarray: ``(N, 17, 3)`` float32 motion in Human3.6M joint order.
    """
    kpts_all = halpe2h36m(np.asarray(keypoints, dtype=np.float64))
    motion = kpts_all
    if vid_size:
        w, h = vid_size
        scale = min(w, h) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] - np.array([w, h]) / 2.0
        kpts_all[:, :, :2] = kpts_all[:, :, :2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)


def read_input(json_path, vid_size, scale_range, focus):
    """Load an AlphaPose/Halpe JSON and normalise it into MotionBERT's input space."""
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        if focus is not None and item['idx'] != focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1, 3])
        kpts_all.append(kpts)
    return prepare_motion(np.array(kpts_all), vid_size, scale_range)


class WildDetDataset(Dataset):
    """Clip-batched 2D keypoint sequence.

    Accepts either an in-memory ``(N, J, 3)`` Halpe keypoint array or a path to an
    AlphaPose/Halpe JSON file. The array form is what the library uses; the JSON form
    exists for the upstream file-based entry point.
    """

    def __init__(self, source, clip_len=243, vid_size=None, scale_range=None, focus=None):
        self.clip_len = clip_len
        if isinstance(source, (str, os.PathLike)):
            self.json_path = str(source)
            self.vid_all = read_input(source, vid_size, scale_range, focus)
        else:
            self.json_path = None
            self.vid_all = prepare_motion(source, vid_size, scale_range)
        
    def __len__(self):
        'Denotes the total number of samples'
        return math.ceil(len(self.vid_all) / self.clip_len)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        st = index*self.clip_len
        end = min((index+1)*self.clip_len, len(self.vid_all))
        return self.vid_all[st:end]