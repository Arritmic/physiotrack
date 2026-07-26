import os
import numpy as np
from tqdm import tqdm
import imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils.utils import get_config
from .utils.utils_data import flip_data
from .utils.dataloader import WildDetDataset
from functools import partial
from .model.DSTformer import DSTformer

from ..._checkpoints import strip_data_parallel_prefix
from ..._logging import get_logger

logger = get_logger(__name__)


class MotionBERTInference:
    def __init__(self, 
                 config_path="configs/MB_ft_h36m_global_lite.yaml",
                 checkpoint_path='checkpoint/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin',
                 clip_len=243,
                 testloader_params=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize PoseInference class"""
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.clip_len = clip_len
        self.device = device
        self.args = get_config(config_path)
        
        self.model_pos = self._load_model()
        
        self.testloader_params = testloader_params or {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': True,
            'prefetch_factor': None,
            'persistent_workers': False,
            'drop_last': False
        }
        self.fps_in = 30
    
    def _load_model(self):
        """Load and initialize the model"""
        model_backbone = DSTformer(
            dim_in=3, 
            dim_out=3, 
            dim_feat=self.args.dim_feat, 
            dim_rep=self.args.dim_rep,
            depth=self.args.depth, 
            num_heads=self.args.num_heads, 
            mlp_ratio=self.args.mlp_ratio, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            maxlen=self.args.maxlen, 
            num_joints=self.args.num_joints
        )
        
        if self.device == 'cuda' and torch.cuda.is_available():
            model_backbone = nn.DataParallel(model_backbone)
            model_backbone = model_backbone.cuda()
        else:
            model_backbone = model_backbone.to(self.device)
        
        logger.info("Loading MotionBERT checkpoint: %s", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=lambda storage, loc: storage)
        # The released checkpoints were saved from a DataParallel-wrapped model, so their
        # keys carry a "module." prefix that only matches when we happen to wrap too.
        # Normalising it away makes the load work on CPU and GPU alike.
        state = strip_data_parallel_prefix(checkpoint['model_pos'])
        if isinstance(model_backbone, nn.DataParallel):
            model_backbone.module.load_state_dict(state, strict=True)
        else:
            model_backbone.load_state_dict(state, strict=True)
        model_backbone.eval()
        
        return model_backbone
    
    def update_testloader_params(self, **kwargs):
        """Update dataloader parameters"""
        self.testloader_params.update(kwargs)
    
    def _prepare_dataset(self, source, vid_path=None, pixel=False, focus=None,
                         scale_range=None, fps=None, frame_size=None):
        """Prepare the clip dataset for inference.

        Args:
            source (np.ndarray | str | os.PathLike): Halpe keypoints ``(N, J, 3)`` or a
                path to an AlphaPose/Halpe JSON.
            vid_path (str, optional): Source video, read only to recover ``fps`` and
                frame size when they are not supplied directly.
            pixel (bool): Keep pixel coordinates (requires a known frame size).
            focus (int, optional): Subject index filter, JSON input only.
            scale_range (list, optional): Scale range when not in pixel mode.
            fps (float, optional): Frame rate, when already known.
            frame_size (tuple[int, int], optional): ``(width, height)``, when already
                known.

        Returns:
            tuple: ``(dataset, fps, frame_size)``.

        Raises:
            ValueError: If ``pixel`` is requested without a frame size and without a
                video to read one from.
        """
        # Only touch the video when something is actually missing: lifting a keypoint
        # array should not require the original file to be present.
        if (fps is None or frame_size is None) and vid_path is not None:
            vid = imageio.get_reader(vid_path, 'ffmpeg')
            meta = vid.get_meta_data()
            fps = fps if fps is not None else meta['fps']
            frame_size = frame_size if frame_size is not None else meta['size']

        if pixel and frame_size is None:
            raise ValueError(
                "pixel=True needs the source frame size to convert normalised "
                "coordinates back to pixels. Pass frame_size=(width, height), or a "
                "vid_path to read it from."
            )

        self.fps_in = fps

        if pixel:
            # Keep relative scale with pixel coordinates
            wild_dataset = WildDetDataset(
                source,
                clip_len=self.clip_len,
                vid_size=frame_size,
                scale_range=None,
                focus=focus
            )
        else:
            # Scale to [-1,1] or custom scale range
            wild_dataset = WildDetDataset(
                source,
                clip_len=self.clip_len,
                scale_range=scale_range or [1, 1],
                focus=focus
            )

        return wild_dataset, fps, frame_size
    
    def _process_batch(self, batch_input, no_conf=None, flip=None, rootrel=None, gt_2d=None):
        """Process a single batch through the model"""
        N, T = batch_input.shape[:2]
        
        no_conf = no_conf if no_conf is not None else self.args.no_conf
        flip = flip if flip is not None else self.args.flip
        rootrel = rootrel if rootrel is not None else self.args.rootrel
        gt_2d = gt_2d if gt_2d is not None else self.args.gt_2d
        
        if self.device == 'cuda' and torch.cuda.is_available():
            batch_input = batch_input.cuda()
        else:
            batch_input = batch_input.to(self.device)
        
        if no_conf:
            batch_input = batch_input[:, :, :, :2]
        
        if flip:
            batch_input_flip = flip_data(batch_input)
            predicted_3d_pos_1 = self.model_pos(batch_input)
            predicted_3d_pos_flip = self.model_pos(batch_input_flip)
            predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)
            predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
        else:
            predicted_3d_pos = self.model_pos(batch_input)
        
        if rootrel:
            predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
        else:
            predicted_3d_pos[:, 0, 0, 2] = 0
        
        if gt_2d:
            predicted_3d_pos[..., :2] = batch_input[..., :2]
        
        return predicted_3d_pos
    
    def infer(self,
              source,
              vid_path=None,
              pixel=False,
              focus=None,
              scale_range=None,
              no_conf=None,
              flip=None,
              rootrel=None,
              gt_2d=None,
              custom_testloader_params=None,
              fps=None,
              frame_size=None):
        """Lift a 2D keypoint sequence to 3D.

        Args:
            source (np.ndarray | str | os.PathLike): Halpe keypoints ``(N, J, 3)`` or a
                path to an AlphaPose/Halpe JSON.
            vid_path (str, optional): Source video, read only to recover fps/frame size.
            pixel (bool): Return pixel-space rather than normalised coordinates.
            focus (int, optional): Subject index filter, JSON input only.
            scale_range (list, optional): Scale range when not in pixel mode.
            no_conf, flip, rootrel, gt_2d (bool, optional): Backend flags; ``None``
                keeps the checkpoint's configured default.
            custom_testloader_params (dict, optional): Override DataLoader params.
            fps (float, optional): Frame rate, when already known.
            frame_size (tuple[int, int], optional): ``(width, height)``, when known.

        Returns:
            np.ndarray: ``(N, 17, 3)`` poses in Human3.6M joint order.
        """
        testloader_params = custom_testloader_params or self.testloader_params

        wild_dataset, fps_in, vid_size = self._prepare_dataset(
            source, vid_path, pixel, focus, scale_range, fps, frame_size
        )
        test_loader = DataLoader(wild_dataset, **testloader_params)
        
        results_all = []
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                predicted_3d_pos = self._process_batch(
                    batch_input, no_conf, flip, rootrel, gt_2d
                )
                results_all.append(predicted_3d_pos.cpu().numpy())
        
        results_all = np.hstack(results_all)
        results_all = np.concatenate(results_all)

        if pixel:
            results_all = results_all * (min(vid_size) / 2.0)
            results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0
        
        return results_all
    
    def process_results(self, results, vid_size, pixel=False):
        """Post-process results separately"""
        if pixel:
            results = results * (min(vid_size) / 2.0)
            results[:, :, :2] = results[:, :, :2] + np.array(vid_size) / 2.0
        return results
