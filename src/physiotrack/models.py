from enum import Enum
import inspect
import os
import requests
from tqdm import tqdm
from pathlib import Path


class Models:
    """Central registry of every pretrained model physiotrack can download and run.

    ``Models`` is a namespace of nested classes and ``Enum`` groups arranged by a
    four-level hierarchy::

        Models.<Task>.<Backend>.<Enum>.<member>

    - **Task** — what the model does: ``Detection``, ``Pose``, ``Pose3D``,
      ``Depth``, ``Segmentation``.
    - **Backend** — the architecture/family, e.g. ``YOLO``, ``RTDETR``, ``Sapiens``,
      ``ViTPose``, ``MotionBERT``, ``DDH``, ``DepthAnythingV2``, ``SegFace``.
    - **Enum** — a group of interchangeable checkpoints (often by dataset or
      variant), e.g. ``Detection.YOLO.PERSON`` or ``Pose.ViTPose.WholeBody``.
    - **member** — a single checkpoint. Each member's ``.value`` is the **weight
      filename** on disk (or a relative path); ``.name`` is the short handle.

    A few groups nest one level deeper or sit directly under the task: ``Pose3D``
    backends (``MotionBERT``, ``DDH``, ``FaceOrientation``) are ``Enum``\\ s directly
    under ``Pose3D``; ``Pose3D.Canonicalizer`` holds ``Models`` (3DPCNet weights) and
    ``View`` (a plain string enum of canonical viewpoints); ``Depth.DepthAnythingV2``
    is an ``Enum`` directly under ``Depth``.

    Selecting a member does not download anything by itself. Pass a member to
    [`download_model`][physiotrack.Models.download_model] to fetch its weights, which
    are cached locally and auto-fetched (mostly from the project's HuggingFace repos)
    on first use. Members whose ``.value`` is an empty string (e.g.
    ``Canonicalizer.Models.GEOMETRIC``) are markers for weight-free / algorithmic
    paths and are not downloaded.

    The ``validate_*`` static methods check that a given member belongs to the
    expected task/subclass, raising a descriptive ``ValueError`` otherwise; the
    high-level predictors use them to guard their ``model=`` argument.

    Example:
        ```python
        import physiotrack as pt
        from physiotrack import Models

        # Pick a checkpoint from the registry ...
        model = Models.Pose.ViTPose.WholeBody.s_wholebody
        print(model.value)                        # 'vitpose-s-wholebody.pth'

        # ... download its weights (cached after the first call) ...
        weights_path = Models.download_model(model)

        # ... and hand it to a predictor.
        pose = pt.Pose(model=model)
        ```

    Note:
        The first ``download_model`` call for a checkpoint hits the network and may
        transfer a large file; subsequent calls reuse the cached copy. YOLO
        ``PERSON`` detection/segmentation variants and all ``Pose.YOLO`` checkpoints
        are fetched automatically by ultralytics instead, so ``download_model``
        returns ``None`` for them.

    See Also:
        [`Detection`][physiotrack.Detection], [`Pose`][physiotrack.Pose],
        [`Segmentation`][physiotrack.Segmentation], [`Depth`][physiotrack.Depth]:
        predictors that consume these registry members.
    """

    class Detection:
        class YOLO:
            class PERSON(Enum):
                m_person = "yolo11m.pt"
                l_person = "yolo11l.pt"
                n_person = "yolo11n.pt"

            class FACE(Enum):
                n_face = "yolov11n-face.pt"
                m_face = "yolov11m-face.pt"
                l_face = "yolov11l-face.pt"

            class VRFACE(Enum):
                l_vrface = "yolov12l-face.pt"

            class VR(Enum):
                m_vr = "yolo11m_vr.pt"
                l_vr = "yolo11l_vr.pt"

            class VRSTUDENT(Enum):
                m_vrstudent = "yolo11m_VRstudent.pt"
                l_vrstudent = "yolo11l_VRstudent.pt"

        class RTDETR:
            class PERSON(Enum):
                x_person = "rtdetr-x.pt"
                l_person = "rtdetr-l.pt"

            class VRSTUDENT(Enum):
                x_person = "yolo11x_RLDETR_VRstudent.pt"
                l_person = "yolo11l_RLDETR_VRstudent.pt"
                
    class Pose:
        class YOLO:
            class COCO(Enum):
                M11 = "yolo11m-pose.pt"
                L11 = "yolo11l-pose.pt"
            
        class Sapiens:
            class WholeBody(Enum):
                # COCO wholebody
                B1_TS_COCOHB = "sapiens_1b_coco_wholebody_best_coco_wholebody_AP_727_torchscript.pt2"
                B06_TS_COCOHB = "sapiens_0.6b_coco_wholebody_best_coco_wholebody_AP_695_torchscript.pt2"
                B03_TS_COCOHB = "sapiens_0.3b_coco_wholebody_best_coco_wholebody_AP_620_torchscript.pt2"
            
        class ViTPose:
            class WholeBody(Enum):
                s_wholebody = "vitpose-s-wholebody.pth"
                b_wholebody = "vitpose-b-wholebody.pth"
                l_wholebody = "vitpose-l-wholebody.pth"
                h_wholebody = "vitpose-h-wholebody.pth"

            class COCO(Enum):
                b_coco = "vitpose-b-coco.pth"
                h_coco = "vitpose-h-coco.pth"
                l_coco = "vitpose-l-coco.pth"
                s_coco = "vitpose-s-coco.pth"

    class Pose3D:
        class MotionBERT(Enum):
            mb_ft_h36m_global_lite = 'FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin'
            mb_ft_h36m = 'FT_MB_release_MB_ft_h36m/best_epoch.bin'
            # mb_ft_h36m_global = ''
            mb_train_h36m = 'MB_train_h36m/best_epoch.bin'

        class DDH(Enum):
            best = 'best_epoch_DDHPose.bin'

        class FaceOrientation(Enum):
            default = '6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth'
            VR = 'CMVS-FO-VR_epoch80.pth'

        class Canonicalizer:
            class Models(Enum):
                _3DPCNetS2 = 'best_model_3DPCNetS2.pth'
                _3DPCNetS3 = 'best_model_3DPCNetS3.pth'
                _3DPCNetTC48_byCam = 'best_model_3DPCNetTC48_byCam.pth'
                _3DPCNetTC48_byAction = 'best_model_3DPCNetTC48_byAction.pth'
                GEOMETRIC = ''

            class View(Enum):
                FRONT = "front"
                BACK = "back" 
                LEFT_SIDE = "left_side"
                RIGHT_SIDE = "right_side"

    class Depth:
        class DepthAnythingV2(Enum):
            vits = "depth_anything_v2_vits.pth"
            vitb = "depth_anything_v2_vitb.pth"
            vitl = "depth_anything_v2_vitl.pth"

        class ZipDepth(Enum):
            # Lightweight monocular depth. Both checkpoints share the same
            # variant='base'/global_mode='balanced' encoder+decoder weights and
            # differ only in the upsampling head.
            base = "zipdepth_base.pth"          # GPU/server head (convex unfold)
            npu = "zipdepth_base_npu.pth"       # NPU/CPU/mobile-friendly head

        # DepthAnythingV2 architecture config per encoder type. ``input_size`` is
        # the default square inference resolution for the encoder.
        MODEL_CONFIGS = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'input_size': 518},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768], 'input_size': 518},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024], 'input_size': 518},
        }

        # ZipDepth build config per variant. ``upsample_unfold`` selects the head
        # matching each checkpoint; ``input_size`` is the shorter-side resolution
        # (aspect ratio preserved) the model was trained at.
        ZIPDEPTH_CONFIGS = {
            'base': {'variant': 'base', 'global_mode': 'balanced', 'upsample_unfold': True, 'input_size': 384},
            'npu': {'variant': 'base', 'global_mode': 'balanced', 'upsample_unfold': False, 'input_size': 384},
        }

    class Segmentation:
        class Sapiens:
            class BodyPart(Enum):
                B1_TS_SEG = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
                B06_TS_SEG = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
                B03_TS_SEG = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"

        class YOLO:
            class VRHEAD(Enum):
                M11 = "yolo11m_VR_head.pt"
                M8_251029 =  'yolo8m_VR_head_251029.pt'

            class PERSON(Enum):
                m_person = "yolo11m-seg.pt"
                l_person = "yolo11l-seg.pt"

        class SegFace:
            # Face-part parsing (CelebAMask-HQ, 19 classes). Swin-Base @ 512.
            class Face(Enum):
                swinb_celeba_512 = "segface_swinb_celeba_512.pt"


    @staticmethod
    def _get_model_info(model_enum):
        """Extract model information from enum instance"""
        if not isinstance(model_enum, Enum):
            return None
            
        for category_name in ['Detection', 'Pose', 'Segmentation', 'Pose3D', 'Depth']:
            category = getattr(Models, category_name, None)
            if not category:
                continue
            for backend_name in dir(category):
                if backend_name.startswith('_'):
                    continue
                    
                backend = getattr(category, backend_name)
                if not inspect.isclass(backend):
                    continue
                if category_name == "Pose3D":
                    if issubclass(backend, Enum) and isinstance(model_enum, backend):
                        return {
                            'category': category_name,
                            'backend': backend_name,
                            'enum_class': backend_name,  # For Pose3D, backend and enum_class are the same
                            'model_name': model_enum.name,
                            'file_name': model_enum.value
                        }
                    # Check for Canonicalizer models
                    elif backend_name == 'Canonicalizer':
                        for enum_class_name in dir(backend):
                            if enum_class_name.startswith('_'):
                                continue
                            enum_class = getattr(backend, enum_class_name)
                            if (inspect.isclass(enum_class) and
                                issubclass(enum_class, Enum) and
                                isinstance(model_enum, enum_class)):
                                return {
                                    'category': category_name,
                                    'backend': 'Canonicalizer',
                                    'enum_class': enum_class_name,
                                    'model_name': model_enum.name,
                                    'file_name': model_enum.value
                                }
                elif category_name == "Depth":
                    # Depth has enums directly under the category (e.g., Depth.DepthAnythingV2)
                    if issubclass(backend, Enum) and isinstance(model_enum, backend):
                        return {
                            'category': category_name,
                            'backend': backend_name,  # e.g., 'DepthAnythingV2'
                            'enum_class': backend_name,
                            'model_name': model_enum.name,  # e.g., 'vitl'
                            'file_name': model_enum.value  # e.g., 'depth_anything_v2_vitl.pth'
                        }
                else:
                    for enum_class_name in dir(backend):
                        if enum_class_name.startswith('_'):
                            continue
                        enum_class = getattr(backend, enum_class_name)
                        if (inspect.isclass(enum_class) and 
                            issubclass(enum_class, Enum) and 
                            isinstance(model_enum, enum_class)):
                            return {
                                'category': category_name,
                                'backend': backend_name,
                                'enum_class': enum_class_name,
                                'model_name': model_enum.name,
                                'file_name': model_enum.value
                            }
        return None
    
    @staticmethod
    def _download_yolo_model(model_info, download_path):
        """Download ViTPose models from HuggingFace"""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_sapiens_model(model_info, download_path):
        """Download Sapiens models from HuggingFace"""
        file_name = model_info['file_name']

        parts = file_name.split('_')
        size = parts[1] if len(parts) > 1 else "1b"

        size_map = {"03b": "0.3b", "06b": "0.6b", "1b": "1b"}
        size = size_map.get(size, size)

        if model_info['category'] == 'Pose':
            task = "pose-coco"
            format_type = "torchscript"
            base_url = f"https://huggingface.co/noahcao/sapiens-{task}/resolve/main/sapiens_lite_host/{format_type}/pose/checkpoints/sapiens_{size}"
        elif model_info['category'] == 'Segmentation':
            # Sapiens segmentation models - all use facebook repos
            task = "seg"
            format_type = "torchscript"
            base_url = f"https://huggingface.co/facebook/sapiens-{task}-{size}-{format_type}/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_vitpose_model(model_info, download_path):
        """Download ViTPose models from HuggingFace"""
        file_name = model_info['file_name']
        dataset = model_info['enum_class'].lower()  # 'wholebody' or 'coco'
        base_url = f"https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/{dataset}"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)
    
    @staticmethod
    def _download_motionbert_model(model_info, download_path):
        """Download MotionBERT models from HuggingFace"""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        actual_filename = os.path.basename(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/walterzhu/MotionBERT/resolve/main/checkpoint/pose3d"
        download_url = f"{base_url}/{file_name}?download=true"
        
        return Models._download_file(download_url, actual_filename, full_download_path)
    
    def _download_ddh_model(model_info, download_path):
        """Download MotionBERT models from HuggingFace"""
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        actual_filename = os.path.basename(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        
        return Models._download_file(download_url, actual_filename, full_download_path)
    
    @staticmethod
    def _download_canonicalizer_model(model_info, download_path):
        """Download a Canonicalizer (3DPCNet) checkpoint from HuggingFace.

        All 3DPCNet checkpoints share one architecture and load from a single
        bundled inference config, so only the ``.pth`` weights are downloaded.
        """
        file_name = model_info['file_name']
        file_dir = os.path.dirname(file_name)
        full_download_path = os.path.join(download_path, file_dir)
        os.makedirs(full_download_path, exist_ok=True)
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"

        model_download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(model_download_url, file_name, full_download_path)

    @staticmethod
    def _download_depth_model(model_info, download_path):
        """Download DepthAnythingV2 models from tharindu326/physiotrack HuggingFace repo"""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_zipdepth_model(model_info, download_path):
        """Download a ZipDepth checkpoint from the tharindu326/physiotrack HuggingFace repo."""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_segface_model(model_info, download_path):
        """Download a SegFace face-parsing checkpoint from the physiotrack HuggingFace repo."""
        file_name = model_info['file_name']
        base_url = f"https://huggingface.co/tharindu326/physiotrack/resolve/main"
        download_url = f"{base_url}/{file_name}?download=true"
        return Models._download_file(download_url, file_name, download_path)

    @staticmethod
    def _download_file(url, file_name, download_path):
        """Generic file download with progress bar"""
        os.makedirs(download_path, exist_ok=True)
        file_path = os.path.join(download_path, file_name)

        if os.path.exists(file_path):
            print(f"File {file_name} already exists at {file_path}")
            return file_path

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }

            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8KB blocks

            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=file_name) as pbar:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:  # Filter out keep-alive chunks
                            f.write(chunk)
                            pbar.update(len(chunk))

            # print(f"Successfully downloaded {file_name} to {file_path}")
            return file_path

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {file_name}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

    @staticmethod
    def download_model(model_enum, download_path=f"{os.path.join(os.path.dirname(__file__))}/modules/model_data"):
        """Download a registry model's weights and return the local file path.

        Resolves which task/backend the member belongs to, then fetches the weight
        file from the appropriate HuggingFace repository, showing a progress bar. If
        the file already exists at the destination it is reused (no re-download).

        Args:
            model_enum (enum.Enum): A registry member, e.g.
                ``Models.Pose.ViTPose.WholeBody.s_wholebody`` or
                ``Models.Depth.DepthAnythingV2.vitl``.
            download_path (str, optional): Directory to download into. Defaults to
                the package-local ``modules/model_data`` directory. For members whose
                value contains a subdirectory (e.g. MotionBERT), that subdirectory is
                created under this path.

        Returns:
            str | None: Absolute path to the downloaded (or cached) weight file, or
                ``None`` for backends handled elsewhere — specifically any
                ``Pose.YOLO`` model and any ``PERSON`` YOLO/RTDETR variant, which
                ultralytics downloads on demand.

        Raises:
            ValueError: If ``model_enum`` is not an ``Enum`` instance, cannot be
                located in the registry, or belongs to an unknown backend.
            requests.exceptions.RequestException: If the HTTP download fails (a
                partial file is removed before re-raising).

        Example:
            ```python
            from physiotrack import Models
            path = Models.download_model(Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB)
            ```

        Note:
            The first call transfers the full checkpoint over the network; later
            calls for the same file return the cached path immediately.
        """
        if not isinstance(model_enum, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model_enum)}")
        
        model_info = Models._get_model_info(model_enum)
        if not model_info:
            raise ValueError(f"Could not determine model information for {model_enum}")
        
        # print(f"Downloading {model_info['category']} model: {model_info['backend']}.{model_info['enum_class']}.{model_info['model_name']}")
        if model_info['backend'] in ('YOLO', 'RTDETR'):
            # Pose-YOLO and any PERSON variant (detection or segmentation) auto-download
            # via ultralytics; everything else (FACE/VR/VRSTUDENT/VRHEAD/...) is hosted.
            if model_info['category'] == 'Pose' or model_info['enum_class'] == 'PERSON':
                return None
            return Models._download_yolo_model(model_info, download_path)
        elif model_info['backend'] == 'Sapiens':
            return Models._download_sapiens_model(model_info, download_path)
        elif model_info['backend'] == 'ViTPose':
            return Models._download_vitpose_model(model_info, download_path)
        elif model_info['backend'] == 'MotionBERT':
            return Models._download_motionbert_model(model_info, download_path)
        elif model_info['backend'] == 'DDH':
            return Models._download_ddh_model(model_info, download_path)
        elif model_info['backend'] == 'Canonicalizer':
            return Models._download_canonicalizer_model(model_info, download_path)
        elif model_info['backend'] == 'FaceOrientation':
            # FaceOrientation uses HuggingFace download like DDH
            return Models._download_ddh_model(model_info, download_path)
        elif model_info['backend'] == 'DepthAnythingV2':
            return Models._download_depth_model(model_info, download_path)
        elif model_info['backend'] == 'ZipDepth':
            return Models._download_zipdepth_model(model_info, download_path)
        elif model_info['backend'] == 'SegFace':
            return Models._download_segface_model(model_info, download_path)
        else:
            raise ValueError(f"Unknown backend: {model_info['backend']}")

    @staticmethod
    def validate_det_model(model, expected_subclass: str):
        """Verify a detection model belongs to the named detection subclass.

        Checks ``model`` against the enum named ``expected_subclass`` under either
        ``Models.Detection.YOLO`` or ``Models.Detection.RTDETR`` (matched
        case-insensitively). Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate detection registry member, e.g.
                ``Models.Detection.YOLO.PERSON.m_person``.
            expected_subclass (str): Name of the required enum group, e.g.
                ``"PERSON"``, ``"FACE"``, ``"VR"``, ``"VRSTUDENT"``,
                ``"VRFACE"`` (case-insensitive).

        Raises:
            ValueError: If ``model`` is not an ``Enum``, if no subclass named
                ``expected_subclass`` exists in YOLO or RTDETR, or if ``model`` is
                not a member of that subclass (the message lists valid members).

        Example:
            ```python
            from physiotrack import Models
            Models.validate_det_model(Models.Detection.YOLO.PERSON.m_person, "PERSON")
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum member for `model`, got {type(model).__name__}")
        target = expected_subclass.strip().upper()
        enum_classes = []
        for backend in (Models.Detection.YOLO, Models.Detection.RTDETR):
            if hasattr(backend, target):
                enum_classes.append(getattr(backend, target))
        if not enum_classes:
            raise ValueError(f"No detection subclass named '{expected_subclass}' in YOLO or RTDETR.")
        for enum_cls in enum_classes:
            if isinstance(model, enum_cls):
                return  # ✅ valid
        all_valid = []
        for enum_cls in enum_classes:
            names = ", ".join(e.name for e in enum_cls)
            all_valid.append(f"{enum_cls.__module__.split('.')[-1]}.{enum_cls.__name__}: [{names}]")
        valid_str = "\n  ".join(all_valid)
        raise ValueError(
            f"Model '{model.name}' is not valid for subclass '{expected_subclass}'.\n"
            f"Valid members are:\n  {valid_str}"
        )

    @staticmethod
    def validate_seg_model(model, expected_subclass: str = None):
        """Verify a segmentation model is valid, optionally for a specific subclass.

        With ``expected_subclass`` given, checks ``model`` against the enum of that
        name under ``Models.Segmentation.YOLO`` or ``Models.Segmentation.Sapiens``.
        Without it, accepts ``model`` if it is a member of any enum under any
        ``Models.Segmentation`` backend. Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate segmentation registry member, e.g.
                ``Models.Segmentation.YOLO.PERSON.m_person``.
            expected_subclass (str, optional): Name of the required enum group, e.g.
                ``"PERSON"``, ``"VRHEAD"``, ``"BodyPart"`` (matched
                case-insensitively). Defaults to ``None`` (any segmentation model
                accepted).

        Raises:
            ValueError: If ``model`` is not an ``Enum``, if ``expected_subclass`` is
                given but not found in YOLO or Sapiens, or if ``model`` is not a
                valid segmentation member.

        Example:
            ```python
            from physiotrack import Models
            Models.validate_seg_model(Models.Segmentation.Sapiens.BodyPart.B03_TS_SEG)
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum member for `model`, got {type(model).__name__}")

        # If expected_subclass is provided, validate against specific subclass
        if expected_subclass:
            target = expected_subclass.strip().upper()
            enum_classes = []
            for backend in (Models.Segmentation.YOLO, Models.Segmentation.Sapiens):
                # Check if the target exists in the backend
                for attr_name in dir(backend):
                    if attr_name.upper() == target:
                        enum_classes.append(getattr(backend, attr_name))

            if not enum_classes:
                raise ValueError(f"No segmentation subclass named '{expected_subclass}' in YOLO or Sapiens.")

            for enum_cls in enum_classes:
                if isinstance(model, enum_cls):
                    return  # ✅ valid

            all_valid = []
            for enum_cls in enum_classes:
                names = ", ".join(e.name for e in enum_cls)
                all_valid.append(f"{enum_cls.__module__.split('.')[-1]}.{enum_cls.__name__}: [{names}]")
            valid_str = "\n  ".join(all_valid)
            raise ValueError(
                f"Model '{model.name}' is not valid for subclass '{expected_subclass}'.\n"
                f"Valid members are:\n  {valid_str}"
            )
        else:
            # General validation - check if it's any valid segmentation model
            for backend_name in dir(Models.Segmentation):
                if backend_name.startswith('_'):
                    continue
                backend = getattr(Models.Segmentation, backend_name)
                if not inspect.isclass(backend):
                    continue

                for enum_class_name in dir(backend):
                    if enum_class_name.startswith('_'):
                        continue
                    enum_class = getattr(backend, enum_class_name)
                    if (inspect.isclass(enum_class) and
                        issubclass(enum_class, Enum) and
                        isinstance(model, enum_class)):
                        return  # ✅ valid

            raise ValueError(
                f"Invalid segmentation model: {repr(model)}.\n"
                f"Expected a valid enum member from Models.Segmentation.<Backend>.<EnumClass>"
            )

    @staticmethod
    def validate_pose_model(model):
        """Verify a model is a valid 2D pose registry member.

        Accepts ``model`` if it is a member of any enum under any
        ``Models.Pose`` backend (``YOLO``, ``Sapiens``, ``ViTPose``). Returns
        ``None`` on success.

        Args:
            model (enum.Enum): The candidate pose registry member, e.g.
                ``Models.Pose.ViTPose.WholeBody.s_wholebody``.

        Raises:
            ValueError: If ``model`` is not an ``Enum`` or is not a member of any
                ``Models.Pose.<Backend>.<EnumClass>``.

        Example:
            ```python
            from physiotrack import Models
            Models.validate_pose_model(Models.Pose.YOLO.COCO.M11)
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")
            
        for attr_name in dir(Models.Pose):
            if attr_name.startswith('_'):
                continue
                
            backend = getattr(Models.Pose, attr_name)
            if not inspect.isclass(backend):
                continue
                
            for sub_attr_name in dir(backend):
                if sub_attr_name.startswith('_'):
                    continue
                    
                sub = getattr(backend, sub_attr_name)
                if (inspect.isclass(sub) and 
                    issubclass(sub, Enum) and 
                    isinstance(model, sub)):
                    return  # ✅ Valid model found
                    
        raise ValueError(
            f"Invalid pose model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Pose.<Backend>.<EnumClass>"
        )

    @staticmethod
    def validate_pose3d_model(model, expected_subclass=None):
        """Verify a model is a valid 3D-pose registry member.

        Accepts ``model`` if it is a member of any enum under ``Models.Pose3D``
        (including the nested ``Canonicalizer`` groups). When ``expected_subclass``
        is given, also requires ``model``'s enum class name to match it exactly.
        Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate 3D-pose registry member, e.g.
                ``Models.Pose3D.MotionBERT.mb_ft_h36m``.
            expected_subclass (str, optional): Backend/enum-class name the model must
                come from, e.g. ``"MotionBERT"``, ``"DDH"``, ``"FaceOrientation"``.
                Defaults to ``None`` (any Pose3D member accepted).

        Raises:
            ValueError: If ``model`` is not an ``Enum``, if its class name does not
                match ``expected_subclass``, or if it is not a valid Pose3D member
                (the message lists all valid members).

        Example:
            ```python
            from physiotrack import Models
            Models.validate_pose3d_model(
                Models.Pose3D.MotionBERT.mb_ft_h36m, "MotionBERT"
            )
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")
        
        # If expected_subclass is provided, validate it matches
        if expected_subclass:
            model_class_name = model.__class__.__name__
            if model_class_name != expected_subclass:
                raise ValueError(
                    f"Expected model from Models.Pose3D.{expected_subclass}, "
                    f"but got {model_class_name}"
                )
            
        for attr_name in dir(Models.Pose3D):
            if attr_name.startswith('_'):
                continue
                
            backend = getattr(Models.Pose3D, attr_name)
            if not inspect.isclass(backend):
                continue
                
            # Check if this backend is an Enum class itself
            if issubclass(backend, Enum) and isinstance(model, backend):
                return  # ✅ Valid model found
                
            # Check sub-classes within the backend
            for sub_attr_name in dir(backend):
                if sub_attr_name.startswith('_'):
                    continue
                    
                sub = getattr(backend, sub_attr_name)
                if (inspect.isclass(sub) and 
                    issubclass(sub, Enum) and 
                    isinstance(model, sub)):
                    return  # ✅ Valid model found
                    
        # If we reach here, the model is not valid
        valid_models = []
        for attr_name in dir(Models.Pose3D):
            if attr_name.startswith('_'):
                continue
            backend = getattr(Models.Pose3D, attr_name)
            if inspect.isclass(backend) and issubclass(backend, Enum):
                for member in backend:
                    valid_models.append(f"Models.Pose3D.{attr_name}.{member.name}")
                    
        valid_str = "\n  ".join(valid_models)
        raise ValueError(
            f"Invalid pose3d model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Pose3D.<Backend>.<model_name>\n"
            f"Valid models are:\n  {valid_str}"
        )

    @staticmethod
    def validate_depth_model(model):
        """Verify a model is a valid depth registry member.

        Accepts ``model`` if it is a member of any enum under ``Models.Depth``
        (``DepthAnythingV2`` or ``ZipDepth``). Returns ``None`` on success.

        Args:
            model (enum.Enum): The candidate depth registry member, e.g.
                ``Models.Depth.DepthAnythingV2.vitl``.

        Raises:
            ValueError: If ``model`` is not an ``Enum`` or is not a valid
                ``Models.Depth.<Backend>.<model_name>`` (the message lists valid
                members).

        Example:
            ```python
            from physiotrack import Models
            Models.validate_depth_model(Models.Depth.DepthAnythingV2.vits)
            ```
        """
        if not isinstance(model, Enum):
            raise ValueError(f"Expected an Enum instance, got {type(model)}")

        # Check if model is from Depth category
        for attr_name in dir(Models.Depth):
            if attr_name.startswith('_'):
                continue

            backend = getattr(Models.Depth, attr_name)
            if not inspect.isclass(backend):
                continue

            if issubclass(backend, Enum) and isinstance(model, backend):
                return  # ✅ Valid model found

        # If we reach here, the model is not valid
        valid_models = []
        for attr_name in dir(Models.Depth):
            if attr_name.startswith('_'):
                continue
            backend = getattr(Models.Depth, attr_name)
            if inspect.isclass(backend) and issubclass(backend, Enum):
                for member in backend:
                    valid_models.append(f"Models.Depth.{attr_name}.{member.name}")

        valid_str = "\n  ".join(valid_models)
        raise ValueError(
            f"Invalid depth model: {repr(model)}.\n"
            f"Expected a valid enum member from Models.Depth.<Backend>.<model_name>\n"
            f"Valid models are:\n  {valid_str}"
        )

    @staticmethod
    def get_depth_config(model):
        """Return the build config for a depth model, dispatched by backend.

        Looks up the settings needed to construct the depth network for ``model``.
        The returned dict is backend-specific but always carries an ``input_size``
        key giving the model's default inference resolution:

        - ``DepthAnythingV2`` members return the encoder config with keys
          ``"encoder"``, ``"features"``, ``"out_channels"`` and ``"input_size"``.
        - ``ZipDepth`` members return the build config with keys ``"variant"``,
          ``"global_mode"``, ``"upsample_unfold"`` and ``"input_size"``.

        Args:
            model (Models.Depth.*): A depth registry member, e.g.
                ``Models.Depth.DepthAnythingV2.vitl`` or ``Models.Depth.ZipDepth.base``.

        Returns:
            dict: The config for the model (see above).

        Raises:
            ValueError: If ``model`` is not a recognized ``Models.Depth`` member
                or its variant is unknown.

        Example:
            ```python
            from physiotrack import Models
            cfg = Models.get_depth_config(Models.Depth.ZipDepth.base)
            ```
        """
        if isinstance(model, Models.Depth.DepthAnythingV2):
            encoder_name = model.name  # 'vits', 'vitb', or 'vitl'
            if encoder_name not in Models.Depth.MODEL_CONFIGS:
                raise ValueError(f"Unknown DepthAnythingV2 encoder: {encoder_name}")
            return Models.Depth.MODEL_CONFIGS[encoder_name]

        if isinstance(model, Models.Depth.ZipDepth):
            variant_name = model.name  # 'base' or 'npu'
            if variant_name not in Models.Depth.ZIPDEPTH_CONFIGS:
                raise ValueError(f"Unknown ZipDepth variant: {variant_name}")
            return Models.Depth.ZIPDEPTH_CONFIGS[variant_name]

        raise ValueError(
            f"Expected a Models.Depth.<Backend> enum member, got {type(model)}"
        )


if __name__ == "__main__":
    try:
        vitpose_path = Models.download_model(Models.Pose.ViTPose.WholeBody.s_wholebody)
        sapiens_path = Models.download_model(Models.Pose.Sapiens.WholeBody.B03_TS_COCOHB)
        yolo_path = Models.download_model(Models.Detection.YOLO.VRSTUDENT.m_vrstudent)
        
    except Exception as e:
        print(f"Error: {e}")