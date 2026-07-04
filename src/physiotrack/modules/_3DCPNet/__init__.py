"""3DPCNet — 3D pose canonicalization network backend.

Provides the learned viewpoint-canonicalization transforms used by
:class:`physiotrack.PoseCanonicalizer`. The two public entry points re-exported at
the top level of :mod:`physiotrack` are
:func:`~physiotrack.apply_3dpcnet_transform` and
:func:`~physiotrack.reverse_3dpcnet_transform`.
"""
