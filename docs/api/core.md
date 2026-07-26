# Overlay Views

The composited side panels the [`Video`][physiotrack.Video] orchestrator lays around the
annotated frame, plus the anti-aliased drawing primitives they are built from. Exported so a
custom capture loop can compose the same panels rather than reimplementing them.

## Panels

::: physiotrack.core.RadarView

::: physiotrack.core.DepthView

::: physiotrack.core.EgoVideoView

::: physiotrack.core.ROMSkeletonView

## Drawing primitives

::: physiotrack.core.OverlayCanvas

::: physiotrack.core.draw_label

::: physiotrack.core.alpha_composite
