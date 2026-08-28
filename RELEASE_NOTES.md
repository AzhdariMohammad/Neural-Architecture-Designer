# Neural Architecture Designer v1.0

Initial public release.

## Highlights

- Publication-ready neural-network diagram editor
- Built-in editable Samples
- Blue, prominent Samples selector
- Resizable three-pane workspace
- Visible draggable scrollbar for Style & Connections
- Fixed Export panel
- Explicit project save/open controls
- Straight and curved-horizontal connectivity
- Surface-aware gap and outside-junction controls
- Path and cascade highlighting
- PNG/JPG/SVG/PDF/EPS export
- Windows launcher and single-file EXE builder
## Windows executable export fix

- The EXE builder explicitly bundles Matplotlib backends required for PNG, JPG, SVG, PDF, and EPS export.
- Pillow PNG/JPEG plugins are explicitly included in frozen builds.
- `BUILD_EXE.bat` performs a backend preflight before packaging.

