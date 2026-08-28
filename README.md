# Neural Architecture Designer

**Version 1.0**

Neural Architecture Designer is a desktop application for creating editable, publication-ready neural-network architecture diagrams with Python, Tkinter, Matplotlib, and Pillow.

## Features

- Editable network layers and node counts
- Circle, square, ellipse, rounded, diamond, triangle, hexagon, pentagon, and activation-block styles
- Dense, sampled, adjacent-only, or disabled layer-pair connectivity
- Straight and curved-horizontal connections
- Lines and directional arrows
- Surface-aware connection gaps
- Outside-junction positioning with positive or negative offset
- Single, palette-cycle, and random-palette connection colors
- Randomized connection widths
- Per-layer-pair connection overrides
- Path and cascade highlighting
- Text, image, arrow, line, divider, and shape annotations
- Input/output item boxes
- Built-in editable Samples
- Resizable three-column workspace
- Scrollable Style & Connections panel with a fixed Export panel
- PNG, JPG, SVG, PDF, and EPS export
- Editable project save/load

## Windows quick start

1. Install **Python 3.10+**.
2. Download or clone the repository.
3. Double-click **`RUN_APP.bat`**.
4. On the first launch, the script creates a local `.venv` and installs the required packages.
5. Choose an architecture from the blue **Samples** selector.
6. Edit the project and export the finished figure from the Export panel.

## Run manually

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python neural_architecture_designer.py
```

## Build the Windows executable

Double-click **`BUILD_EXE.bat`**. The builder checks and bundles the Matplotlib backends required by PNG, JPG, SVG, PDF, and EPS export. The resulting executable is placed in the project root:

```text
NeuralArchitectureDesigner.exe
```

## Project controls

- **New Project**
- **Open Project**
- **Save Project**
- **Save Project As**

Figure export is available separately in the Export panel.

## Documentation

- `QUICK_START.txt` — minimal setup instructions
- `USER_GUIDE.md` — practical usage guide
- `RELEASE_NOTES.md` — v1.0 release summary
- `CHANGELOG.md` — version history

## Requirements

- Python 3.10+
- Matplotlib 3.8+
- Pillow 10+
- Tkinter

## Citation

For academic use, see `CITATION.cff`.

## License

MIT License. See `LICENSE`.
