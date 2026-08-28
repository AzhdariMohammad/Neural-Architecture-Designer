# Neural Architecture Designer v1.0 — User Guide

## 1. Starting the application

On Windows, run `RUN_APP.bat`. The launcher creates a local virtual environment and installs Matplotlib and Pillow if required.

The main workspace has three panes. Both vertical separators can be dragged, so you can allocate more width to Architecture, Settings, or Live Preview. At startup the approximate ratio is 23% / 14% / 63%.

## 2. Samples

The blue **Samples** drop-down at the upper-left contains the built-in editable templates. Selecting a Sample loads its architecture and styling into the current workspace. You can then modify every relevant layer and style setting.

## 3. Project controls

The upper-right controls are deliberately named:

- **New Project** — start a new editable project.
- **Open Project** — load a saved project.
- **Save Project** — save the current editable project.
- **Save Project As** — save to another project file.

These are project operations. Image export is handled only from the fixed Export panel.

## 4. Left pane — architecture and annotations

### Architecture

Double-click a layer to edit it. You can add, duplicate, delete, or reorder layers. Layer settings include node count, visible node count, shape, fill/border, text inside nodes, activation labels, and semi-automatic X/Y offsets.

### Groups / headers

Groups add labels over one layer or over a range of layers. Header styles include text, pill, and box styles where available.

### Canvas objects / annotations

Use free-form text, images, shapes, arrows, dividers, and related annotations to create publication-ready explanatory diagrams.

## 5. Middle pane — style and connections

The upper middle area has a visible vertical scrollbar with a draggable thumb. The mouse wheel also scrolls this area. The Export panel at the bottom is fixed and remains visible.

### General layout

- **Node radius** controls node size.
- **Layer spacing** controls horizontal layer separation.
- **Vertical spacing** controls vertical separation between visible items.
- **Layout mode** can use automatic or semi-automatic offsets.

### Connection selection

- **Dense** — connect all relevant source/target nodes.
- **Sampled** — draw a representative subset when a full dense graph would be excessive.
- **Adjacent-only** — connect corresponding/nearby positions.
- **None** — disable the pair's connections.

### Rendering

- **Lines** — plain connections.
- **Arrows** — directed connections.

### Geometry

- **Straight** — direct segments.
- **Curved-horizontal** — smooth curves with horizontal entry/exit behavior.

### End mode

- **Trim at cells** — connection endpoints are calculated from the visible cell surface. `Line gap` or `Arrow gap` then adds extra distance beyond the surface.
- **Outside junctions** — endpoints are displaced beyond the cell boundary. `Junction offset` moves the junction farther outward; negative values are allowed when a controlled inward shift is desired.

### Connection colors

- **Single** — use one connection color.
- **Palette cycle** — cycle through a palette.
- **Random palette** — select colors from the specified palette.

### Random line widths

Enable random emphasis and adjust probability, minimum factor, maximum factor, and seed. This can visually encode heterogeneous or illustrative connection strengths.

### Layer-pair styles

Use `Layer-pair styles…` when different pairs require different rendering, colors, widths, gaps, or random-emphasis settings.

## 6. Terminal arrows

Incoming/outgoing terminal arrows are controlled separately from inter-layer connections. They can be shown as lines or arrows and have independent length, width, head size, and color.

## 7. Input/output boxes

Input/output item boxes are useful for named features, variables, labels, regression outputs, and similar diagram elements. Enter one item per line.

## 8. Fixed Export panel

The fixed Export panel contains:

- Figure width
- Figure height
- DPI
- **Update Preview**
- **Smart Auto-fit**
- **Export Figure…**
- **Export All Formats**

The buttons have different visual styles so preview, fitting, single export, and multi-format export are easy to distinguish.

## 9. Recommended publication workflow

1. Start from the closest Sample.
2. Adjust layer count and visible nodes.
3. Set node/connection colors.
4. Choose straight or curved geometry.
5. Tune node radius, layer spacing, and vertical spacing.
6. Add annotations only after the architecture is stable.
7. Use Smart Auto-fit if required.
8. Set final figure dimensions and DPI.
9. Export a vector format for papers where possible, plus PNG/JPG for quick review.
10. Save the editable Project separately.

## 10. Building an EXE

Run `BUILD_EXE.bat`. It installs PyInstaller into the local `.venv`, produces one executable, and places:

`NeuralArchitectureDesigner.exe`

directly in the extracted application folder.
