#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Architecture Designer v1.0

Desktop editor for creating editable, publication-ready neural-network
architecture diagrams. Features include a built-in sample library, layer and
annotation editing, per-pair connection styles, surface-aware connection gaps,
outside-junction control, path/cascade highlighting, project save/load,
synchronized live preview, and export to PNG/JPG/SVG/PDF/EPS.
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import random
import shutil
import sys
import tempfile
import traceback
import zipfile
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser, font as tkfont
from tkinter.scrolledtext import ScrolledText

import matplotlib
# Use TkAgg on normal desktop sessions, but remain importable for headless QA/rendering.
if sys.platform.startswith("win") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from PIL import Image, ImageTk

# Explicit export backend imports. Matplotlib selects some savefig backends
# dynamically by file format; keeping direct references here makes frozen
# PyInstaller builds include every backend used by Export All Formats.
from matplotlib.backends import backend_agg as _nad_backend_agg
from matplotlib.backends import backend_svg as _nad_backend_svg
from matplotlib.backends import backend_pdf as _nad_backend_pdf
from matplotlib.backends import backend_ps as _nad_backend_ps
from PIL import JpegImagePlugin as _nad_jpeg_plugin
from PIL import PngImagePlugin as _nad_png_plugin

APP_NAME = "Neural Architecture Designer"
APP_VERSION = "1.0"
SCHEMA_VERSION = 7
BASE_DIR = Path(__file__).resolve().parent

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["mathtext.default"] = "regular"

# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class LayerSpec:
    kind: str = "FC"
    name: str = "Hidden"
    size: int = 64
    activation: str = "ReLU"
    visible_nodes: int = 6
    color: str = "#F4B400"
    shape: str = "circle"
    note: str = ""
    show_ellipsis: bool = True
    node_texts: List[str] = field(default_factory=list)
    node_text_font_size: float = 9.0
    node_text_color: str = "#111111"
    node_text_font_family: str = "Times New Roman"
    border_color: str = "#202020"
    border_width: float = 1.0
    fill_alpha: float = 1.0
    node_scale: float = 1.0
    caption_text: str = ""
    caption_color: str = "#111111"
    caption_font_size: float = 12.0
    caption_position: str = "Bottom"  # Top / Bottom / None
    caption_bold: bool = False
    x_offset: float = 0.0
    y_offset: float = 0.0


@dataclass
class PairStyleSpec:
    source_layer: int = 0
    target_layer: int = 1
    mode: str = "Inherit"  # Inherit / Dense / Sampled / Adjacent-only / None
    render: str = "Inherit"  # Inherit / Lines / Arrows
    color: str = ""
    alpha: float = -1.0
    width: float = -1.0
    arrow_head_size: float = -1.0
    gap: float = -1.0
    random_width_enabled: Optional[bool] = None
    random_width_probability: float = -1.0
    random_width_min_factor: float = -1.0
    random_width_max_factor: float = -1.0
    seed_offset: int = 0


@dataclass
class GroupSpec:
    title: str = "Hidden layers"
    start_layer: int = 0
    end_layer: int = 0
    style: str = "Text"  # Text / Pill / Box
    color: str = "#111111"
    fill: str = "#FFFFFF"
    border_color: str = ""
    font_size: float = 14.0
    bold: bool = False
    y_offset: float = 0.45
    height: float = 0.26
    padding: float = 0.22


@dataclass
class AnnotationSpec:
    kind: str = "Text"  # Text/Image/Arrow/Line/Rectangle/Rounded/Pill/Circle/Divider
    text: str = ""
    x: float = 0.0
    y: float = 0.0
    x2: float = 1.0
    y2: float = 0.0
    width: float = 1.0
    height: float = 0.4
    fill: str = "#FFFFFF"
    border_color: str = "#222222"
    color: str = "#111111"
    alpha: float = 1.0
    line_width: float = 1.0
    line_style: str = "Solid"  # Solid / Dashed / Dotted
    font_family: str = "Times New Roman"
    font_size: float = 12.0
    bold: bool = False
    italic: bool = False
    ha: str = "center"
    va: str = "center"
    rotation: float = 0.0
    image_path: str = ""
    image_scale: float = 0.5
    preserve_alpha: bool = True
    zorder: float = 10.0
    locked: bool = False
    arrow_head_size: float = 10.0


@dataclass
class ProjectSpec:
    schema_version: int = SCHEMA_VERSION
    app_version: str = APP_VERSION
    title: str = ""
    subtitle: str = ""
    background: str = "#FFFFFF"
    font_family: str = "Times New Roman"
    item_font_family: str = "Times New Roman"
    math_fontset: str = "stix"
    title_font_size: float = 18.0
    layer_font_size: float = 12.0
    item_font_size: float = 11.0
    node_radius: float = 0.13
    layer_spacing: float = 1.65
    vertical_spacing: float = 0.42
    connection_color: str = "#3F6FA7"
    connection_color_mode: str = "Single"  # Single / Palette cycle / Random palette
    connection_palette: str = "#FF4FA3, #66E5FF, #7D8CFF, #7FE07F, #F7C948, #FF8A3D"
    connection_alpha: float = 0.65
    connection_width: float = 0.75
    connection_mode: str = "Sampled"
    connection_style: str = "Lines"
    connection_geometry: str = "Straight"  # Straight / Curved-horizontal
    connection_end_mode: str = "Trim at cells"  # Trim at cells / Outside junctions
    junction_offset: float = 0.08
    curve_strength: float = 0.32
    max_connections_per_pair: int = 220
    line_gap: float = 0.035
    arrow_gap: float = 0.055
    arrow_head_size: float = 9.0
    random_width_enabled: bool = False
    random_width_probability: float = 0.18
    random_width_min_factor: float = 1.6
    random_width_max_factor: float = 3.0
    random_seed: int = 7
    show_input_arrows: bool = False
    show_output_arrows: bool = False
    external_arrow_style: str = "Arrows"  # Arrows / Lines
    external_arrow_color: str = "#222222"
    external_arrow_length: float = 0.55
    external_arrow_gap: float = 0.06
    external_arrow_width: float = 1.0
    external_arrow_head_size: float = 9.0
    margins: float = 0.45
    figure_width: float = 14.0
    figure_height: float = 7.5
    dpi: int = 600
    show_nominal_size: bool = True
    show_activation: bool = True
    show_input_items: bool = False
    show_output_items: bool = False
    input_group_title: str = "Model inputs"
    output_group_title: str = "Model outputs"
    input_box_width: float = 0.0  # 0 = auto
    output_box_width: float = 0.0
    input_items: List[str] = field(default_factory=list)
    output_items: List[str] = field(default_factory=list)
    # Compatibility fields for loading project files. Global card/panel framing is not rendered.
    card_enabled: bool = False
    card_fill: str = "#FFFFFF"
    card_border: str = "#E8E8E8"
    card_rounding: float = 0.025
    card_shadow: bool = False
    layout_mode: str = "Automatic"  # Automatic / Semi-Automatic
    layers: List[LayerSpec] = field(default_factory=list)
    pair_styles: List[PairStyleSpec] = field(default_factory=list)
    groups: List[GroupSpec] = field(default_factory=list)
    annotations: List[AnnotationSpec] = field(default_factory=list)
    highlight_paths: List[Dict[str, Any]] = field(default_factory=list)
    reference_image: str = ""  # reserved project field


# -----------------------------------------------------------------------------
# Compatibility / validation helpers
# -----------------------------------------------------------------------------

def _filter_dataclass_kwargs(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def project_from_dict(data: Dict[str, Any]) -> ProjectSpec:
    """Load v4 and most v3 JSONs without failing on unknown/missing fields."""
    data = copy.deepcopy(data)
    raw_layers = data.pop("layers", [])
    raw_pairs = data.pop("pair_styles", [])
    raw_groups = data.pop("groups", [])
    raw_annotations = data.pop("annotations", [])
    raw_highlight_paths = data.pop("highlight_paths", [])

    # v3 compatibility: orientation/show_group_boxes/show_layer_count can be ignored.
    data.pop("orientation", None)
    data.pop("show_group_boxes", None)
    data.pop("show_layer_count", None)

    p = ProjectSpec(**_filter_dataclass_kwargs(ProjectSpec, data))
    p.schema_version = SCHEMA_VERSION
    p.app_version = APP_VERSION
    p.layers = [LayerSpec(**_filter_dataclass_kwargs(LayerSpec, x)) for x in raw_layers]
    p.pair_styles = [PairStyleSpec(**_filter_dataclass_kwargs(PairStyleSpec, x)) for x in raw_pairs]
    p.groups = [GroupSpec(**_filter_dataclass_kwargs(GroupSpec, x)) for x in raw_groups]
    p.annotations = [AnnotationSpec(**_filter_dataclass_kwargs(AnnotationSpec, x)) for x in raw_annotations]
    p.highlight_paths = copy.deepcopy(raw_highlight_paths) if isinstance(raw_highlight_paths, list) else []
    return p


def validate_project(p: ProjectSpec) -> None:
    if not p.layers:
        raise ValueError("Add at least one layer.")
    if not (0.0 <= p.connection_alpha <= 1.0):
        raise ValueError("Connection alpha must be between 0 and 1.")
    if p.connection_width < 0:
        raise ValueError("Connection width must be non-negative.")
    if p.max_connections_per_pair < 1:
        raise ValueError("Max connections per pair must be at least 1.")
    if p.node_radius <= 0:
        raise ValueError("Node radius must be positive.")
    if p.layer_spacing <= 0 or p.vertical_spacing <= 0:
        raise ValueError("Layer/vertical spacing must be positive.")
    if p.dpi < 72 or p.dpi > 2400:
        raise ValueError("Export DPI must be between 72 and 2400.")
    if p.junction_offset < -2.0:
        raise ValueError("Junction offset is too negative. Use a value above -2.0.")
    if p.curve_strength < 0:
        raise ValueError("Curve strength must be non-negative.")
    for i, l in enumerate(p.layers):
        if l.visible_nodes < 1:
            raise ValueError(f"Layer {i+1}: visible nodes must be at least 1.")
        if not (0.0 <= l.fill_alpha <= 1.0):
            raise ValueError(f"Layer {i+1}: fill alpha must be between 0 and 1.")
        if l.node_scale <= 0:
            raise ValueError(f"Layer {i+1}: node scale must be positive.")
    for hp in getattr(p, "highlight_paths", []) or []:
        if not isinstance(hp, dict):
            raise ValueError("Each highlight path must be a dictionary.")
        nodes = hp.get("nodes", [])
        if nodes and len(nodes) != len(p.layers):
            raise ValueError("Each highlight path must define one node index per layer.")

    for s in p.pair_styles:
        if not (0 <= s.source_layer < len(p.layers) and 0 <= s.target_layer < len(p.layers)):
            raise ValueError("A pair style points to a layer that no longer exists.")
        if s.alpha != -1.0 and not (0 <= s.alpha <= 1):
            raise ValueError("Pair-style alpha must be between 0 and 1.")


# -----------------------------------------------------------------------------
# Preset helpers
# -----------------------------------------------------------------------------

def L(kind, name, size, shown, fill, *, shape="circle", activation="", texts=None,
      border="#202020", border_width=1.0, fill_alpha=1.0, scale=1.0,
      caption="", caption_color="#111111", caption_size=11, caption_pos="Bottom",
      note="", xoff=0.0, yoff=0.0, text_color="#111111", text_size=9.0,
      show_ellipsis=True) -> LayerSpec:
    return LayerSpec(
        kind=kind, name=name, size=size, activation=activation, visible_nodes=shown,
        color=fill, shape=shape, note=note, show_ellipsis=show_ellipsis,
        node_texts=list(texts or []), node_text_font_size=text_size,
        node_text_color=text_color, border_color=border, border_width=border_width,
        fill_alpha=fill_alpha, node_scale=scale,
        caption_text=caption, caption_color=caption_color,
        caption_font_size=caption_size, caption_position=caption_pos,
        x_offset=xoff, y_offset=yoff,
    )


def PS(a, b, *, mode="Inherit", render="Inherit", color="", alpha=-1.0, width=-1.0,
       head=-1.0, gap=-1.0, random_on=None, prob=-1.0, minf=-1.0, maxf=-1.0, seed=0):
    return PairStyleSpec(a, b, mode, render, color, alpha, width, head, gap,
                         random_on, prob, minf, maxf, seed)


def A(kind, **kwargs) -> AnnotationSpec:
    a = AnnotationSpec(kind=kind)
    for k, v in kwargs.items():
        if hasattr(a, k):
            setattr(a, k, v)
    return a


def G(title, start, end, *, style="Text", color="#111111", fill="#FFFFFF",
      border="", size=14, bold=False, yoff=0.45, height=0.26, padding=0.22):
    return GroupSpec(title, start, end, style, color, fill, border, size, bold, yoff, height, padding)


def _math_nodes(layer_index: int, n: int) -> List[str]:
    return [rf"$a_{{{i}}}^{{({layer_index})}}$" for i in range(1, n+1)]


def sample_01_formula_network() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=14.5, figure_height=8.0,
                    node_radius=0.24, layer_spacing=1.95, vertical_spacing=0.55,
                    connection_color="#0B0E78", connection_alpha=0.92,
                    connection_width=0.95, connection_mode="Dense", connection_style="Arrows",
                    connection_geometry="Straight", connection_end_mode="Trim at cells",
                    arrow_head_size=8.0, line_gap=0.0, arrow_gap=0.02, margins=0.55,
                    show_nominal_size=False, show_activation=False,
                    reference_image="")
    p.layers = [
        L("Input", "Input", 4, 4, "#BFE9BE", border="#075D17", border_width=2.0,
          texts=_math_nodes(0,4), text_size=15, caption_pos="None"),
        L("FC", "Hidden 1", 5, 5, "#C7C9F1", border="#13146E", border_width=2.0,
          texts=_math_nodes(1,5), text_size=15, caption_pos="None"),
        L("FC", "Hidden 2", 5, 5, "#C7C9F1", border="#13146E", border_width=2.0,
          texts=_math_nodes(2,5), text_size=15, caption_pos="None"),
        L("FC", "Hidden 3", 5, 5, "#C7C9F1", border="#13146E", border_width=2.0,
          texts=_math_nodes(3,5), text_size=15, caption_pos="None"),
        L("Output", "Output", 3, 3, "#F2C9C9", border="#7A0000", border_width=2.0,
          texts=_math_nodes(4,3), text_size=15, caption_pos="None"),
    ]
    p.groups = [
        G("input layer",0,0,color="#008B16",size=19,bold=False,yoff=0.62),
        G("hidden layers",1,3,color="#071B8D",size=21,bold=False,yoff=0.72),
        G("output layer",4,4,color="#A00000",size=19,bold=False,yoff=0.62),
    ]
    p.pair_styles = [
        PS(0,1,color="#1D8F3A",render="Arrows",gap=0.02,width=0.98),
        PS(1,2,color="#0B0E78",render="Arrows",gap=0.02,width=0.95),
        PS(2,3,color="#0B0E78",render="Arrows",gap=0.02,width=0.95),
        PS(3,4,color="#B22222",render="Arrows",gap=0.02,width=0.98),
    ]
    return p


def sample_02_activation_blocks() -> ProjectSpec:
    p = ProjectSpec(background="#F7F7F7", figure_width=16, figure_height=8.0,
                    node_radius=0.15, layer_spacing=1.18, vertical_spacing=0.70,
                    connection_color="#5DB1E5", connection_color_mode="Random palette",
                    connection_palette="#4FB3E8, #37A2D9, #7A87FF, #5CD5D8, #36C7B7, #C16BE6",
                    connection_alpha=0.95, connection_width=1.25, connection_mode="Adjacent-only", connection_style="Arrows",
                    connection_geometry="Straight", connection_end_mode="Trim at cells",
                    arrow_head_size=8.5, arrow_gap=0.08, show_nominal_size=False,
                    show_activation=False, margins=0.5,
                    random_width_enabled=True, random_width_probability=1.0, random_width_min_factor=1.6, random_width_max_factor=4.6, random_seed=22,
                    reference_image="")
    p.layers = [
        L("Input","Input",2,2,"#5A9BD4",border="#1E4C70",scale=1.05,caption="raw",caption_size=10,caption_pos="Bottom"),
        L("FC","Affine A",4,4,"#B9DDF5",border="#245B83",caption="affine",caption_size=10,caption_pos="Bottom"),
        L("Activation","Sigmoid A",4,4,"#F2F2F2",shape="activation",activation="Sigmoid",border="#3976A4",scale=1.28,caption="activate",caption_size=10,caption_pos="Bottom"),
        L("FC","Feature A",4,4,"#55E8EF",border="#1C8194",caption="features",caption_size=10,caption_pos="Bottom"),
        L("FC","Affine B",5,5,"#B9DDF5",border="#245B83",caption="mix",caption_size=10,caption_pos="Bottom"),
        L("Activation","Sigmoid B",5,5,"#F2F2F2",shape="activation",activation="Sigmoid",border="#3976A4",scale=1.28,caption="gate",caption_size=10,caption_pos="Bottom"),
        L("FC","Feature B",5,5,"#55E8EF",border="#1C8194",caption="refine",caption_size=10,caption_pos="Bottom"),
        L("FC","Affine C",3,3,"#B9DDF5",border="#245B83",caption="project",caption_size=10,caption_pos="Bottom"),
        L("Activation","Sigmoid C",3,3,"#F2F2F2",shape="activation",activation="Sigmoid",border="#3976A4",scale=1.28,caption="score",caption_size=10,caption_pos="Bottom"),
        L("Output","Output",3,3,"#F07AE6",border="#6B2C77",caption="output",caption_size=10,caption_pos="Bottom"),
    ]
    p.pair_styles = [
        PS(0,1,mode="Dense",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.6,maxf=4.2,seed=11),
        PS(1,2,mode="Adjacent-only",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.6,maxf=4.2,seed=12),
        PS(2,3,mode="Adjacent-only",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.6,maxf=4.2,seed=13),
        PS(3,4,mode="Dense",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.8,maxf=4.8,seed=14),
        PS(4,5,mode="Adjacent-only",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.6,maxf=4.2,seed=15),
        PS(5,6,mode="Adjacent-only",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.6,maxf=4.2,seed=16),
        PS(6,7,mode="Dense",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.8,maxf=4.8,seed=17),
        PS(7,8,mode="Adjacent-only",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.6,maxf=4.2,seed=18),
        PS(8,9,mode="Adjacent-only",render="Arrows",color="#5DB1E5",width=1.15,random_on=True,prob=1.0,minf=1.6,maxf=4.2,seed=19),
    ]
    return p


def sample_03_external_arrows() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=11.5, figure_height=5.3,
                    node_radius=0.18, layer_spacing=1.55, vertical_spacing=0.48,
                    connection_color="#2B2B2B", connection_alpha=0.85,
                    connection_width=0.65, connection_mode="Dense", connection_style="Lines",
                    connection_geometry="Curved-horizontal", connection_end_mode="Trim at cells", junction_offset=.08, curve_strength=.36,
                    line_gap=0.03,
                    show_input_arrows=True, show_output_arrows=True,
                    external_arrow_style="Arrows", external_arrow_color="#202020",
                    external_arrow_width=1.0, external_arrow_length=0.65, external_arrow_gap=0.08,
                    show_nominal_size=False, show_activation=False, margins=0.5,
                    reference_image="")
    p.layers = [
        L("Input","Input",3,3,"#FFB20F",border="#FFB20F",caption_pos="None"),
        L("FC","Hidden 1",4,4,"#0D4B5A",border="#0D4B5A",caption_pos="None"),
        L("FC","Hidden 2",4,4,"#0D4B5A",border="#0D4B5A",caption_pos="None"),
        L("FC","Hidden 3",4,4,"#0D4B5A",border="#0D4B5A",caption_pos="None"),
        L("Output","Output",1,1,"#FF7133",border="#FF7133",caption_pos="None"),
    ]
    return p


def sample_04_card_deep_network() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=13.0, figure_height=6.0,
                    node_radius=0.18, layer_spacing=1.52, vertical_spacing=0.48,
                    connection_color="#27B9D0", connection_alpha=0.84,
                    connection_width=0.78, connection_mode="Dense", connection_style="Arrows",
                    connection_geometry="Straight", connection_end_mode="Outside junctions", junction_offset=.075,
                    arrow_head_size=7.5, arrow_gap=0.09, show_nominal_size=False, show_activation=False,
                    random_width_enabled=True, random_width_probability=.20,
                    random_width_min_factor=1.4, random_width_max_factor=3.0,
                    margins=0.6, card_enabled=False,
                    reference_image="")
    p.layers = [
        L("Input","Input",5,5,"#0C63FF",border="#0C63FF",caption_pos="None"),
        L("FC","Hidden 1",5,5,"#36CDD1",border="#36CDD1",caption_pos="None"),
        L("FC","Hidden 2",5,5,"#36CDD1",border="#36CDD1",caption_pos="None"),
        L("FC","Hidden 3",5,5,"#36CDD1",border="#36CDD1",caption_pos="None"),
        L("Output","Output",3,3,"#719CF4",border="#719CF4",caption_pos="None"),
    ]
    p.groups = [
        G("Input layer",0,0,color="#111111",size=11,yoff=.55),
        G("Multiple hidden layer",1,3,color="#111111",size=11,yoff=.55),
        G("Output layer",4,4,color="#111111",size=11,yoff=.55),
    ]
    p.pair_styles = [
        PS(0,1,color="#0D6BFF",render="Arrows",gap=.10,random_on=True,prob=.26,minf=1.4,maxf=3.2,seed=11),
        PS(1,2,color="#22C9D2",render="Arrows",gap=.10,random_on=True,prob=.24,minf=1.4,maxf=3.0,seed=12),
        PS(2,3,color="#74E1E4",render="Arrows",gap=.10,random_on=True,prob=.22,minf=1.4,maxf=2.8,seed=13),
        PS(3,4,color="#2F79F7",render="Arrows",gap=.10,random_on=True,prob=.26,minf=1.4,maxf=3.2,seed=14),
    ]
    return p


def sample_05_pill_headers() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=12.8, figure_height=5.2,
                    node_radius=0.17, layer_spacing=1.65, vertical_spacing=0.48,
                    connection_color="#343A40", connection_alpha=0.86,
                    connection_width=0.68, connection_mode="Dense", connection_style="Lines",
                    connection_geometry="Curved-horizontal", connection_end_mode="Outside junctions", junction_offset=.07, curve_strength=.34,
                    line_gap=0.07,
                    show_output_arrows=True, external_arrow_style="Arrows", external_arrow_color="#434A54",
                    show_nominal_size=False, show_activation=False, margins=0.55,
                    reference_image="")
    p.layers = [
        L("Input","Input",2,2,"#FF9C3A",border="#6C5B48",caption_pos="None"),
        L("FC","Hidden 1",3,3,"#18A9E2",border="#345669",caption_pos="None"),
        L("FC","Hidden 2",3,3,"#18A9E2",border="#345669",caption_pos="None"),
        L("FC","Hidden 3",3,3,"#18A9E2",border="#345669",caption_pos="None"),
        L("Output","Output",1,1,"#F0102D",border="#F0102D",caption_pos="None"),
    ]
    p.groups = [
        G("Input layer",0,0,style="Text",color="#333333",size=10,yoff=.62),
        G("Hidden layers",1,3,style="Text",color="#333333",size=10,yoff=.62),
        G("Output layer",4,4,style="Text",color="#333333",size=10,yoff=.62),
    ]
    p.pair_styles = [
        PS(0,1,color="#4A4A4A",width=.75),
        PS(1,2,color="#4A4A4A",width=.72),
        PS(2,3,color="#4A4A4A",width=.72),
        PS(3,4,color="#4A4A4A",width=.78),
    ]
    p.annotations = [
        A("Text",text="Nodes",x=-1.05,y=.24,font_family="Arial",font_size=9,color="#555555",ha="right"),
        A("Text",text="Nodes",x=-1.05,y=-.24,font_family="Arial",font_size=9,color="#555555",ha="right"),
        A("Arrow",x=-.82,y=.24,x2=-.16,y2=.24,color="#555555",line_width=.9,arrow_head_size=7),
        A("Arrow",x=-.82,y=-.24,x2=-.16,y2=-.24,color="#555555",line_width=.9,arrow_head_size=7),
    ]
    return p


def sample_06_paper_dividers() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=14.8, figure_height=8.0,
                    node_radius=0.15, layer_spacing=1.72, vertical_spacing=0.56,
                    connection_color="#111111", connection_alpha=0.88,
                    connection_width=0.72, connection_mode="Dense", connection_style="Lines",
                    line_gap=0.045, show_nominal_size=False, show_activation=False, margins=0.72,
                    reference_image="")
    # Ellipsis is a layout slot, not text drawn on top of a neuron.
    # Input: 3 shown neurons + one ellipsis slot. Hidden: 4 shown + ellipsis. Output: 2 shown + ellipsis.
    p.layers = [
        L("Input","Input",10,4,"#9AAEB0",border="#1E292A",caption_pos="None",show_ellipsis=True),
        L("FC","h1",10,5,"#9AAEB0",border="#1E292A",caption_pos="None",show_ellipsis=True),
        L("FC","h2",10,5,"#9AAEB0",border="#1E292A",caption_pos="None",show_ellipsis=True),
        L("FC","hn",10,5,"#9AAEB0",border="#1E292A",caption_pos="None",show_ellipsis=True),
        L("Output","Output",10,3,"#9AAEB0",border="#1E292A",caption_pos="None",show_ellipsis=True),
    ]
    p.groups = [
        G("Input layer",0,0,color="#111111",size=16,yoff=1.02),
        G("Hidden layers",1,3,color="#111111",size=16,yoff=1.02),
        G("Output layer",4,4,color="#111111",size=16,yoff=1.02),
    ]
    p.annotations = [
        A("Text",text=r"$i$",x=0,y=2.02,font_size=14),
        A("Text",text=r"$h_1$",x=1.72,y=2.02,font_size=14),
        A("Text",text=r"$h_2$",x=3.44,y=2.02,font_size=14),
        A("Text",text=r"$h_n$",x=5.16,y=2.02,font_size=14),
        A("Text",text=r"$o$",x=6.88,y=2.02,font_size=14),
        A("Divider",x=.86,y=-2.08,x2=.86,y2=2.22,color="#111111",line_width=.8,line_style="Dotted",zorder=1),
        A("Divider",x=6.02,y=-2.08,x2=6.02,y2=2.22,color="#111111",line_width=.8,line_style="Dotted",zorder=1),
        A("Text",text="Input 1",x=-1.14,y=.84,font_size=12,ha="right"),
        A("Text",text="Input 2",x=-1.14,y=.28,font_size=12,ha="right"),
        A("Text",text="Input n",x=-1.14,y=-.84,font_size=12,ha="right"),
        A("Arrow",x=-1.05,y=.84,x2=-.18,y2=.84,color="#111111",line_width=.9,arrow_head_size=8),
        A("Arrow",x=-1.05,y=.28,x2=-.18,y2=.28,color="#111111",line_width=.9,arrow_head_size=8),
        A("Arrow",x=-1.05,y=-.84,x2=-.18,y2=-.84,color="#111111",line_width=.9,arrow_head_size=8),
        A("Text",text="Output 1",x=8.10,y=.56,font_size=12,ha="left"),
        A("Text",text="Output n",x=8.10,y=-.56,font_size=12,ha="left"),
        A("Arrow",x=7.12,y=.56,x2=7.98,y2=.56,color="#111111",line_width=.9,arrow_head_size=8),
        A("Arrow",x=7.12,y=-.56,x2=7.98,y2=-.56,color="#111111",line_width=.9,arrow_head_size=8),
    ]
    return p

def sample_07_mixed_shapes() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=11.2, figure_height=6.3,
                    node_radius=0.14, layer_spacing=1.4, vertical_spacing=0.60,
                    connection_color="#B5B5B5", connection_color_mode="Random palette",
                    connection_palette="#E83F5B, #49BEB7, #F6A623, #4CAF72, #4A73B5, #999999",
                    connection_alpha=0.74,
                    connection_width=0.72, connection_mode="Dense", connection_style="Lines",
                    connection_geometry="Curved-horizontal", line_gap=0.055,
                    random_width_enabled=True, random_width_probability=0.85, random_width_min_factor=1.4, random_width_max_factor=3.2, random_seed=23,
                    show_input_arrows=True, show_output_arrows=True,
                    external_arrow_style="Lines", external_arrow_color="#A0A0A0",
                    external_arrow_width=.82, external_arrow_length=.6, external_arrow_gap=.08,
                    show_nominal_size=False, show_activation=False, margins=.45,
                    reference_image="")
    p.layers = [
        L("Input","Circles",5,5,"#FFFFFF",shape="circle",border="#E83F5B",border_width=2.2,fill_alpha=1.0,caption_pos="None"),
        L("FC","Squares",6,6,"#FFFFFF",shape="square",border="#49BEB7",border_width=2.0,fill_alpha=1.0,caption_pos="None"),
        L("FC","Triangles",6,6,"#FFFFFF",shape="triangle",border="#F6A623",border_width=2.0,fill_alpha=1.0,caption_pos="None"),
        L("FC","Diamonds",6,6,"#FFFFFF",shape="diamond",border="#4CAF72",border_width=2.0,fill_alpha=1.0,caption_pos="None"),
        L("Output","Hexagons",5,5,"#FFFFFF",shape="hexagon",border="#4A73B5",border_width=2.0,fill_alpha=1.0,caption_pos="None"),
    ]
    p.pair_styles = [
        PS(0,1,color="#CBCBCB",width=.62,random_on=True,prob=.8,minf=1.4,maxf=2.8,seed=8),
        PS(1,2,color="#C7C7C7",width=.62,random_on=True,prob=.9,minf=1.4,maxf=3.0,seed=9),
        PS(2,3,color="#C7C7C7",width=.62,random_on=True,prob=.9,minf=1.4,maxf=3.0,seed=10),
        PS(3,4,color="#CBCBCB",width=.62,random_on=True,prob=.8,minf=1.4,maxf=2.8,seed=11),
    ]
    return p


def sample_08_dog_classifier() -> ProjectSpec:
    p = ProjectSpec(background="#162856", figure_width=15.2, figure_height=7.0,
                    node_radius=0.15, layer_spacing=1.58, vertical_spacing=0.49,
                    connection_color="#4C81E8", connection_alpha=0.72,
                    connection_width=0.78, connection_mode="Dense", connection_style="Lines",
                    connection_geometry="Straight", connection_end_mode="Outside junctions", junction_offset=.065,
                    line_gap=0.045,
                    random_width_enabled=True, random_width_probability=.18,
                    random_width_min_factor=1.5, random_width_max_factor=3.2,
                    random_seed=41, show_nominal_size=False, show_activation=False, margins=.65,
                    reference_image="")
    p.layers = [
        L("Input","L1",5,5,"#4C82E7",border="#4C82E7",texts=["1","0","1","1","1"],text_color="#16305F",text_size=10,caption_pos="None"),
        L("FC","L2",5,5,"#49BBD2",border="#49BBD2",texts=["1","0","1","0","1"],text_color="#17356B",text_size=10,caption_pos="None"),
        L("FC","L3",5,5,"#9AD8D1",border="#9AD8D1",texts=["1","1","0","1","1"],text_color="#17356B",text_size=10,caption_pos="None"),
        L("FC","L4",5,5,"#D7E85B",border="#D7E85B",texts=["1","0","0","1","1"],text_color="#17356B",text_size=10,caption_pos="None"),
        L("Output","L5",3,3,"#64D95D",border="#64D95D",texts=["0","1","0"],text_color="#16305F",text_size=10,caption_pos="None"),
    ]
    p.pair_styles = [
        PS(0,1,color="#5284E8",width=.78,random_on=True,prob=.22,minf=1.3,maxf=3.0,seed=1),
        PS(1,2,color="#42C0D4",width=.78,random_on=True,prob=.18,minf=1.3,maxf=3.0,seed=2),
        PS(2,3,color="#8ED4C7",width=.78,random_on=True,prob=.18,minf=1.3,maxf=3.0,seed=3),
        PS(3,4,color="#D5E957",width=.88,random_on=True,prob=.20,minf=1.4,maxf=3.4,seed=4),
    ]
    generic = str((BASE_DIR / "assets" / "generic_input_image.png").resolve())
    p.annotations = [
        A("Image",image_path=generic,x=-1.92,y=.30,image_scale=.54,zorder=20),
        A("Text",text="INPUT\nGeneric image",x=-1.92,y=-1.08,font_family="Arial",font_size=12,
          color="#AFC0DE",ha="center",va="top",zorder=30),
        A("Text",text="OUTPUT\nPredicted\nclass",x=7.00,y=.14,font_family="Arial",font_size=13,color="#AFC0DE",ha="left",zorder=30),
        A("Arrow",x=6.22,y=.06,x2=6.82,y2=.06,color="#AFC0DE",line_width=.8,arrow_head_size=7,zorder=8),
        A("Text",text="L1\nLow-level\nintensities",x=0,y=-1.52,font_family="Arial",font_size=10,color="#6D9AF4",va="top"),
        A("Text",text="L2\nLocal\npatterns",x=1.58,y=-1.52,font_family="Arial",font_size=10,color="#56C5E0",va="top"),
        A("Text",text="L3\nMid-level\nfeatures",x=3.16,y=-1.52,font_family="Arial",font_size=10,color="#9CDAD3",va="top"),
        A("Text",text="L4\nSemantic\nfeatures",x=4.74,y=-1.52,font_family="Arial",font_size=10,color="#D7EB5D",va="top"),
        A("Text",text="L5\nDecision\nspace",x=6.32,y=-1.52,font_family="Arial",font_size=10,color="#69DE61",va="top"),
    ]
    for y in [.98,.49,0,-.49,-.98]:
        p.annotations.append(A("Arrow",x=-1.15,y=y*.62,x2=-.20,y2=y,color="#9AB5D8",line_width=.70,arrow_head_size=6,zorder=6))
    return p


# Built-in architecture samples.
def classic_deep_mlp() -> ProjectSpec:
    p = sample_03_external_arrows()
    p.reference_image = ""
    p.show_input_arrows = False; p.show_output_arrows = False
    p.layers = [
        L("Input","Input",4,4,"#49A35B",border="#2A6734",caption="Input"),
        L("FC","Hidden 1",64,7,"#4567D6",border="#27458F",activation="ReLU",caption="Hidden 1"),
        L("FC","Hidden 2",32,6,"#55B985",border="#2F7B58",activation="ReLU",caption="Hidden 2"),
        L("Output","Output",2,2,"#F0A6A6",border="#A75757",activation="Linear",caption="Output"),
    ]
    p.connection_style="Arrows"; p.connection_geometry="Curved-horizontal"; p.connection_end_mode="Outside junctions"; p.junction_offset=.07; p.curve_strength=.34; p.connection_color="#5A6C9A"; p.background="#FFFFFF"
    p.connection_alpha=.70; p.connection_width=.78; p.arrow_gap=.08
    p.random_width_enabled=True; p.random_width_probability=.16; p.random_width_min_factor=1.4; p.random_width_max_factor=2.8
    p.pair_styles=[
        PS(0,1,color="#6880C8",render="Arrows",random_on=True,prob=.16,minf=1.4,maxf=2.4,seed=3),
        PS(1,2,color="#6A8BCD",render="Arrows",random_on=True,prob=.18,minf=1.4,maxf=2.8,seed=4),
        PS(2,3,color="#7588C7",render="Arrows",random_on=True,prob=.16,minf=1.4,maxf=2.4,seed=5),
    ]
    p.show_nominal_size=True; p.show_activation=True
    return p


def classic_mlp_regression() -> ProjectSpec:
    p = classic_deep_mlp()
    p.layers[0].size=6; p.layers[0].visible_nodes=6
    p.layers[1].size=64; p.layers[1].visible_nodes=8
    p.layers[2].size=32; p.layers[2].visible_nodes=7
    p.layers[3].size=3; p.layers[3].visible_nodes=3
    p.input_items=[r"$x_1$",r"$x_2$",r"$x_3$",r"$x_4$",r"$x_5$",r"$x_6$"]; p.show_input_items=True
    p.output_items=[r"$y_1$",r"$y_2$",r"$y_3$"]; p.show_output_items=True
    p.input_group_title="Input features"; p.output_group_title="Regression outputs"
    return p


def classic_classifier() -> ProjectSpec:
    p = classic_deep_mlp()
    p.layers[-1].size=3; p.layers[-1].visible_nodes=3; p.layers[-1].activation="Softmax"; p.layers[-1].color="#F0A6A6"; p.layers[-1].border_color="#A75757"
    p.layers[-1].caption_text="Class probabilities"
    p.pair_styles[-1]=PS(2,3,color="#7A88C8",render="Arrows",random_on=True,prob=.18,minf=1.4,maxf=2.6,seed=16)
    return p


def classic_autoencoder() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=13, figure_height=6.5,
                    node_radius=.13, layer_spacing=1.6, vertical_spacing=.43,
                    connection_color="#6A6A8D",connection_alpha=.60,connection_width=.68,
                    connection_mode="Sampled",connection_style="Arrows",connection_geometry="Straight",connection_end_mode="Outside junctions",junction_offset=.065,arrow_gap=.075,
                    random_width_enabled=True, random_width_probability=.12, random_width_min_factor=1.4, random_width_max_factor=2.4)
    p.layers=[
        L("Input","Input",128,8,"#64B96A",border="#34723A",caption="Input"),
        L("FC","Encoder",64,7,"#E8A928",border="#9A6B0C",activation="ReLU",caption="Encoder"),
        L("FC","Latent",16,5,"#8C65C6",border="#584183",activation="Linear",caption="Latent space"),
        L("FC","Decoder",64,7,"#6DB86B",border="#3A793A",activation="ReLU",caption="Decoder"),
        L("Output","Reconstruction",128,8,"#F3F3F3",border="#666666",activation="Linear",caption="Reconstruction"),
    ]
    p.pair_styles=[
        PS(0,1,color="#8B939B",render="Arrows"),
        PS(1,2,color="#8A74B7",render="Arrows",random_on=True,prob=.14,minf=1.4,maxf=2.4,seed=6),
        PS(2,3,color="#88A37C",render="Arrows",random_on=True,prob=.14,minf=1.4,maxf=2.4,seed=7),
        PS(3,4,color="#A3A3B7",render="Arrows"),
    ]
    return p


def classic_lenet() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF",figure_width=15,figure_height=6.8,node_radius=.12,
                    layer_spacing=1.5,vertical_spacing=.42,connection_color="#4777A9",connection_alpha=.64,
                    connection_width=.66,connection_mode="Sampled",connection_style="Arrows",connection_geometry="Curved-horizontal",connection_end_mode="Outside junctions",junction_offset=0.0,curve_strength=.32,arrow_gap=.07,
                    random_width_enabled=True, random_width_probability=.10, random_width_min_factor=1.3, random_width_max_factor=2.2)
    p.layers=[
        L("Input","Image",1024,6,"#62BB6A",shape="square",border="#34723A",caption="32×32 input"),
        L("Conv","Conv 1",6,6,"#4E83C0",shape="rounded",activation="Tanh",caption="Conv 5×5"),
        L("Pool","Pool 1",6,5,"#A5AAB3",shape="diamond",activation="AvgPool",caption="AvgPool"),
        L("Conv","Conv 2",16,7,"#4E83C0",shape="rounded",activation="Tanh",caption="Conv 5×5"),
        L("FC","FC 1",120,7,"#E6B33E",activation="Tanh",caption="FC 120"),
        L("FC","FC 2",84,7,"#79BB65",activation="Tanh",caption="FC 84"),
        L("Output","Output",10,8,"#F5F5F5",border="#666666",activation="Softmax",caption="10 classes"),
    ]
    p.pair_styles=[
        PS(0,1,color="#7AA4CC",render="Arrows"),
        PS(1,2,color="#A4B1C4",render="Arrows"),
        PS(2,3,color="#7AA4CC",render="Arrows",random_on=True,prob=.12,minf=1.3,maxf=2.1,seed=31),
        PS(3,4,color="#D3A655",render="Arrows"),
        PS(4,5,color="#A1B96E",render="Arrows"),
        PS(5,6,color="#C3C3C3",render="Arrows"),
    ]
    return p




def sample_09_monochrome_dense_labels() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=14.2, figure_height=7.6,
                    node_radius=0.16, layer_spacing=1.36, vertical_spacing=0.50,
                    connection_color="#111111", connection_alpha=0.95, connection_width=0.92,
                    connection_mode="Dense", connection_style="Arrows", connection_geometry="Straight",
                    connection_end_mode="Trim at cells", line_gap=0.0, arrow_gap=0.0, arrow_head_size=8.0,
                    show_nominal_size=False, show_activation=False, margins=0.48)
    p.layers = [
        L("Input","Input",8,8,"#C7D6E6",border="#202020",caption_pos="None"),
        L("FC","Hidden 1",9,9,"#C7D6E6",border="#202020",caption_pos="None"),
        L("FC","Hidden 2",8,8,"#C7D6E6",border="#202020",caption_pos="None"),
        L("FC","Hidden 3",9,9,"#C7D6E6",border="#202020",caption_pos="None"),
        L("Output","Output",4,4,"#C7D6E6",border="#202020",caption_pos="None"),
    ]
    p.groups = [G("input layer",0,0,size=13,yoff=.62), G("hidden layer 1",1,1,size=13,yoff=.62), G("hidden layer 2",2,2,size=13,yoff=.62), G("hidden layer 3",3,3,size=13,yoff=.62), G("output layer",4,4,size=13,yoff=.62)]
    p.show_output_arrows = True
    p.external_arrow_style = "Arrows"
    p.external_arrow_length = 0.52
    return p

def sample_10_neon_gradient_dark() -> ProjectSpec:
    p = ProjectSpec(background="#000000", figure_width=13.8, figure_height=4.8,
                    node_radius=0.17, layer_spacing=3.0, vertical_spacing=1.0,
                    connection_color="#FF4FA3", connection_color_mode="Palette cycle",
                    connection_palette="#FF4FA3, #FFC2D5, #77F3FF, #82A6FF, #8D80FF",
                    connection_alpha=0.95, connection_width=1.05, connection_mode="Dense",
                    connection_style="Lines", connection_geometry="Curved-horizontal", line_gap=0.035,
                    show_nominal_size=False, show_activation=False, margins=0.50, font_family="DejaVu Sans",
                    random_width_enabled=True, random_width_probability=1.0, random_width_min_factor=1.8, random_width_max_factor=5.0, random_seed=29)
    p.layers = [
        L("Input","Input",5,5,"#FF4FA3",border="#FF4FA3",caption="input",caption_pos="Top",caption_size=10),
        L("FC","Hidden 1",5,5,"#FF8CB2",border="#FF8CB2",caption_pos="None"),
        L("FC","Hidden 2",5,5,"#78F5FF",border="#78F5FF",caption_pos="None"),
        L("FC","Hidden 3",5,5,"#86A8FF",border="#86A8FF",caption_pos="None"),
        L("Output","Output",3,3,"#8F85FF",border="#8F85FF",caption_pos="None"),
    ]
    p.groups = [G("hidden layer 1",1,1,color="#E4A4BF",size=10,yoff=.63), G("hidden layer 2",2,2,color="#C2FCFF",size=10,yoff=.63), G("hidden layer 3",3,3,color="#8EA8FF",size=10,yoff=.63), G("output layer",4,4,color="#8F85FF",size=10,yoff=.63)]
    p.show_output_arrows = True; p.external_arrow_style = "Arrows"; p.external_arrow_color = "#8F85FF"; p.external_arrow_width = 1.2; p.external_arrow_length = 0.55
    return p

def sample_11_rainbow_dense_tapered() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=11.0, figure_height=10.6,
                    node_radius=0.18, layer_spacing=1.20, vertical_spacing=0.62,
                    connection_color="#444444", connection_color_mode="Random palette",
                    connection_palette="#FF00AA, #00D6FF, #00B934, #3C47FF, #D88A00, #9B27FF, #222222",
                    connection_alpha=0.82, connection_width=0.9, connection_mode="Dense",
                    connection_style="Lines", connection_geometry="Straight", line_gap=0.03,
                    show_nominal_size=False, show_activation=False, margins=0.42,
                    random_width_enabled=True, random_width_probability=1.0, random_width_min_factor=1.8, random_width_max_factor=5.2, random_seed=31)
    p.layers = [
        L("Input","L1",8,8,"#000000",border="#000000",caption_pos="None"),
        L("FC","L2",6,6,"#000000",border="#000000",caption_pos="None"),
        L("FC","L3",5,5,"#000000",border="#000000",caption_pos="None"),
        L("FC","L4",4,4,"#000000",border="#000000",caption_pos="None"),
        L("FC","L5",2,2,"#000000",border="#000000",caption_pos="None"),
        L("Output","L6",1,1,"#000000",border="#000000",caption_pos="None"),
    ]
    return p

def sample_12_curved_magenta_cyan_flow() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=9.2, figure_height=5.4,
                    node_radius=0.06, layer_spacing=0.96, vertical_spacing=0.28,
                    connection_color="#20C6D0", connection_color_mode="Palette cycle",
                    connection_palette="#C050C8, #A0A0A0, #20C6D0",
                    connection_alpha=0.78, connection_width=0.95, connection_mode="Dense",
                    connection_style="Lines", connection_geometry="Curved-horizontal",
                    connection_end_mode="Outside junctions", junction_offset=0.0, curve_strength=0.52,
                    line_gap=0.04, show_nominal_size=False, show_activation=False, margins=0.45,
                    random_width_enabled=True, random_width_probability=1.0, random_width_min_factor=1.5, random_width_max_factor=3.8, random_seed=33)
    p.layers = [L("Input","Input",5,5,"#000000",border="#000000",scale=0.32,caption_pos="None"), L("FC","H1",7,7,"#000000",border="#000000",scale=0.28,caption_pos="None"), L("FC","H2",7,7,"#000000",border="#000000",scale=0.28,caption_pos="None"), L("Output","Output",1,1,"#000000",border="#000000",scale=0.35,caption_pos="None")]
    return p

def sample_13_highlighted_pathways() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=11.1, figure_height=7.2,
                    node_radius=0.16, layer_spacing=1.45, vertical_spacing=0.42,
                    connection_color="#B8B8B8", connection_alpha=0.55, connection_width=0.72,
                    connection_mode="Dense", connection_style="Lines", line_gap=0.045,
                    show_nominal_size=False, show_activation=False, margins=0.48)
    p.layers = [
        L("Input","Input",9,9,"#FFFFFF",border="#2C2C2C",border_width=2.0,caption_pos="None"),
        L("FC","Hidden 1",8,8,"#FFFFFF",border="#2C2C2C",border_width=2.0,caption_pos="None"),
        L("FC","Hidden 2",8,8,"#FFFFFF",border="#2C2C2C",border_width=2.0,caption_pos="None"),
        L("Output","Output",6,6,"#FFFFFF",border="#2C2C2C",border_width=2.0,caption_pos="None"),
    ]
    p.pair_styles = [PS(0,1,color="#C8C8C8",width=0.72), PS(1,2,color="#C8C8C8",width=0.72), PS(2,3,color="#C8C8C8",width=0.72)]
    p.highlight_paths = [
        {"name": "Path 1", "nodes": [2, 3, 4, 1], "color": "#F03A3A", "alpha": 0.96, "width_factor": 2.6, "highlight_nodes": True},
        {"name": "Path 2", "nodes": [6, 1, 6, 4], "color": "#F03A3A", "alpha": 0.96, "width_factor": 2.6, "highlight_nodes": True},
    ]
    return p

def sample_14_hybrid_sequential_dense() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=9.8, figure_height=4.9,
                    node_radius=0.19, layer_spacing=1.08, vertical_spacing=0.52,
                    connection_color="#111111", connection_alpha=0.9, connection_width=0.9,
                    connection_mode="Adjacent-only", connection_style="Lines", line_gap=0.05,
                    show_nominal_size=False, show_activation=False, margins=0.42)
    p.layers = [L("Input","Input",5,5,"#F5C623",border="#F5C623",caption_pos="None"), L("FC","Seq 1",5,5,"#E99AEF",border="#E99AEF",caption_pos="None"), L("FC","Seq 2",5,5,"#E99AEF",border="#000000",border_width=1.6,caption_pos="None"), L("FC","Bridge",2,2,"#E99AEF",border="#000000",border_width=1.6,caption_pos="None"), L("FC","Dense 1",4,4,"#67D31B",border="#67D31B",caption_pos="None"), L("FC","Dense 2",4,4,"#67D31B",border="#67D31B",caption_pos="None"), L("Output","Output",3,3,"#FF6D17",border="#FF6D17",caption_pos="None")]
    p.pair_styles = [PS(0,1,mode="Adjacent-only",color="#111111"), PS(1,2,mode="Adjacent-only",color="#111111"), PS(2,3,mode="Adjacent-only",color="#111111"), PS(3,4,mode="Adjacent-only",color="#111111"), PS(4,5,mode="Dense",color="#111111"), PS(5,6,mode="Dense",color="#111111")]
    return p

def sample_15_symbolic_dense_labels() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=14.6, figure_height=8.0,
                    node_radius=0.15, layer_spacing=3.0, vertical_spacing=1.0,
                    connection_color="#111111", connection_alpha=0.9, connection_width=0.8,
                    connection_mode="Dense", connection_style="Lines", line_gap=0.04, arrow_gap=0.07,
                    show_nominal_size=False, show_activation=False, margins=0.50)
    p.layers = [
        L("Input","I",10,10,"#20D0C0",border="#111111",texts=[rf"$i_{{{k}}}$" for k in range(1,11)],text_size=15,caption_pos="None"),
        L("FC","H1",6,6,"#F2A100",border="#111111",texts=[rf"$h_{{{k}}}^{{(1)}}$" for k in range(1,7)],text_size=14,caption_pos="None"),
        L("FC","H2",5,5,"#F2A100",border="#111111",texts=[rf"$h_{{{k}}}^{{(2)}}$" for k in range(1,6)],text_size=14,caption_pos="None"),
        L("FC","H3",7,7,"#F2A100",border="#111111",texts=[rf"$h_{{{k}}}^{{(3)}}$" for k in range(1,8)],text_size=14,caption_pos="None"),
        L("Output","O",11,11,"#2DB6FF",border="#111111",texts=[rf"$o_{{{k}}}$" for k in range(1,12)],text_size=14,caption_pos="None"),
    ]
    p.groups = []
    p.show_input_arrows = True; p.show_output_arrows = True; p.external_arrow_style = "Arrows"; p.external_arrow_length = 0.35; p.external_arrow_color = "#111111"
    return p

def sample_16_io_boxes_offsets() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=13.5, figure_height=7.0,
                    node_radius=0.13, layer_spacing=1.56, vertical_spacing=0.42,
                    connection_color="#6C7AA8", connection_alpha=0.62, connection_width=0.72,
                    connection_mode="Sampled", connection_style="Arrows", line_gap=0.04, arrow_gap=0.07,
                    show_nominal_size=False, show_activation=True, margins=0.42, layout_mode="Semi-Automatic",
                    show_input_items=True, show_output_items=True, input_box_width=1.65, output_box_width=1.55,
                    input_items=[r"$x_1$", r"$x_2$", r"$x_3$", r"$x_4$", r"$x_5$"],
                    output_items=[r"$\hat{y}_1$", r"$\hat{y}_2$", r"$\hat{y}_3$"],
                    input_group_title="Input features", output_group_title="Predictions")
    p.layers = [
        L("Input","Input",5,5,"#FFFFFF",border="#2F7E39",border_width=1.8,caption="Input items",caption_pos="Top"),
        L("FC","Encoder",32,6,"#DDE7FF",border="#4A67C5",activation="ReLU",caption="Feature encoder",xoff=0.0,yoff=0.18),
        L("Activation","Act",32,4,"#FFF3BF",border="#B18A00",shape="activation",activation="tanh",caption="Nonlinear block",xoff=0.0,yoff=-0.12),
        L("FC","Head",16,5,"#E3DCF8",border="#6D57B5",activation="ReLU",caption="Prediction head",xoff=0.05,yoff=0.08),
        L("Output","Output",3,3,"#FFF0F0",border="#B54C4C",activation="Linear",caption="Outputs",caption_pos="Top"),
    ]
    p.groups = [G("Encoder stack",1,3,style="Pill",color="#1B2D5C",fill="#EEF3FF",border="#CBD6F8",size=14,yoff=.80,height=.28,padding=.30)]
    p.pair_styles = [
        PS(0,1,mode="Dense",render="Arrows",color="#91A1D7",alpha=.58,width=.68,head=7,gap=.05),
        PS(1,2,mode="Adjacent-only",render="Arrows",color="#A17DD6",alpha=.80,width=.95,head=8,gap=.08),
        PS(2,3,mode="Dense",render="Arrows",color="#8A76C9",alpha=.72,width=.72,head=8,gap=.07),
        PS(3,4,mode="Dense",render="Arrows",color="#C77C7C",alpha=.72,width=.78,head=8,gap=.07),
    ]
    p.annotations = []
    return p

def sample_17_random_palette_flow() -> ProjectSpec:
    p = ProjectSpec(background="#101217", figure_width=13.2, figure_height=6.2,
                    node_radius=0.12, layer_spacing=1.34, vertical_spacing=0.38,
                    connection_color="#FF4FA3", connection_color_mode="Random palette",
                    connection_palette="#FF4FA3, #66E5FF, #7D8CFF, #7FE07F, #F7C948, #FF8A3D",
                    connection_alpha=0.85, connection_width=0.88, connection_mode="Dense",
                    connection_style="Lines", connection_geometry="Curved-horizontal",
                    connection_end_mode="Outside junctions", junction_offset=.08, curve_strength=.44,
                    random_width_enabled=True, random_width_probability=.23, random_width_min_factor=1.4, random_width_max_factor=3.3,
                    show_nominal_size=False, show_activation=False, margins=.48,
                    show_input_arrows=True, show_output_arrows=True, external_arrow_style="Arrows", external_arrow_color="#8A95C2", external_arrow_length=.42)
    p.layers = [
        L("Input","Input",6,6,"#FF4FA3",border="#FF4FA3",caption_pos="None"),
        L("FC","H1",6,6,"#F28BB8",border="#F28BB8",caption_pos="None"),
        L("FC","H2",5,5,"#66E5FF",border="#66E5FF",caption_pos="None"),
        L("FC","H3",5,5,"#7D8CFF",border="#7D8CFF",caption_pos="None"),
        L("Output","Output",3,3,"#A8A5FF",border="#A8A5FF",caption_pos="None"),
    ]
    p.groups = [G("Random palette connections",0,4,style="Text",color="#CFD5FF",size=16,yoff=.85)]
    return p


def sample_18_multi_path_focus() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=12.4, figure_height=7.0,
                    node_radius=0.15, layer_spacing=1.38, vertical_spacing=0.39,
                    connection_color="#D0D0D0", connection_alpha=0.55, connection_width=0.66,
                    connection_mode="Dense", connection_style="Lines", show_nominal_size=False, show_activation=False, margins=.45)
    p.layers = [
        L("Input","Input",8,8,"#FFFFFF",border="#2C2C2C",border_width=1.8,caption="Input",caption_pos="Top"),
        L("FC","Hidden 1",8,8,"#FFFFFF",border="#2C2C2C",border_width=1.8,caption="Hidden 1",caption_pos="Top"),
        L("FC","Hidden 2",8,8,"#FFFFFF",border="#2C2C2C",border_width=1.8,caption="Hidden 2",caption_pos="Top"),
        L("FC","Hidden 3",7,7,"#FFFFFF",border="#2C2C2C",border_width=1.8,caption="Hidden 3",caption_pos="Top"),
        L("Output","Output",4,4,"#FFFFFF",border="#2C2C2C",border_width=1.8,caption="Output",caption_pos="Top"),
    ]
    p.highlight_paths = [
        {"name": "Red path", "nodes": [1, 5, 2, 4, 0], "color": "#E53935", "alpha": .95, "width_factor": 2.8, "highlight_nodes": True},
        {"name": "Blue path", "nodes": [6, 1, 6, 2, 2], "color": "#1E88E5", "alpha": .95, "width_factor": 2.6, "highlight_nodes": True},
        {"name": "Green path", "nodes": [3, 3, 1, 5, 3], "color": "#43A047", "alpha": .95, "width_factor": 2.6, "highlight_nodes": True},
    ]
    p.annotations = [A("Text", text="Selected node-to-node paths are highlighted", x=2.8, y=-2.05, font_size=11, color="#666666")]
    return p


def sample_19_annotated_workflow_card() -> ProjectSpec:
    p = ProjectSpec(background="#F7F7FA", figure_width=14.0, figure_height=7.6,
                    node_radius=.13, layer_spacing=1.45, vertical_spacing=.42,
                    connection_color="#9098B6", connection_alpha=.62, connection_width=.72,
                    connection_mode="Sampled", connection_style="Arrows", connection_geometry="Straight",
                    connection_end_mode="Outside junctions", junction_offset=.07, arrow_gap=.07,
                    show_nominal_size=False, show_activation=True, margins=.50, card_enabled=True, card_shadow=True,
                    card_fill="#FFFFFF", card_border="#E3E5EF")
    p.layers = [
        L("Input","Input",4,4,"#D7F0D8",border="#4F9A58",texts=["raw","feat","meta","img"],text_size=10,caption="Multi-modal input",caption_pos="Top"),
        L("FC","Fusion",12,5,"#E9E5FF",border="#7B67C6",activation="ReLU",caption="Fusion layer",caption_pos="Top"),
        L("Activation","Gate",12,4,"#FFF6CC",border="#C49D12",shape="activation",activation="sigmoid",caption="Gate",caption_pos="Top"),
        L("FC","Decision",8,4,"#E4F0FF",border="#4B7ECF",activation="ReLU",caption="Decision block",caption_pos="Top"),
        L("Output","Output",2,2,"#FFE2E2",border="#C56C6C",texts=["accept","reject"],text_size=10,caption="Decision",caption_pos="Top"),
    ]
    p.groups = [
        G("Inputs",0,0,style="Box",color="#38503D",fill="#F2FBF2",border="#CFE8D0",size=13,yoff=.76,height=.26),
        G("Learned representation",1,3,style="Pill",color="#2F3966",fill="#EEF1FF",border="#D8DDF8",size=14,yoff=.76,height=.28,padding=.34),
    ]
    generic = str((BASE_DIR / "assets" / "generic_input_image.png").resolve())
    p.annotations = [
        A("Image", image_path=generic, x=-1.85, y=.15, image_scale=.42, zorder=18),
        A("Arrow", x=-1.1, y=.18, x2=-.18, y2=.18, color="#7E88AA", line_width=1.0, arrow_head_size=8),
        A("Rounded", x=1.55, y=-1.92, width=2.0, height=.44, fill="#F8F9FE", border_color="#D8DDF1", line_width=.9),
        A("Text", text="Image, text, and boxed annotations can be mixed", x=1.55, y=-1.92, font_size=11, color="#5D6784"),
        A("Divider", x=3.65, y=-1.55, x2=3.65, y2=1.7, color="#D7DAE6", line_width=.9, line_style="Dashed", zorder=1),
        A("Pill", x=4.65, y=1.63, width=1.45, height=.32, fill="#EDF2FF", border_color="#D5DEFB", color="#46588E", text="workflow", font_size=11, bold=True),
    ]
    return p


def sample_20_staggered_hybrid_showcase() -> ProjectSpec:
    p = ProjectSpec(background="#FFFFFF", figure_width=13.6, figure_height=6.8,
                    node_radius=.14, layer_spacing=1.22, vertical_spacing=.39,
                    connection_color="#333333", connection_alpha=.88, connection_width=.80,
                    connection_mode="Sampled", connection_style="Lines", connection_geometry="Curved-horizontal",
                    connection_end_mode="Outside junctions", junction_offset=.09, curve_strength=.36,
                    show_nominal_size=False, show_activation=False, margins=.48, layout_mode="Semi-Automatic")
    p.layers = [
        L("Input","Input",6,6,"#F5C623",border="#E0AE00",shape="circle",caption="Input",caption_pos="Top",xoff=0.0,yoff=.0),
        L("FC","Seq A",6,6,"#E99AEF",border="#E99AEF",shape="circle",caption="Sequential",caption_pos="Top",xoff=.00,yoff=.0),
        L("FC","Seq B",5,5,"#E99AEF",border="#000000",border_width=1.4,shape="rounded",caption="Rounded",caption_pos="Top",xoff=.10,yoff=.18),
        L("FC","Bridge",4,4,"#E99AEF",border="#000000",border_width=1.4,shape="ellipse",caption="Bridge",caption_pos="Top",xoff=.12,yoff=-.18),
        L("FC","Dense A",4,4,"#67D31B",border="#67D31B",shape="square",caption="Dense A",caption_pos="Top",xoff=.0,yoff=.0),
        L("FC","Dense B",4,4,"#67D31B",border="#67D31B",shape="diamond",caption="Dense B",caption_pos="Top",xoff=.08,yoff=.12),
        L("Output","Output",3,3,"#FF6D17",border="#FF6D17",shape="hexagon",caption="Output",caption_pos="Top",xoff=.0,yoff=.0),
    ]
    p.pair_styles = [
        PS(0,1,mode="Adjacent-only",color="#111111",width=.9),
        PS(1,2,mode="Adjacent-only",color="#111111",width=.9),
        PS(2,3,mode="Adjacent-only",color="#111111",width=.9),
        PS(3,4,mode="Sampled",color="#888888",alpha=.55,width=.7),
        PS(4,5,mode="Dense",color="#111111",width=.85),
        PS(5,6,mode="Dense",color="#111111",width=.85),
    ]
    p.annotations = [A("Text", text="Semi-automatic offsets + mixed shapes + hybrid connectivity", x=3.7, y=-1.88, font_size=11, color="#666666")]
    return p


def blank_custom() -> ProjectSpec:
    p=ProjectSpec(connection_geometry="Curved-horizontal", connection_end_mode="Outside junctions", junction_offset=.06, curve_strength=.32)
    p.layers=[L("Input","Input",3,3,"#63B36C",caption="Input"),
              L("FC","Hidden",8,5,"#4C78C2",activation="ReLU",caption="Hidden"),
              L("Output","Output",1,1,"#F29A65",activation="Linear",caption="Output")]
    return p


def sample_25_cascade_fan_out_focus() -> ProjectSpec:
    p = sample_18_multi_path_focus()
    p.title = ""
    p.subtitle = ""
    p.annotations = []
    p.card_enabled = False
    p.highlight_paths = [{
        "mode": "cascade", "start_layer": 1, "source_node": 2, "pivot_node": 3,
        "color": "#D81B60", "alpha": 0.95, "width_factor": 2.8,
        "render": "Arrows", "arrow_head_size": 10.0,
        "highlight_nodes": True, "node_fill": "#FFFFFF", "node_border": "#D81B60",
        "all_downstream": True
    }]
    return p

def sample_26_surface_gap_showcase() -> ProjectSpec:
    p = sample_03_external_arrows()
    p.title = "Surface-gap showcase"
    p.subtitle = "Gaps measured from the visible node surface"
    p.connection_end_mode = "Trim at cells"
    p.connection_geometry = "Straight"
    p.line_gap = 0.060
    p.arrow_gap = 0.080
    p.show_input_arrows = False
    p.show_output_arrows = False
    p.card_enabled = False
    return p

def sample_27_outside_junction_curved_flow() -> ProjectSpec:
    p = sample_03_external_arrows()
    p.title = "Outside-junction curved flow"
    p.subtitle = "Visible outside junctions with curved-horizontal links"
    p.connection_end_mode = "Outside junctions"
    p.connection_geometry = "Curved-horizontal"
    p.connection_style = "Arrows"
    p.junction_offset = 0.10
    p.line_gap = 0.040
    p.arrow_gap = 0.060
    p.card_enabled = False
    return p

PRESETS = {
    "Sample 01 — Formula-labelled MLP": sample_01_formula_network,
    "Sample 02 — Activation-block network": sample_02_activation_blocks,
    "Sample 03 — Curved external-arrow network": sample_03_external_arrows,
    "Sample 05 — Paper-style dividers & labels": sample_06_paper_dividers,
    "Sample 06 — Mixed node shapes": sample_07_mixed_shapes,
    "Sample 07 — Generic image classifier": sample_08_dog_classifier,
    "Sample 08 — Monochrome dense labels": sample_09_monochrome_dense_labels,
    "Sample 09 — Neon gradient dark network": sample_10_neon_gradient_dark,
    "Sample 10 — Rainbow tapered dense network": sample_11_rainbow_dense_tapered,
    "Sample 11 — Curved magenta-cyan flow": sample_12_curved_magenta_cyan_flow,
    "Sample 12 — Highlighted pathways network": sample_13_highlighted_pathways,
    "Sample 13 — Hybrid sequential-dense network": sample_14_hybrid_sequential_dense,
    "Sample 14 — Symbolic dense labelled network": sample_15_symbolic_dense_labels,
    "Sample 15 — Input/output boxes with offsets": sample_16_io_boxes_offsets,
    "Sample 16 — Random palette flow": sample_17_random_palette_flow,
    "Sample 17 — Multi-path focus network": sample_18_multi_path_focus,
    "Sample 24 — Classic LeNet-style CNN": classic_lenet,
    "Sample 25 — Cascade fan-out focus": sample_25_cascade_fan_out_focus,
    "Blank / Custom": blank_custom,
}


# -----------------------------------------------------------------------------
# Dialogs
# -----------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, scrollbar_style: Optional[str]=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        sb_kwargs = {"style": scrollbar_style} if scrollbar_style else {}
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview, **sb_kwargs)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window_id = self.canvas.create_window((0,0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window_id, width=e.width))
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        # Pack the scrollbar first so it always retains visible width in narrow panes.
        self.scrollbar.pack(side="right", fill="y", padx=(4,0))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")

    def _is_descendant(self, widget):
        while widget is not None:
            if widget == self:
                return True
            try:
                parent_name = widget.winfo_parent()
                widget = widget._nametowidget(parent_name) if parent_name else None
            except Exception:
                return False
        return False

    def _wheel(self, e):
        try:
            widget = self.winfo_containing(e.x_root, e.y_root)
            if widget is not None and self._is_descendant(widget):
                steps = int(-1 * (e.delta / 120)) if e.delta else 0
                if steps:
                    self.canvas.yview_scroll(steps, "units")
        except Exception:
            pass


class LayerDialog(tk.Toplevel):
    def __init__(self, parent, layer: Optional[LayerSpec]=None):
        super().__init__(parent)
        self.title("Layer settings")
        self.geometry("690x820")
        self.minsize(620,680)
        self.transient(parent); self.grab_set()
        self.result=None
        self.layer=copy.deepcopy(layer or LayerSpec())
        vals=asdict(self.layer)
        self.vars={k:tk.StringVar(value=str(v)) for k,v in vals.items() if k not in ("node_texts","show_ellipsis","caption_bold")}
        self.show_ellipsis=tk.BooleanVar(value=self.layer.show_ellipsis)
        self.caption_bold=tk.BooleanVar(value=self.layer.caption_bold)

        outer=ScrollableFrame(self); outer.pack(fill="both",expand=True,padx=10,pady=10); f=outer.inner
        f.columnconfigure(1,weight=1)
        row=0
        def combo(label,key,values):
            nonlocal row
            ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=3)
            ttk.Combobox(f,textvariable=self.vars[key],values=values,state="readonly",width=28).grid(row=row,column=1,sticky="ew",pady=3); row+=1
        def entry(label,key):
            nonlocal row
            ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=3)
            ttk.Entry(f,textvariable=self.vars[key]).grid(row=row,column=1,sticky="ew",pady=3); row+=1
        def colorrow(label,key):
            nonlocal row
            ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=3)
            q=ttk.Frame(f); q.grid(row=row,column=1,sticky="ew",pady=3); q.columnconfigure(0,weight=1)
            ttk.Entry(q,textvariable=self.vars[key]).grid(row=0,column=0,sticky="ew")
            ttk.Button(q,text="Pick",command=lambda k=key:self._pick(k)).grid(row=0,column=1,padx=(5,0)); row+=1

        combo("Layer type","kind",["Input","Conv","Pool","FC","Activation","Output","Custom"])
        entry("Layer name","name"); entry("Nominal size / units","size"); entry("Visible nodes / blocks","visible_nodes")
        entry("Activation / operation","activation")
        combo("Node/block shape","shape",["circle","ellipse","square","rounded","diamond","triangle","hexagon","pentagon","activation"])
        colorrow("Fill color","color"); entry("Fill alpha (0–1)","fill_alpha")
        colorrow("Border color","border_color"); entry("Border width","border_width"); entry("Node size scale","node_scale")
        entry("Caption note","note")
        ttk.Checkbutton(f,text="Show ellipsis when nominal size > visible nodes",variable=self.show_ellipsis).grid(row=row,column=0,columnspan=2,sticky="w",pady=5); row+=1

        ttk.Separator(f).grid(row=row,column=0,columnspan=2,sticky="ew",pady=8); row+=1
        ttk.Label(f,text="Text inside nodes / blocks",font=("Segoe UI",9,"bold")).grid(row=row,column=0,columnspan=2,sticky="w"); row+=1
        ttk.Label(f,text="One line per visible node. MathText is supported.",foreground="#666").grid(row=row,column=0,columnspan=2,sticky="w"); row+=1
        self.node_text=ScrolledText(f,height=7,wrap="word",undo=True); self.node_text.grid(row=row,column=0,columnspan=2,sticky="nsew",pady=4); row+=1
        self.node_text.insert("1.0","\n".join(self.layer.node_texts))
        fonts=sorted(set(tkfont.families()))
        combo("Node text font","node_text_font_family",fonts); entry("Node text size","node_text_font_size"); colorrow("Node text color","node_text_color")

        ttk.Separator(f).grid(row=row,column=0,columnspan=2,sticky="ew",pady=8); row+=1
        ttk.Label(f,text="Layer caption",font=("Segoe UI",9,"bold")).grid(row=row,column=0,columnspan=2,sticky="w"); row+=1
        entry("Caption text (blank = layer name)","caption_text")
        combo("Caption position","caption_position",["Top","Bottom","None"])
        colorrow("Caption color","caption_color"); entry("Caption font size","caption_font_size")
        ttk.Checkbutton(f,text="Bold caption",variable=self.caption_bold).grid(row=row,column=0,columnspan=2,sticky="w",pady=4); row+=1

        ttk.Separator(f).grid(row=row,column=0,columnspan=2,sticky="ew",pady=8); row+=1
        ttk.Label(f,text="Semi-manual layout offset",font=("Segoe UI",9,"bold")).grid(row=row,column=0,columnspan=2,sticky="w"); row+=1
        entry("X offset","x_offset"); entry("Y offset","y_offset")
        b=ttk.Frame(f); b.grid(row=row,column=0,columnspan=2,sticky="e",pady=10)
        ttk.Button(b,text="Cancel",command=self.destroy).pack(side="right",padx=(6,0)); ttk.Button(b,text="OK",command=self.accept).pack(side="right")
        self.bind("<Escape>",lambda e:self.destroy())

    def _pick(self,key):
        c=colorchooser.askcolor(self.vars[key].get(),parent=self)[1]
        if c:self.vars[key].set(c)
    def accept(self):
        try:
            self.result=LayerSpec(
                kind=self.vars["kind"].get().strip() or "Custom", name=self.vars["name"].get().strip() or "Layer",
                size=max(0,int(float(self.vars["size"].get()))), activation=self.vars["activation"].get().strip(),
                visible_nodes=max(1,int(float(self.vars["visible_nodes"].get()))), color=self.vars["color"].get().strip() or "#CCCCCC",
                shape=self.vars["shape"].get().strip() or "circle", note=self.vars["note"].get().strip(),
                show_ellipsis=bool(self.show_ellipsis.get()),
                node_texts=[x.strip() for x in self.node_text.get("1.0","end").splitlines()],
                node_text_font_size=max(1,float(self.vars["node_text_font_size"].get())),
                node_text_color=self.vars["node_text_color"].get().strip() or "#111111",
                node_text_font_family=self.vars["node_text_font_family"].get().strip() or "Times New Roman",
                border_color=self.vars["border_color"].get().strip() or "#202020",
                border_width=max(0,float(self.vars["border_width"].get())), fill_alpha=min(1,max(0,float(self.vars["fill_alpha"].get()))),
                node_scale=max(.05,float(self.vars["node_scale"].get())), caption_text=self.vars["caption_text"].get().strip(),
                caption_color=self.vars["caption_color"].get().strip() or "#111111",
                caption_font_size=max(1,float(self.vars["caption_font_size"].get())), caption_position=self.vars["caption_position"].get(),
                caption_bold=bool(self.caption_bold.get()), x_offset=float(self.vars["x_offset"].get()), y_offset=float(self.vars["y_offset"].get()),
            )
        except Exception as exc:
            messagebox.showerror("Invalid layer",str(exc),parent=self); return
        self.destroy()


class AnnotationDialog(tk.Toplevel):
    def __init__(self,parent,ann:Optional[AnnotationSpec]=None,force_kind:Optional[str]=None):
        super().__init__(parent); self.title("Annotation / object settings"); self.geometry("680x780"); self.minsize(600,650)
        self.transient(parent); self.grab_set(); self.result=None
        self.ann=copy.deepcopy(ann or AnnotationSpec(kind=force_kind or "Text"))
        if force_kind:self.ann.kind=force_kind
        v=asdict(self.ann)
        self.vars={k:tk.StringVar(value=str(val)) for k,val in v.items() if k not in ("bold","italic","preserve_alpha","locked")}
        self.bold=tk.BooleanVar(value=self.ann.bold); self.italic=tk.BooleanVar(value=self.ann.italic); self.locked=tk.BooleanVar(value=self.ann.locked)
        outer=ScrollableFrame(self); outer.pack(fill="both",expand=True,padx=10,pady=10); f=outer.inner; f.columnconfigure(1,weight=1)
        row=0
        def combo(label,key,vals):
            nonlocal row; ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=3)
            ttk.Combobox(f,textvariable=self.vars[key],values=vals,state="readonly",width=28).grid(row=row,column=1,sticky="ew",pady=3); row+=1
        def entry(label,key):
            nonlocal row; ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=3); ttk.Entry(f,textvariable=self.vars[key]).grid(row=row,column=1,sticky="ew",pady=3); row+=1
        def colorrow(label,key):
            nonlocal row; ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=3); q=ttk.Frame(f); q.grid(row=row,column=1,sticky="ew",pady=3); q.columnconfigure(0,weight=1)
            ttk.Entry(q,textvariable=self.vars[key]).grid(row=0,column=0,sticky="ew"); ttk.Button(q,text="Pick",command=lambda k=key:self._pick(k)).grid(row=0,column=1,padx=(5,0)); row+=1
        combo("Object kind","kind",["Text","Image","Arrow","Line","Rectangle","Rounded","Pill","Circle","Divider"])
        ttk.Label(f,text="Text").grid(row=row,column=0,sticky="nw",pady=3); self.text=ScrolledText(f,height=5,wrap="word"); self.text.grid(row=row,column=1,sticky="ew",pady=3); self.text.insert("1.0",self.ann.text); row+=1
        entry("X","x"); entry("Y","y"); entry("End X (line/arrow)","x2"); entry("End Y (line/arrow)","y2")
        entry("Width","width"); entry("Height","height")
        colorrow("Fill","fill"); colorrow("Border color","border_color"); colorrow("Text/line color","color")
        entry("Alpha (0–1)","alpha"); entry("Line/border width","line_width")
        combo("Line style","line_style",["Solid","Dashed","Dotted"])
        fonts=sorted(set(tkfont.families())); combo("Font family","font_family",fonts); entry("Font size","font_size")
        combo("Horizontal align","ha",["left","center","right"]); combo("Vertical align","va",["top","center","bottom","baseline"])
        entry("Rotation (deg)","rotation"); entry("Z-order","zorder"); entry("Arrow head size","arrow_head_size")
        ttk.Checkbutton(f,text="Bold",variable=self.bold).grid(row=row,column=0,sticky="w"); ttk.Checkbutton(f,text="Italic",variable=self.italic).grid(row=row,column=1,sticky="w"); row+=1
        ttk.Checkbutton(f,text="Lock object against canvas dragging",variable=self.locked).grid(row=row,column=0,columnspan=2,sticky="w",pady=3); row+=1
        ttk.Separator(f).grid(row=row,column=0,columnspan=2,sticky="ew",pady=8); row+=1
        ttk.Label(f,text="Image file").grid(row=row,column=0,sticky="w"); q=ttk.Frame(f); q.grid(row=row,column=1,sticky="ew"); q.columnconfigure(0,weight=1)
        ttk.Entry(q,textvariable=self.vars["image_path"]).grid(row=0,column=0,sticky="ew"); ttk.Button(q,text="Browse",command=self._browse).grid(row=0,column=1,padx=(5,0)); row+=1
        entry("Image scale","image_scale")
        b=ttk.Frame(f); b.grid(row=row,column=0,columnspan=2,sticky="e",pady=10); ttk.Button(b,text="Cancel",command=self.destroy).pack(side="right",padx=(6,0)); ttk.Button(b,text="OK",command=self.accept).pack(side="right")
    def _pick(self,key):
        c=colorchooser.askcolor(self.vars[key].get(),parent=self)[1]
        if c:self.vars[key].set(c)
    def _browse(self):
        fn=filedialog.askopenfilename(parent=self,title="Choose image",filetypes=[("Images","*.png *.jpg *.jpeg *.webp *.svg"),("All files","*.*")])
        if fn:self.vars["image_path"].set(fn)
    def accept(self):
        try:
            self.result=AnnotationSpec(
                kind=self.vars["kind"].get(), text=self.text.get("1.0","end").rstrip("\n"),
                x=float(self.vars["x"].get()),y=float(self.vars["y"].get()),x2=float(self.vars["x2"].get()),y2=float(self.vars["y2"].get()),
                width=max(.001,float(self.vars["width"].get())),height=max(.001,float(self.vars["height"].get())),
                fill=self.vars["fill"].get().strip() or "#FFFFFF",border_color=self.vars["border_color"].get().strip() or "#222222",
                color=self.vars["color"].get().strip() or "#111111",alpha=min(1,max(0,float(self.vars["alpha"].get()))),
                line_width=max(0,float(self.vars["line_width"].get())),line_style=self.vars["line_style"].get(),
                font_family=self.vars["font_family"].get().strip() or "Times New Roman",font_size=max(1,float(self.vars["font_size"].get())),
                bold=bool(self.bold.get()),italic=bool(self.italic.get()),ha=self.vars["ha"].get(),va=self.vars["va"].get(),rotation=float(self.vars["rotation"].get()),
                image_path=self.vars["image_path"].get().strip(),image_scale=max(.01,float(self.vars["image_scale"].get())),
                zorder=float(self.vars["zorder"].get()),locked=bool(self.locked.get()),arrow_head_size=max(1,float(self.vars["arrow_head_size"].get()))
            )
        except Exception as exc:
            messagebox.showerror("Invalid object",str(exc),parent=self); return
        self.destroy()


class GroupDialog(tk.Toplevel):
    def __init__(self,parent,nlayers:int,group:Optional[GroupSpec]=None):
        super().__init__(parent); self.title("Layer group header"); self.geometry("500x470"); self.transient(parent); self.grab_set(); self.result=None
        g=copy.deepcopy(group or GroupSpec()); self.vars={k:tk.StringVar(value=str(v)) for k,v in asdict(g).items() if k!="bold"}; self.bold=tk.BooleanVar(value=g.bold)
        f=ttk.Frame(self,padding=12); f.pack(fill="both",expand=True); f.columnconfigure(1,weight=1); row=0
        def e(label,key):
            nonlocal row; ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=4); ttk.Entry(f,textvariable=self.vars[key]).grid(row=row,column=1,sticky="ew",pady=4); row+=1
        def c(label,key,vals):
            nonlocal row; ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=4); ttk.Combobox(f,textvariable=self.vars[key],values=vals,state="readonly").grid(row=row,column=1,sticky="ew",pady=4); row+=1
        e("Title","title"); e(f"Start layer index (0–{max(0,nlayers-1)})","start_layer"); e("End layer index","end_layer"); c("Style","style",["Text","Pill","Box"])
        e("Text color","color"); e("Fill color","fill"); e("Border color (blank = none)","border_color"); e("Font size","font_size"); e("Y offset","y_offset"); e("Header height","height"); e("Horizontal padding","padding")
        ttk.Checkbutton(f,text="Bold",variable=self.bold).grid(row=row,column=0,columnspan=2,sticky="w",pady=4); row+=1
        b=ttk.Frame(f); b.grid(row=row,column=0,columnspan=2,sticky="e",pady=10); ttk.Button(b,text="Cancel",command=self.destroy).pack(side="right",padx=(6,0)); ttk.Button(b,text="OK",command=self.accept).pack(side="right")
    def accept(self):
        try:
            self.result=GroupSpec(title=self.vars["title"].get(),start_layer=int(self.vars["start_layer"].get()),end_layer=int(self.vars["end_layer"].get()),style=self.vars["style"].get(),
                                  color=self.vars["color"].get(),fill=self.vars["fill"].get(),border_color=self.vars["border_color"].get(),font_size=float(self.vars["font_size"].get()),bold=bool(self.bold.get()),
                                  y_offset=float(self.vars["y_offset"].get()),height=float(self.vars["height"].get()),padding=float(self.vars["padding"].get()))
        except Exception as exc: messagebox.showerror("Invalid group",str(exc),parent=self); return
        self.destroy()


class PairStyleDialog(tk.Toplevel):
    def __init__(self,parent,nlayers:int,style:Optional[PairStyleSpec]=None):
        super().__init__(parent); self.title("Layer-pair connection style"); self.geometry("540x620"); self.transient(parent); self.grab_set(); self.result=None
        s=copy.deepcopy(style or PairStyleSpec())
        d=asdict(s); self.vars={k:tk.StringVar(value=str(v)) for k,v in d.items() if k!="random_width_enabled"}; self.random_mode=tk.StringVar(value="Inherit" if s.random_width_enabled is None else ("On" if s.random_width_enabled else "Off"))
        f=ttk.Frame(self,padding=12); f.pack(fill="both",expand=True); f.columnconfigure(1,weight=1); row=0
        def e(label,key):
            nonlocal row; ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=4); ttk.Entry(f,textvariable=self.vars[key]).grid(row=row,column=1,sticky="ew",pady=4); row+=1
        def c(label,key,vals):
            nonlocal row; ttk.Label(f,text=label).grid(row=row,column=0,sticky="w",pady=4); ttk.Combobox(f,textvariable=self.vars[key],values=vals,state="readonly").grid(row=row,column=1,sticky="ew",pady=4); row+=1
        e(f"Source layer index (0–{max(0,nlayers-1)})","source_layer"); e("Target layer index","target_layer")
        c("Connection selection","mode",["Inherit","Dense","Sampled","Adjacent-only","None"]); c("Rendering","render",["Inherit","Lines","Arrows"])
        e("Color (blank = inherit)","color"); e("Alpha (-1 = inherit)","alpha"); e("Width (-1 = inherit)","width"); e("Arrow head (-1 = inherit)","arrow_head_size"); e("Gap (-1 = inherit)","gap")
        ttk.Label(f,text="Random width").grid(row=row,column=0,sticky="w",pady=4); ttk.Combobox(f,textvariable=self.random_mode,values=["Inherit","On","Off"],state="readonly").grid(row=row,column=1,sticky="ew",pady=4); row+=1
        e("Random probability (-1 = inherit)","random_width_probability"); e("Min factor (-1 = inherit)","random_width_min_factor"); e("Max factor (-1 = inherit)","random_width_max_factor"); e("Seed offset","seed_offset")
        b=ttk.Frame(f); b.grid(row=row,column=0,columnspan=2,sticky="e",pady=10); ttk.Button(b,text="Cancel",command=self.destroy).pack(side="right",padx=(6,0)); ttk.Button(b,text="OK",command=self.accept).pack(side="right")
    def accept(self):
        try:
            rm=self.random_mode.get(); ron=None if rm=="Inherit" else rm=="On"
            self.result=PairStyleSpec(source_layer=int(self.vars["source_layer"].get()),target_layer=int(self.vars["target_layer"].get()),mode=self.vars["mode"].get(),render=self.vars["render"].get(),color=self.vars["color"].get().strip(),
                                      alpha=float(self.vars["alpha"].get()),width=float(self.vars["width"].get()),arrow_head_size=float(self.vars["arrow_head_size"].get()),gap=float(self.vars["gap"].get()),random_width_enabled=ron,
                                      random_width_probability=float(self.vars["random_width_probability"].get()),random_width_min_factor=float(self.vars["random_width_min_factor"].get()),random_width_max_factor=float(self.vars["random_width_max_factor"].get()),seed_offset=int(float(self.vars["seed_offset"].get())))
        except Exception as exc: messagebox.showerror("Invalid pair style",str(exc),parent=self); return
        self.destroy()


# -----------------------------------------------------------------------------
# Renderer
# -----------------------------------------------------------------------------

class ArchitectureRenderer:
    marker_map={"circle":"o","square":"s","diamond":"D","triangle":"^","hexagon":"h","pentagon":"p"}
    def __init__(self, app=None):
        self.app=app
        self.annotation_artists:List[Tuple[Any,int]]=[]

    @staticmethod
    def centered_positions(count:int, spacing:float, max_half_span:Optional[float]=None) -> List[float]:
        count=max(1,int(count)); ys=[0.0] if count==1 else [0.5*(count-1)*spacing-i*spacing for i in range(count)]
        if max_half_span is not None and max_half_span>0 and ys and max(abs(y) for y in ys)>max_half_span:
            scale=max_half_span/max(abs(y) for y in ys); ys=[y*scale for y in ys]
        return ys

    @staticmethod
    def _sample_pairs(na:int, nb:int, max_count:int) -> List[Tuple[int,int]]:
        total=na*nb
        if total<=max_count:
            return [(k//nb,k%nb) for k in range(total)]
        # Sample evenly in flattened Cartesian-index space without constructing all pairs.
        if max_count<=1:return [(0,0)]
        idxs=sorted({min(total-1,round(k*(total-1)/(max_count-1))) for k in range(max_count)})
        return [(idx//nb,idx%nb) for idx in idxs]

    @staticmethod
    def _adjacent_pairs(na:int,nb:int)->List[Tuple[int,int]]:
        m=min(na,nb)
        if m<=1:return [(0,0)]
        return [(round(i*(na-1)/(m-1)),round(i*(nb-1)/(m-1))) for i in range(m)]

    @staticmethod
    def _linestyle(name:str):
        return {"Solid":"-","Dashed":"--","Dotted":":"}.get(name,"-")

    @staticmethod
    def _text_extent_width(items:List[str], minimum=1.45):
        if not items:return minimum
        m=max(len(str(s).replace("$","").replace("\\","")) for s in items)
        return min(3.4,max(minimum,.09*m+.72))

    def _pair_style(self,p:ProjectSpec,a:int,b:int)->PairStyleSpec:
        for s in p.pair_styles:
            if s.source_layer==a and s.target_layer==b:return s
        return PairStyleSpec(a,b)

    def _resolved_pair(self,p:ProjectSpec,s:PairStyleSpec):
        mode=p.connection_mode if s.mode=="Inherit" else s.mode
        render=p.connection_style if s.render=="Inherit" else s.render
        color=s.color or p.connection_color
        alpha=p.connection_alpha if s.alpha<0 else s.alpha
        width=p.connection_width if s.width<0 else s.width
        head=p.arrow_head_size if s.arrow_head_size<0 else s.arrow_head_size
        gap=(p.arrow_gap if render.lower()=="arrows" else p.line_gap) if s.gap<0 else s.gap
        ron=p.random_width_enabled if s.random_width_enabled is None else s.random_width_enabled
        prob=p.random_width_probability if s.random_width_probability<0 else s.random_width_probability
        minf=p.random_width_min_factor if s.random_width_min_factor<0 else s.random_width_min_factor
        maxf=p.random_width_max_factor if s.random_width_max_factor<0 else s.random_width_max_factor
        return mode,render,color,alpha,width,head,gap,ron,prob,minf,maxf
    @staticmethod
    def _parse_palette(p: ProjectSpec) -> List[str]:
        raw = getattr(p, "connection_palette", "") or ""
        vals = [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]
        return vals or [p.connection_color]

    def _pick_connection_color(self, p: ProjectSpec, base_color: str, pair_index: int, conn_index: int, rng: random.Random) -> str:
        mode = (getattr(p, "connection_color_mode", "Single") or "Single").lower()
        if mode == "single" or not base_color:
            return base_color
        palette = self._parse_palette(p)
        if len(palette) == 1:
            return palette[0]
        if mode.startswith("random"):
            return rng.choice(palette)
        return palette[(pair_index + conn_index) % len(palette)]

    @staticmethod
    def _axis_delta_px(ax, dx: float = 0.0, dy: float = 0.0) -> Tuple[float, float]:
        a = ax.transData.transform((0.0, 0.0))
        b = ax.transData.transform((dx, dy))
        return abs(float(b[0] - a[0])), abs(float(b[1] - a[1]))

    def _gap_distance_px(self, ax, gap_value: float, layer_dx: float) -> float:
        """Convert user gap to display pixels after the final transform is locked."""
        gap_value = max(0.0, float(gap_value))
        if gap_value <= 0.0:
            return 0.0
        layer_gap_px, _ = self._axis_delta_px(ax, max(abs(layer_dx), 1e-6), 0.0)
        return max(2.0, gap_value * max(24.0, layer_gap_px))

    def _node_half_extents_px(self, fig, ax, p: ProjectSpec, l: LayerSpec) -> Tuple[float, float]:
        shape = (l.shape or "circle").lower()
        if shape == "activation" or l.kind.lower() == "activation":
            w = .42 * l.node_scale; h = .38 * l.node_scale
            wx, _ = self._axis_delta_px(ax, w, 0.0)
            _, hy = self._axis_delta_px(ax, 0.0, h)
            return max(1.0, wx/2.0), max(1.0, hy/2.0)
        if shape == "rounded":
            w = .30 * l.node_scale; h = .28 * l.node_scale
            wx, _ = self._axis_delta_px(ax, w, 0.0)
            _, hy = self._axis_delta_px(ax, 0.0, h)
            return max(1.0, wx/2.0), max(1.0, hy/2.0)
        r = self._node_radius_px(fig, p, l)
        if shape == "ellipse":
            return 1.35*r, .82*r
        return r, r

    def _node_boundary_distance_px(self, fig, ax, p: ProjectSpec, l: LayerSpec, center: Tuple[float,float], toward: Tuple[float,float]) -> float:
        c = ax.transData.transform(center); t = ax.transData.transform(toward)
        vx = float(t[0]-c[0]); vy = float(t[1]-c[1]); L = math.hypot(vx,vy)
        if L < 1e-9:
            return 0.0
        ux, uy = vx/L, vy/L
        rx, ry = self._node_half_extents_px(fig, ax, p, l)
        shape = (l.shape or "circle").lower()
        eps = 1e-12
        if shape in ("square", "rounded", "activation") or l.kind.lower() == "activation":
            tx = rx/max(abs(ux),eps); ty = ry/max(abs(uy),eps)
            return min(tx,ty)
        if shape == "diamond":
            return 1.0/max(abs(ux)/max(rx,eps)+abs(uy)/max(ry,eps),eps)
        return 1.0/math.sqrt((ux/max(rx,eps))**2 + (uy/max(ry,eps))**2)

    def _outside_junction_distance_px(self, fig, ax, p: ProjectSpec, l: LayerSpec, gap_value: float, layer_dx: float) -> float:
        half_w, _ = self._node_half_extents_px(fig, ax, p, l)
        layer_gap_px, _ = self._axis_delta_px(ax, max(abs(layer_dx),1e-6), 0.0)
        visible_floor = max(5.0, min(14.0, half_w * 0.18))
        user_offset = float(p.junction_offset) * max(24.0, layer_gap_px)
        gap_px = self._gap_distance_px(ax, gap_value, layer_dx)
        return half_w + visible_floor + user_offset + gap_px

    def _resolve_connection_endpoints(self, fig, ax, p: ProjectSpec, p1: Tuple[float,float], p2: Tuple[float,float], l1: Optional[LayerSpec], l2: Optional[LayerSpec], gap_value: float, end_mode: str, start_is_box: bool=False, end_is_box: bool=False):
        layer_dx = max(abs(p2[0]-p1[0]), 1e-6)
        gap_px = self._gap_distance_px(ax, gap_value, layer_dx)
        if str(end_mode).lower().startswith("outside"):
            sign = 1.0 if p2[0] >= p1[0] else -1.0
            if start_is_box or l1 is None:
                q1 = p1
            else:
                d1 = self._outside_junction_distance_px(fig, ax, p, l1, gap_value, layer_dx)
                q1 = self._shift_display(ax, p1, sign*d1, 0.0)
            if end_is_box or l2 is None:
                q2 = p2
            else:
                d2 = self._outside_junction_distance_px(fig, ax, p, l2, gap_value, layer_dx)
                q2 = self._shift_display(ax, p2, -sign*d2, 0.0)
            return q1, q2
        start_surface = 0.0 if start_is_box or l1 is None else self._node_boundary_distance_px(fig, ax, p, l1, p1, p2)
        end_surface = 0.0 if end_is_box or l2 is None else self._node_boundary_distance_px(fig, ax, p, l2, p2, p1)
        return self._shorten_display(ax, p1, p2, start_surface+gap_px, end_surface+gap_px)



    @staticmethod
    def _shorten_display(ax,p1,p2,start_px,end_px):
        P=ax.transData.transform([p1,p2]); x1,y1=P[0]; x2,y2=P[1]; dx=x2-x1;dy=y2-y1;L=math.hypot(dx,dy)
        if L<1e-9:return p1,p2
        maxtrim=.42*L; s=min(max(0,start_px),maxtrim); e=min(max(0,end_px),maxtrim); ux,uy=dx/L,dy/L
        Q=[(x1+ux*s,y1+uy*s),(x2-ux*e,y2-uy*e)]
        d=ax.transData.inverted().transform(Q)
        return tuple(d[0]),tuple(d[1])

    def _node_radius_px(self,fig,p:ProjectSpec,l:LayerSpec):
        # Project node_radius remains familiar while rendering geometry in display units.
        diameter_pt=max(8, p.node_radius*220*l.node_scale)
        return .5*diameter_pt*fig.dpi/72.0

    def _draw_node(self,fig,ax,x,y,l:LayerSpec,p:ProjectSpec,text=""):
        marker=self.marker_map.get(l.shape)
        dia_pt=max(8,p.node_radius*220*l.node_scale)
        s=dia_pt**2
        face=to_rgba(l.color,l.fill_alpha)
        edge=l.border_color
        artists=[]
        if marker:
            c=ax.scatter([x],[y],s=s,marker=marker,facecolors=[face],edgecolors=edge,linewidths=l.border_width,zorder=5)
            artists.append(c)
        elif l.shape=="ellipse":
            # A transformed marker keeps the ellipse stable in screen coordinates even when axes are stretched.
            ms=MarkerStyle("o")
            path=ms.get_path().transformed(ms.get_transform()).transformed(Affine2D().scale(1.35,0.82))
            c=ax.scatter([x],[y],s=s,marker=path,facecolors=[face],edgecolors=edge,linewidths=l.border_width,zorder=5)
            artists.append(c)
        elif l.shape in ("rounded","activation") or l.kind.lower()=="activation":
            # Compact blocks rendered directly in data coordinates; activation glyphs use a squarer box for clarity.
            is_activation=(l.shape=="activation" or l.kind.lower()=="activation")
            if is_activation:
                w=.42*l.node_scale; h=.38*l.node_scale; rounding=0.022
            else:
                w=.30*l.node_scale; h=.28*l.node_scale; rounding=0.035
            box=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle=f"round,pad=0.01,rounding_size={rounding}",
                               facecolor=face,edgecolor=edge,lw=l.border_width,zorder=5)
            ax.add_patch(box); artists.append(box)
            if not text and is_activation:
                self._draw_activation_icon(ax,x,y,l.activation,w,h,edge)
        else:
            c=ax.scatter([x],[y],s=s,marker="o",facecolors=[face],edgecolors=edge,linewidths=l.border_width,zorder=5); artists.append(c)
        if text:
            t=ax.text(x,y,text,ha="center",va="center",fontsize=l.node_text_font_size,family=l.node_text_font_family,color=l.node_text_color,zorder=7)
            artists.append(t)
        return artists

    def _draw_activation_icon(self,ax,x,y,activation,width,height,color):
        left=x-width*.36;right=x+width*.36;bottom=y-height*.31;top=y+height*.31
        ax.add_line(Line2D([left,right],[bottom,bottom],color="#B9B9B9",lw=.55,zorder=6)); ax.add_line(Line2D([left,left],[bottom,top],color="#B9B9B9",lw=.55,zorder=6))
        act=(activation or "sigmoid").lower(); xs=[left+(right-left)*i/30 for i in range(31)]; ys=[]
        for xx in xs:
            u=(xx-x)/(max(width*.11,1e-6))
            if "relu" in act:v=max(0,min(1,.5+.13*u))
            elif "tanh" in act:v=.5*(math.tanh(u)+1)
            elif "linear" in act:v=max(0,min(1,.5+.10*u))
            else:v=1/(1+math.exp(-max(-40,min(40,u))))
            ys.append(bottom+(top-bottom)*v)
        ax.add_line(Line2D(xs,ys,color=color,lw=1.0,zorder=7))

    @staticmethod
    def _shift_display(ax, p, dx_px, dy_px):
        q = ax.transData.transform(p)
        q2 = (q[0] + dx_px, q[1] + dy_px)
        d = ax.transData.inverted().transform(q2)
        return tuple(d)

    def _curve_path(self, p1, p2, strength):
        dx = p2[0] - p1[0]
        c = max(0.05, abs(dx) * max(0.0, strength))
        verts = [p1, (p1[0] + c, p1[1]), (p2[0] - c, p2[1]), p2]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        return MplPath(verts, codes)

    def _draw_connection(self, ax, p1, p2, render, geometry, color, alpha, width, head, rng,
                         random_on, prob, minf, maxf, curve_strength=0.32):
        lw = width
        if random_on and rng.random() < max(0, min(1, prob)):
            lw *= rng.uniform(max(.01, minf), max(minf, maxf))
        is_arrow = render.lower() == "arrows"
        is_curved = geometry.lower().startswith("curved")
        if is_curved:
            path = self._curve_path(p1, p2, curve_strength)
            if is_arrow:
                ar = FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=max(2, head),
                                     lw=lw, color=color, alpha=alpha, zorder=2,
                                     shrinkA=0, shrinkB=0)
                ax.add_patch(ar)
            else:
                ax.add_patch(PathPatch(path, fill=False, lw=lw, edgecolor=color, alpha=alpha, zorder=2))
        else:
            if is_arrow:
                ar = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=max(2, head), lw=lw,
                                     color=color, alpha=alpha, zorder=2, shrinkA=0, shrinkB=0)
                ax.add_patch(ar)
            else:
                ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, lw=lw, zorder=2))

    def _load_image(self,path:str):
        pp=Path(path)
        if not pp.is_absolute():pp=(BASE_DIR/pp).resolve()
        if not pp.exists():return None
        if pp.suffix.lower()==".svg":
            try:
                import cairosvg
                png=cairosvg.svg2png(url=str(pp)); return Image.open(io.BytesIO(png)).convert("RGBA")
            except Exception:
                return None
        return Image.open(pp).convert("RGBA")

    def _draw_annotation(self,ax,a:AnnotationSpec,index:int):
        kind=a.kind.lower(); ls=self._linestyle(a.line_style); artist=None
        if kind=="text":
            artist=ax.text(a.x,a.y,a.text,ha=a.ha,va=a.va,fontsize=a.font_size,family=a.font_family,
                           fontweight="bold" if a.bold else "normal",fontstyle="italic" if a.italic else "normal",
                           color=a.color,alpha=a.alpha,rotation=a.rotation,zorder=a.zorder)
        elif kind=="image":
            im=self._load_image(a.image_path)
            if im is not None:
                oi=OffsetImage(im,zoom=a.image_scale)
                artist=AnnotationBbox(oi,(a.x,a.y),frameon=False,pad=0,zorder=a.zorder)
                ax.add_artist(artist)
            else:
                artist=ax.text(a.x,a.y,"[image unavailable]",color="#B00020",fontsize=9,ha="center",va="center",zorder=a.zorder)
        elif kind in ("arrow","line","divider"):
            if kind=="arrow":
                artist=FancyArrowPatch((a.x,a.y),(a.x2,a.y2),arrowstyle="-|>",mutation_scale=a.arrow_head_size,lw=a.line_width,color=a.color,alpha=a.alpha,linestyle=ls,zorder=a.zorder)
                ax.add_patch(artist)
            else:
                artist=Line2D([a.x,a.x2],[a.y,a.y2],color=a.color,alpha=a.alpha,lw=a.line_width,linestyle=ls,zorder=a.zorder); ax.add_line(artist)
        elif kind in ("rectangle","rounded","pill"):
            rounding=(a.height/2 if kind=="pill" else (.08*min(a.width,a.height) if kind=="rounded" else 0))
            if kind=="rectangle":
                artist=Rectangle((a.x-a.width/2,a.y-a.height/2),a.width,a.height,facecolor=to_rgba(a.fill,a.alpha),edgecolor=a.border_color,lw=a.line_width,linestyle=ls,zorder=a.zorder,angle=a.rotation)
            else:
                artist=FancyBboxPatch((a.x-a.width/2,a.y-a.height/2),a.width,a.height,boxstyle=f"round,pad=0.01,rounding_size={rounding}",facecolor=to_rgba(a.fill,a.alpha),edgecolor=a.border_color,lw=a.line_width,linestyle=ls,zorder=a.zorder)
            ax.add_patch(artist)
            if a.text:
                ax.text(a.x,a.y,a.text,ha="center",va="center",fontsize=a.font_size,family=a.font_family,color=a.color,fontweight="bold" if a.bold else "normal",zorder=a.zorder+1)
        elif kind=="circle":
            artist=Ellipse((a.x,a.y),a.width,a.height,facecolor=to_rgba(a.fill,a.alpha),edgecolor=a.border_color,lw=a.line_width,linestyle=ls,zorder=a.zorder);ax.add_patch(artist)
        if artist is not None:
            try: artist.set_picker(True)
            except Exception: pass
            self.annotation_artists.append((artist,index))
        return artist

    def draw(self,fig,p:ProjectSpec,preview=False):
        validate_project(p)
        matplotlib.rcParams["mathtext.fontset"]=p.math_fontset or "stix"
        fig.clear(); fig.set_facecolor(p.background); ax=fig.add_subplot(111); ax.set_facecolor(p.background); ax.axis("off")
        self.annotation_artists=[]

        layers=p.layers; n=len(layers)
        xs=[i*p.layer_spacing + (layers[i].x_offset if p.layout_mode!="Automatic" else 0.0) for i in range(n)]
        layer_positions=[]; ellipsis=[]
        # First pass without input/output box compression. This fixes the v3 one-layer-span bug.
        global_span=max(0.55, max(.5*(max(1,l.visible_nodes)-1)*p.vertical_spacing for l in layers))
        for i,l in enumerate(layers):
            yoff=l.y_offset if p.layout_mode!="Automatic" else 0.0
            use_ellipsis = bool(l.show_ellipsis and l.size>l.visible_nodes and l.visible_nodes>=2)
            if use_ellipsis:
                total_slots=max(3,l.visible_nodes)
                shown_nodes=max(2,total_slots-1)
                top_count=(shown_nodes+1)//2
                ys_full=self.centered_positions(total_slots,p.vertical_spacing,global_span)
                ell_y=ys_full[top_count]
                ys=ys_full[:top_count]+ys_full[top_count+1:]
                ellipsis.append((xs[i],ell_y+yoff))
            else:
                ys=self.centered_positions(l.visible_nodes,p.vertical_spacing,global_span)
                ellipsis.append(None)
            pts=[(xs[i],y+yoff) for y in ys]; layer_positions.append(pts)

        # Optional input/output item boxes replace visual nodes at the terminal layers.
        has_in=p.show_input_items and bool(p.input_items) and layers[0].kind.lower()=="input"
        has_out=p.show_output_items and bool(p.output_items) and layers[-1].kind.lower()=="output"
        in_width=(p.input_box_width if p.input_box_width>0 else self._text_extent_width(p.input_items)) if has_in else 0
        out_width=(p.output_box_width if p.output_box_width>0 else self._text_extent_width(p.output_items)) if has_out else 0
        in_ys=self.centered_positions(len(p.input_items),p.vertical_spacing,global_span) if has_in else []
        out_ys=self.centered_positions(len(p.output_items),p.vertical_spacing,global_span) if has_out else []
        in_right=[];out_left=[]
        if has_in:
            for y,txt in zip(in_ys,p.input_items):
                box=FancyBboxPatch((xs[0]-in_width/2,y-.15),in_width,.30,boxstyle="round,pad=.02,rounding_size=.04",facecolor="#FFFFFF",edgecolor="#222222",lw=.8,zorder=5);ax.add_patch(box)
                ax.text(xs[0],y,txt,ha="center",va="center",fontsize=p.item_font_size,family=p.item_font_family,color="#111111",zorder=6);in_right.append((xs[0]+in_width/2,y))
        if has_out:
            for y,txt in zip(out_ys,p.output_items):
                box=FancyBboxPatch((xs[-1]-out_width/2,y-.15),out_width,.30,boxstyle="round,pad=.02,rounding_size=.04",facecolor="#FFFFFF",edgecolor="#222222",lw=.8,zorder=5);ax.add_patch(box)
                ax.text(xs[-1],y,txt,ha="center",va="center",fontsize=p.item_font_size,family=p.item_font_family,color="#111111",zorder=6);out_left.append((xs[-1]-out_width/2,y))

        # ------------------------------------------------------------------
        # Canonical layout lock (v1.0): establish final axes geometry BEFORE any
        # display-space endpoint calculation. Preview and export therefore use
        # the same transform.
        # ------------------------------------------------------------------
        layout_y = [y for i,pts in enumerate(layer_positions)
                    if not ((i==0 and has_in) or (i==n-1 and has_out)) for _,y in pts]
        layout_y += list(in_ys) + list(out_ys)
        pre_bottom = min(layout_y or [-global_span]); pre_top = max(layout_y or [global_span])
        pre_header_top = pre_top
        for g in p.groups:
            if 0 <= g.start_layer <= g.end_layer < n:
                gy = pre_top + g.y_offset
                pre_header_top = max(pre_header_top, gy + g.height)
        if p.title.strip():
            pre_header_top += .75
        pre_ext_min = xs[0] - (p.external_arrow_length if p.show_input_arrows and not has_in else 0.0)
        pre_ext_max = xs[-1] + (p.external_arrow_length if p.show_output_arrows and not has_out else 0.0)
        pre_xmin = min(pre_ext_min, xs[0]) - p.margins - (in_width/2 if has_in else 0)
        pre_xmax = max(pre_ext_max, xs[-1]) + p.margins + (out_width/2 if has_out else 0)
        pre_ymin = pre_bottom - p.margins - .55
        pre_ymax = pre_header_top + p.margins
        for a in p.annotations:
            kind = a.kind.lower()
            if kind in ("line","arrow","divider"):
                bx1,bx2 = sorted((a.x,a.x2)); by1,by2 = sorted((a.y,a.y2))
            elif kind in ("rectangle","rounded","pill","circle"):
                bx1,bx2 = a.x-a.width/2, a.x+a.width/2; by1,by2 = a.y-a.height/2, a.y+a.height/2
            elif kind == "image":
                hh = max(.28, 1.15*a.image_scale); hw = max(.35, 1.35*a.image_scale)
                bx1,bx2 = a.x-hw, a.x+hw; by1,by2 = a.y-hh, a.y+hh
            else:
                bx1=bx2=a.x; by1=by2=a.y
            pre_xmin=min(pre_xmin,bx1); pre_xmax=max(pre_xmax,bx2)
            pre_ymin=min(pre_ymin,by1); pre_ymax=max(pre_ymax,by2)
        fig.subplots_adjust(left=.015,right=.985,bottom=.04,top=.98)
        ax.set_xlim(pre_xmin,pre_xmax); ax.set_ylim(pre_ymin,pre_ymax)
        ax.set_aspect("equal", adjustable="box", anchor="C")
        try: ax.apply_aspect()
        except Exception: pass

        # Connections before nodes.
        for li in range(n-1):
            s=self._pair_style(p,li,li+1); mode,render,color,alpha,width,head,gap,ron,prob,minf,maxf=self._resolved_pair(p,s)
            if mode.lower()=="none":continue
            Apts=in_right if (li==0 and has_in) else layer_positions[li]
            Bpts=out_left if (li+1==n-1 and has_out) else layer_positions[li+1]
            if not Apts or not Bpts:continue
            if mode.lower()=="dense":
                total=len(Apts)*len(Bpts)
                # Avoid accidental pathological dense drawings.
                if total>12000:
                    pairs=self._sample_pairs(len(Apts),len(Bpts),min(p.max_connections_per_pair,12000))
                else:pairs=[(k//len(Bpts),k%len(Bpts)) for k in range(total)]
            elif mode.lower()=="adjacent-only":pairs=self._adjacent_pairs(len(Apts),len(Bpts))
            else:pairs=self._sample_pairs(len(Apts),len(Bpts),p.max_connections_per_pair)
            rng=random.Random(p.random_seed+li*1009+s.seed_offset)
            geometry = p.connection_geometry
            end_mode = p.connection_end_mode
            layer_dx = (layer_positions[li+1][0][0] - layer_positions[li][0][0]) if (layer_positions[li] and layer_positions[li+1]) else p.layer_spacing
            for pair_counter, (ia,ib) in enumerate(pairs):
                draw_color = self._pick_connection_color(p, color, li*200003, pair_counter, rng)
                p1,p2 = self._resolve_connection_endpoints(
                    fig, ax, p, Apts[ia], Bpts[ib],
                    None if (li==0 and has_in) else layers[li],
                    None if (li+1==n-1 and has_out) else layers[li+1],
                    gap, end_mode, start_is_box=(li==0 and has_in), end_is_box=(li+1==n-1 and has_out))
                self._draw_connection(ax,p1,p2,render,geometry,draw_color,alpha,width,head,rng,ron,prob,minf,maxf,p.curve_strength)

        # Highlighted connections drawn on top of the base network.
        def _highlight_edge(li, ia, ib, color, alpha, width_factor, render, head):
            Apts = in_right if (li == 0 and has_in) else layer_positions[li]
            Bpts = out_left if (li + 1 == n-1 and has_out) else layer_positions[li+1]
            if not Apts or not Bpts:
                return
            ia=max(0,min(int(ia),len(Apts)-1)); ib=max(0,min(int(ib),len(Bpts)-1))
            gap = p.arrow_gap if str(render).lower()=="arrows" else p.line_gap
            q1,q2 = self._resolve_connection_endpoints(
                fig,ax,p,Apts[ia],Bpts[ib],
                None if (li==0 and has_in) else layers[li],
                None if (li+1==n-1 and has_out) else layers[li+1],
                gap,p.connection_end_mode,start_is_box=(li==0 and has_in),end_is_box=(li+1==n-1 and has_out))
            self._draw_connection(ax,q1,q2,render,p.connection_geometry,color,alpha,
                                  p.connection_width*width_factor,head,random.Random(0),False,0.0,1.0,1.0,p.curve_strength)

        def _highlight_node(li, idx, color, fill="#FFFFFF"):
            if li<0 or li>=n or (li==0 and has_in) or (li==n-1 and has_out):
                return
            pts=layer_positions[li]
            if not pts:return
            idx=max(0,min(int(idx),len(pts)-1)); x,y=pts[idx]
            ax.scatter([x],[y],s=max(16,(max(8,p.node_radius*220*layers[li].node_scale))**2*1.10),
                       marker=self.marker_map.get(layers[li].shape,"o"),facecolors=[to_rgba(fill,1.0)],
                       edgecolors=color,linewidths=max(1.4,layers[li].border_width+1.2),zorder=6)

        for hp in getattr(p, "highlight_paths", []) or []:
            if not isinstance(hp, dict):
                continue
            hmode=str(hp.get("mode","path")).lower()
            color=hp.get("color", "#E53935"); alpha=float(hp.get("alpha", 0.95)); width_factor=float(hp.get("width_factor", 2.6))
            render=hp.get("render", p.connection_style); head=float(hp.get("arrow_head_size", p.arrow_head_size))
            node_fill = hp.get("node_fill", "#FFFFFF")
            node_border = hp.get("node_border", color)
            if hmode.startswith("cascade"):
                start_layer=max(0,min(int(hp.get("start_layer",0)),max(0,n-2)))
                source=int(hp.get("source_node",0)); pivot=int(hp.get("pivot_node",0))
                _highlight_edge(start_layer,source,pivot,color,alpha,width_factor,render,head)
                if hp.get("highlight_nodes",True):
                    _highlight_node(start_layer,source,node_border,node_fill)
                    _highlight_node(start_layer+1,pivot,node_border,node_fill)
                active=[pivot]
                for li in range(start_layer+1,n-1):
                    Bpts=out_left if (li+1==n-1 and has_out) else layer_positions[li+1]
                    if not Bpts: break
                    targets=list(range(len(Bpts)))
                    for ia in active:
                        for ib in targets:
                            _highlight_edge(li,ia,ib,color,alpha,width_factor,render,head)
                    if hp.get("highlight_nodes",True):
                        for ib in targets:
                            _highlight_node(li+1,ib,node_border,node_fill)
                    active = targets if hp.get("all_downstream", True) else []
                    if not active:
                        break
                continue
            nodes = hp.get("nodes", [])
            if not nodes or len(nodes) != n:
                continue
            for li in range(n-1):
                _highlight_edge(li,nodes[li],nodes[li+1],color,alpha,width_factor,render,head)
            if hp.get("highlight_nodes"):
                for li, node_index in enumerate(nodes):
                    _highlight_node(li,node_index,node_border,node_fill)

        # Nodes / blocks and ellipses.
        all_y=[]
        for i,l in enumerate(layers):
            if (i==0 and has_in) or (i==n-1 and has_out):continue
            pts=layer_positions[i]; texts=l.node_texts or []
            for j,(x,y) in enumerate(pts):
                self._draw_node(fig,ax,x,y,l,p,texts[j] if j<len(texts) else ""); all_y.append(y)
            if ellipsis[i] is not None:
                x,y=ellipsis[i]; ax.text(x,y,r"$\vdots$",ha="center",va="center",fontsize=max(10,p.layer_font_size),family=p.font_family,color=l.caption_color,zorder=8)

        # Layer captions.
        bottom=min(all_y or [-global_span]); top=max(all_y or [global_span])
        for i,l in enumerate(layers):
            if l.caption_position=="None":continue
            txt=(l.caption_text or l.name).strip()
            if p.show_nominal_size and l.size>0:txt += f"\n({l.size})"
            if p.show_activation and l.activation:txt += f"\n{l.activation}"
            if l.note:txt += f"\n{l.note}"
            if not txt:continue
            yy=top+.38 if l.caption_position=="Top" else bottom-.44
            ax.text(xs[i],yy,txt,ha="center",va="bottom" if l.caption_position=="Top" else "top",fontsize=l.caption_font_size,family=p.font_family,color=l.caption_color,fontweight="bold" if l.caption_bold else "normal",zorder=9)

        # External arrows/lines per terminal node.
        ext_min=xs[0];ext_max=xs[-1]
        def extline(start,end):
            if p.external_arrow_style.lower()=="lines":ax.add_line(Line2D([start[0],end[0]],[start[1],end[1]],color=p.external_arrow_color,lw=p.external_arrow_width,zorder=2))
            else:ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=p.external_arrow_head_size,color=p.external_arrow_color,lw=p.external_arrow_width,zorder=2))
        if p.show_input_arrows and not has_in:
            rpx=self._node_radius_px(fig,p,layers[0]);
            for pt in layer_positions[0]:
                start=(pt[0]-p.external_arrow_length,pt[1]); q1,q2=self._shorten_display(ax,start,pt,0,rpx+p.external_arrow_gap*fig.dpi*2);extline(q1,q2);ext_min=min(ext_min,start[0])
        if p.show_output_arrows and not has_out:
            rpx=self._node_radius_px(fig,p,layers[-1]);
            for pt in layer_positions[-1]:
                end=(pt[0]+p.external_arrow_length,pt[1]); q1,q2=self._shorten_display(ax,pt,end,rpx+p.external_arrow_gap*fig.dpi*2,0);extline(q1,q2);ext_max=max(ext_max,end[0])

        # Group headers.
        header_top=top
        for g in p.groups:
            if not (0<=g.start_layer<=g.end_layer<n):continue
            x1=xs[g.start_layer];x2=xs[g.end_layer];cx=(x1+x2)/2; y=top+g.y_offset; header_top=max(header_top,y+g.height)
            if g.style.lower()=="text":
                ax.text(cx,y,g.title,ha="center",va="center",fontsize=g.font_size,family=p.font_family,color=g.color,fontweight="bold" if g.bold else "normal",zorder=12)
            else:
                w=max(.8,(x2-x1)+g.padding*2); rounding=g.height/2 if g.style.lower()=="pill" else .05
                patch=FancyBboxPatch((cx-w/2,y-g.height/2),w,g.height,boxstyle=f"round,pad=.01,rounding_size={rounding}",facecolor=g.fill,edgecolor=g.border_color or "none",lw=.8,zorder=10);ax.add_patch(patch)
                ax.text(cx,y,g.title,ha="center",va="center",fontsize=g.font_size,family=p.font_family,color=g.color,fontweight="bold" if g.bold else "normal",zorder=11)

        # Free-form annotations.
        for idx,a in enumerate(p.annotations):self._draw_annotation(ax,a,idx)

        # Project title/subtitle.
        if p.title.strip():
            ax.text((xs[0]+xs[-1])/2,header_top+.50,p.title.strip(),ha="center",va="bottom",fontsize=p.title_font_size,family=p.font_family,fontweight="bold",color="#111111",zorder=30)
            if p.subtitle.strip():ax.text((xs[0]+xs[-1])/2,header_top+.27,p.subtitle.strip(),ha="center",va="bottom",fontsize=max(8,p.title_font_size-4),family=p.font_family,color="#555555",zorder=30)
            header_top+=.75

        # Limits include annotations. Annotation images are offset artists, so a generous margin is used.
        xmin=min(ext_min,xs[0]) - p.margins - (in_width/2 if has_in else 0)
        xmax=max(ext_max,xs[-1]) + p.margins + (out_width/2 if has_out else 0)
        ymin=bottom-p.margins-.55; ymax=header_top+p.margins
        for a in p.annotations:
            xmin=min(xmin,a.x,a.x2 if a.kind.lower() in ("line","arrow","divider") else a.x)
            xmax=max(xmax,a.x,a.x2 if a.kind.lower() in ("line","arrow","divider") else a.x)
            ymin=min(ymin,a.y,a.y2 if a.kind.lower() in ("line","arrow","divider") else a.y)
            ymax=max(ymax,a.y,a.y2 if a.kind.lower() in ("line","arrow","divider") else a.y)
        ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_aspect("equal", adjustable="box", anchor="C")
        fig.subplots_adjust(left=.015,right=.985,bottom=.04,top=.98)
        return ax


# -----------------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}  v{APP_VERSION}")
        self.geometry("1600x940")
        self.minsize(1240,760)
        self.project=sample_01_formula_network(); self.renderer=ArchitectureRenderer(self); self.vars:Dict[str,tk.Variable]={}; self.dirty=False
        self._drag_ann_index=None; self._drag_last=None; self.package_tempdir=None
        self._apply_theme(); self._build_ui(); self._load_project_into_ui(); self.dirty=False
        self.after(120,self._set_initial_column_ratios)
        self.after(180,self.redraw)
        self.protocol("WM_DELETE_WINDOW",self.on_close)

    def _apply_theme(self):
        s=ttk.Style(self)
        try:s.theme_use("clam")
        except Exception:pass
        s.configure("Title.TLabel",font=("Segoe UI",15,"bold"))
        s.configure("Section.TLabel",font=("Segoe UI",10,"bold"))
        s.configure("Accent.TButton",font=("Segoe UI",10,"bold"))
        s.configure("Samples.TLabel",font=("Segoe UI",12,"bold"),foreground="#17345F")
        s.configure("Samples.TCombobox",font=("Segoe UI",12,"bold"),fieldbackground="#1E6FD9",background="#1558AF",foreground="#FFFFFF",arrowsize=19,padding=6)
        s.map("Samples.TCombobox",
              fieldbackground=[("readonly","#1E6FD9"), ("active","#287BE6")],
              background=[("readonly","#1558AF"), ("active","#104A95")],
              foreground=[("readonly","#FFFFFF")],
              selectbackground=[("readonly","#1E6FD9")],
              selectforeground=[("readonly","#FFFFFF")])
        s.configure("Middle.Vertical.TScrollbar",gripcount=0,background="#5D79A9",troughcolor="#E6EBF3",bordercolor="#C8D1DF",darkcolor="#5D79A9",lightcolor="#7F98BF",arrowcolor="#FFFFFF",width=17)
        s.map("Middle.Vertical.TScrollbar",background=[("active","#486A9F"), ("pressed","#355889")])
        s.configure("Project.TButton",font=("Segoe UI",9,"bold"),padding=(10,6))
        s.configure("PreviewAction.TButton",font=("Segoe UI",10,"bold"),background="#2F6FED",foreground="#FFFFFF",padding=(8,7))
        s.map("PreviewAction.TButton",background=[("active","#2459C7"), ("pressed","#1E4BAA")])
        s.configure("AutoFitAction.TButton",font=("Segoe UI",10,"bold"),background="#F0B429",foreground="#1D2433",padding=(8,7))
        s.map("AutoFitAction.TButton",background=[("active","#D99C13"), ("pressed","#C28A0C")])
        s.configure("ExportFigure.TButton",font=("Segoe UI",10,"bold"),background="#14866D",foreground="#FFFFFF",padding=(8,8))
        s.map("ExportFigure.TButton",background=[("active","#0F705B"), ("pressed","#0B5C4A")])
        s.configure("ExportAll.TButton",font=("Segoe UI",10,"bold"),background="#6B4BC3",foreground="#FFFFFF",padding=(8,8))
        s.map("ExportAll.TButton",background=[("active","#5739AA"), ("pressed","#472F8B")])
        # Braces keep the family name with a space valid in Tcl/Tk when the drop-down posts.
        self.option_add("*TCombobox*Listbox.font", "{Segoe UI} 11")
        self.option_add("*TCombobox*Listbox.background", "#F7FAFF")
        self.option_add("*TCombobox*Listbox.foreground", "#17243A")
        self.option_add("*TCombobox*Listbox.selectBackground", "#1E6FD9")
        self.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

    def _prepare_samples_dropdown(self):
        """Refresh the Samples list immediately before the drop-down is posted."""
        try:
            self.samples_combo.configure(values=list(PRESETS))
        except Exception:
            pass

    def _set_initial_column_ratios(self):
        """Set the opening column ratio to about 23% / 14% / 63%.

        The panes remain fully user-resizable by dragging either vertical sash.
        This method runs once after Tk has calculated the real window size.
        """
        try:
            self.update_idletasks()
            w=max(1,self.main_paned.winfo_width())
            # Screenshot-inspired initial proportions: left ~23%, middle ~14%, preview ~63%.
            self.main_paned.sashpos(0,int(w*0.23))
            self.main_paned.sashpos(1,int(w*0.37))
        except Exception:
            pass

    def _build_ui(self):
        # ------------------------------------------------------------------
        # Top bar: Samples at the far-left, project operations at the right.
        # ------------------------------------------------------------------
        top=ttk.Frame(self,padding=(10,8));top.pack(fill="x")
        ttk.Label(top,text="Samples:",style="Samples.TLabel").pack(side="left",padx=(0,7))
        self.preset_var=tk.StringVar(value=list(PRESETS)[0])
        self.samples_combo=ttk.Combobox(top,textvariable=self.preset_var,values=list(PRESETS),state="readonly",width=43,height=18,style="Samples.TCombobox",takefocus=True)
        self.samples_combo.pack(side="left",ipady=2)
        self.samples_combo.bind("<<ComboboxSelected>>",self.apply_preset)
        self.samples_combo.configure(postcommand=self._prepare_samples_dropdown)

        project_bar=ttk.Frame(top);project_bar.pack(side="right")
        for text,cmd in [("New Project",self.new_project),("Open Project",self.open_project),("Save Project",self.save_project),("Save Project As",self.save_project_as)]:
            ttk.Button(project_bar,text=text,style="Project.TButton",command=cmd).pack(side="left",padx=3)

        # ------------------------------------------------------------------
        # Three resizable columns. Default opening ratio is set after layout.
        # ------------------------------------------------------------------
        self.main_paned=ttk.Panedwindow(self,orient="horizontal")
        self.main_paned.pack(fill="both",expand=True,padx=8,pady=(0,8))
        left=ttk.Frame(self.main_paned,padding=7)
        middle=ttk.Frame(self.main_paned,padding=7)
        right=ttk.Frame(self.main_paned,padding=7)
        self.main_paned.add(left,weight=23)
        self.main_paned.add(middle,weight=14)
        self.main_paned.add(right,weight=63)

        # ------------------------------------------------------------------
        # Left column
        # ------------------------------------------------------------------
        ttk.Label(left,text="Architecture",style="Section.TLabel").pack(anchor="w")
        ttk.Label(left,text="Double-click a layer to edit. Semi-manual offsets are available per layer.",foreground="#666",wraplength=330).pack(anchor="w",pady=(2,6))
        self.layer_list=tk.Listbox(left,height=12,exportselection=False);self.layer_list.pack(fill="x");self.layer_list.bind("<Double-1>",lambda e:self.edit_layer())
        b=ttk.Frame(left);b.pack(fill="x",pady=5)
        ttk.Button(b,text="+ Add",command=self.add_layer).pack(side="left",padx=2);ttk.Button(b,text="Edit",command=self.edit_layer).pack(side="left",padx=2);ttk.Button(b,text="Duplicate",command=self.duplicate_layer).pack(side="left",padx=2);ttk.Button(b,text="Delete",command=self.delete_layer).pack(side="left",padx=2);ttk.Button(b,text="↑",width=3,command=lambda:self.move_layer(-1)).pack(side="right");ttk.Button(b,text="↓",width=3,command=lambda:self.move_layer(1)).pack(side="right")

        ttk.Separator(left).pack(fill="x",pady=7);ttk.Label(left,text="Groups / headers",style="Section.TLabel").pack(anchor="w")
        self.group_list=tk.Listbox(left,height=5,exportselection=False);self.group_list.pack(fill="x",pady=(3,3));self.group_list.bind("<Double-1>",lambda e:self.edit_group())
        q=ttk.Frame(left);q.pack(fill="x");ttk.Button(q,text="+ Group",command=self.add_group).pack(side="left",padx=2);ttk.Button(q,text="Edit",command=self.edit_group).pack(side="left",padx=2);ttk.Button(q,text="Delete",command=self.delete_group).pack(side="left",padx=2)

        ttk.Separator(left).pack(fill="x",pady=7);ttk.Label(left,text="Canvas objects / annotations",style="Section.TLabel").pack(anchor="w")
        self.ann_list=tk.Listbox(left,height=7,exportselection=False);self.ann_list.pack(fill="both",expand=True,pady=(3,3));self.ann_list.bind("<Double-1>",lambda e:self.edit_annotation())
        q=ttk.Frame(left);q.pack(fill="x");ttk.Button(q,text="+ Text",command=lambda:self.add_annotation("Text")).pack(side="left",padx=1);ttk.Button(q,text="+ Image",command=self.add_image).pack(side="left",padx=1);ttk.Button(q,text="+ Shape",command=lambda:self.add_annotation("Rounded")).pack(side="left",padx=1);ttk.Button(q,text="+ Arrow",command=lambda:self.add_annotation("Arrow")).pack(side="left",padx=1)
        q=ttk.Frame(left);q.pack(fill="x",pady=(3,0));ttk.Button(q,text="Edit",command=self.edit_annotation).pack(side="left",padx=1);ttk.Button(q,text="Duplicate",command=self.duplicate_annotation).pack(side="left",padx=1);ttk.Button(q,text="Delete",command=self.delete_annotation).pack(side="left",padx=1)
        ttk.Label(left,text="Tip: unlocked canvas objects can be dragged directly in the preview.",foreground="#666",wraplength=330).pack(anchor="w",pady=(3,0))

        # ------------------------------------------------------------------
        # Middle column: upper controls scroll, Export stays fixed at bottom.
        # ------------------------------------------------------------------
        export_panel=ttk.Frame(middle,padding=(0,8,0,0));export_panel.pack(side="bottom",fill="x")
        sf=ScrollableFrame(middle,scrollbar_style="Middle.Vertical.TScrollbar");sf.pack(side="top",fill="both",expand=True)
        self.middle_scroll=sf
        m=sf.inner
        ttk.Label(m,text="Style & connections",style="Section.TLabel").pack(anchor="w",pady=(0,6))

        def entry(label,key,value,width=15):
            f=ttk.Frame(m);f.pack(fill="x",pady=2);ttk.Label(f,text=label).pack(side="left");v=tk.StringVar(value=str(value));self.vars[key]=v;e=ttk.Entry(f,textvariable=v,width=width);e.pack(side="right");v.trace_add("write",lambda *_:self._schedule_preview());return v
        def check(label,key,value):
            v=tk.BooleanVar(value=value);self.vars[key]=v;ttk.Checkbutton(m,text=label,variable=v,command=self.redraw).pack(anchor="w",pady=2);return v
        def combo(label,key,value,values):
            f=ttk.Frame(m);f.pack(fill="x",pady=2);ttk.Label(f,text=label).pack(side="left");v=tk.StringVar(value=value);self.vars[key]=v;c=ttk.Combobox(f,textvariable=v,values=values,state="readonly",width=16);c.pack(side="right");c.bind("<<ComboboxSelected>>",lambda e:self.redraw());return v
        def colorrow(label,key,value):
            f=ttk.Frame(m);f.pack(fill="x",pady=2);ttk.Label(f,text=label).pack(side="left");v=tk.StringVar(value=value);self.vars[key]=v;ttk.Entry(f,textvariable=v,width=12).pack(side="right");ttk.Button(f,text="Pick",width=5,command=lambda k=key:self.pick_color(k)).pack(side="right",padx=4)

        entry("Title","title",self.project.title,22);entry("Subtitle","subtitle",self.project.subtitle,22)
        fonts=sorted(set(tkfont.families()));combo("Main font","font_family",self.project.font_family,fonts);combo("Math font","math_fontset",self.project.math_fontset,["stix","stixsans","cm","dejavuserif","dejavusans"])
        entry("Title font size","title_font_size",self.project.title_font_size);entry("Node radius","node_radius",self.project.node_radius);entry("Layer spacing","layer_spacing",self.project.layer_spacing);entry("Vertical spacing","vertical_spacing",self.project.vertical_spacing);entry("Margins","margins",self.project.margins)
        combo("Layout mode","layout_mode",self.project.layout_mode,["Automatic","Semi-Automatic"]);colorrow("Background","background",self.project.background)

        ttk.Separator(m).pack(fill="x",pady=7);ttk.Label(m,text="Connections",style="Section.TLabel").pack(anchor="w")
        combo("Selection","connection_mode",self.project.connection_mode,["Dense","Sampled","Adjacent-only","None"]);combo("Rendering","connection_style",self.project.connection_style,["Lines","Arrows"]);combo("Geometry","connection_geometry",self.project.connection_geometry,["Straight","Curved-horizontal"]);combo("End mode","connection_end_mode",self.project.connection_end_mode,["Trim at cells","Outside junctions"]);colorrow("Color","connection_color",self.project.connection_color)
        combo("Color mode","connection_color_mode",getattr(self.project,"connection_color_mode","Single"),["Single","Palette cycle","Random palette"]);entry("Palette colors","connection_palette",getattr(self.project,"connection_palette","#FF4FA3, #66E5FF, #7D8CFF"),22)
        entry("Width","connection_width",self.project.connection_width);entry("Alpha","connection_alpha",self.project.connection_alpha);entry("Junction offset","junction_offset",self.project.junction_offset);entry("Curve strength","curve_strength",self.project.curve_strength);entry("Max shown/pair","max_connections_per_pair",self.project.max_connections_per_pair);entry("Line gap","line_gap",self.project.line_gap);entry("Arrow gap","arrow_gap",self.project.arrow_gap);entry("Arrow head","arrow_head_size",self.project.arrow_head_size)
        check("Randomly emphasize some connections","random_width_enabled",self.project.random_width_enabled);entry("Random probability","random_width_probability",self.project.random_width_probability);entry("Random min factor","random_width_min_factor",self.project.random_width_min_factor);entry("Random max factor","random_width_max_factor",self.project.random_width_max_factor);entry("Random seed","random_seed",self.project.random_seed)
        ttk.Button(m,text="Layer-pair styles…",command=self.manage_pair_styles).pack(fill="x",pady=4)

        ttk.Separator(m).pack(fill="x",pady=7);ttk.Label(m,text="Terminal arrows",style="Section.TLabel").pack(anchor="w")
        check("Show incoming arrows/lines","show_input_arrows",self.project.show_input_arrows);check("Show outgoing arrows/lines","show_output_arrows",self.project.show_output_arrows);combo("External style","external_arrow_style",self.project.external_arrow_style,["Arrows","Lines"]);colorrow("External color","external_arrow_color",self.project.external_arrow_color);entry("External length","external_arrow_length",self.project.external_arrow_length);entry("External width","external_arrow_width",self.project.external_arrow_width);entry("External head","external_arrow_head_size",self.project.external_arrow_head_size)

        ttk.Separator(m).pack(fill="x",pady=7);ttk.Label(m,text="Input / output boxes",style="Section.TLabel").pack(anchor="w")
        check("Show input item boxes","show_input_items",self.project.show_input_items);check("Show output item boxes","show_output_items",self.project.show_output_items);entry("Input box width (0=auto)","input_box_width",self.project.input_box_width);entry("Output box width (0=auto)","output_box_width",self.project.output_box_width)
        ttk.Label(m,text="Input items (one per line)").pack(anchor="w");self.input_text=ScrolledText(m,height=4,wrap="word");self.input_text.pack(fill="x",pady=2);self.input_text.bind("<KeyRelease>",lambda e:self._schedule_preview())
        ttk.Label(m,text="Output items (one per line)").pack(anchor="w");self.output_text=ScrolledText(m,height=4,wrap="word");self.output_text.pack(fill="x",pady=(2,10));self.output_text.bind("<KeyRelease>",lambda e:self._schedule_preview())

        # Fixed Export panel
        ttk.Separator(export_panel).pack(fill="x",pady=(0,7));ttk.Label(export_panel,text="Export",style="Section.TLabel").pack(anchor="w",pady=(0,4))
        def export_entry(label,key,value,width=12):
            f=ttk.Frame(export_panel);f.pack(fill="x",pady=2);ttk.Label(f,text=label).pack(side="left");v=tk.StringVar(value=str(value));self.vars[key]=v;ttk.Entry(f,textvariable=v,width=width).pack(side="right");v.trace_add("write",lambda *_:self._schedule_preview());return v
        export_entry("Figure width (in)","figure_width",self.project.figure_width);export_entry("Figure height (in)","figure_height",self.project.figure_height);export_entry("DPI","dpi",self.project.dpi)
        ttk.Button(export_panel,text="Update Preview",style="PreviewAction.TButton",command=self.redraw).pack(fill="x",pady=(6,3))
        ttk.Button(export_panel,text="Smart Auto-fit",style="AutoFitAction.TButton",command=self.autofit).pack(fill="x",pady=3)
        ttk.Button(export_panel,text="Export Figure…",style="ExportFigure.TButton",command=self.export_figure).pack(fill="x",pady=3)
        ttk.Button(export_panel,text="Export All Formats",style="ExportAll.TButton",command=self.export_all).pack(fill="x",pady=3)
        ttk.Label(export_panel,text="PNG / JPG / SVG / PDF / EPS",foreground="#666").pack(anchor="w",pady=(3,2))

        # ------------------------------------------------------------------
        # Preview column
        # ------------------------------------------------------------------
        head=ttk.Frame(right);head.pack(fill="x");ttk.Label(head,text="Live Preview",style="Section.TLabel").pack(side="left");self.status_var=tk.StringVar(value="Ready");ttk.Label(head,textvariable=self.status_var,foreground="#666").pack(side="right")
        self.preview_frame=ttk.Frame(right);self.preview_frame.pack(fill="both",expand=True,pady=(5,0))
        self.preview_label=ttk.Label(self.preview_frame,anchor="center");self.preview_label.pack(fill="both",expand=True)
        self.preview_frame.bind("<Configure>",lambda e:self._schedule_preview())
        self.fig=None;self.canvas=None;self._preview_photo=None

    def _schedule_preview(self):
        self.dirty=True
        if hasattr(self,"_preview_after"):
            try:self.after_cancel(self._preview_after)
            except Exception:pass
        self._preview_after=self.after(260,self.redraw)

    def _read_ui_into_project(self):
        p=self.project
        p.title=self.vars["title"].get().strip();p.subtitle=self.vars["subtitle"].get().strip();p.font_family=self.vars["font_family"].get().strip() or "Times New Roman";p.math_fontset=self.vars["math_fontset"].get().strip() or "stix";p.layout_mode=self.vars["layout_mode"].get()
        float_fields=["title_font_size","node_radius","layer_spacing","vertical_spacing","margins","connection_width","connection_alpha","junction_offset","curve_strength","line_gap","arrow_gap","arrow_head_size","random_width_probability","random_width_min_factor","random_width_max_factor","external_arrow_length","external_arrow_width","external_arrow_head_size","input_box_width","output_box_width","figure_width","figure_height"]
        int_fields=["max_connections_per_pair","random_seed","dpi"]
        string_fields=["background","connection_color","connection_color_mode","connection_palette","connection_mode","connection_style","connection_geometry","connection_end_mode","external_arrow_style","external_arrow_color"]
        bool_fields=["random_width_enabled","show_input_arrows","show_output_arrows","show_input_items","show_output_items"]
        for k in float_fields:setattr(p,k,float(self.vars[k].get()))
        for k in int_fields:setattr(p,k,int(float(self.vars[k].get())))
        for k in string_fields:setattr(p,k,self.vars[k].get().strip())
        for k in bool_fields:setattr(p,k,bool(self.vars[k].get()))
        # Card fields are accepted in project files but are not rendered.
        p.card_enabled=False; p.card_shadow=False
        p.input_items=[x.strip() for x in self.input_text.get("1.0","end").splitlines() if x.strip()];p.output_items=[x.strip() for x in self.output_text.get("1.0","end").splitlines() if x.strip()]
        validate_project(p)

    def _load_project_into_ui(self):
        self._refresh_lists();d=asdict(self.project)
        if self.vars:
            for k,v in self.vars.items():
                if k in d:
                    try:v.set(d[k])
                    except Exception:pass
        self.input_text.delete("1.0","end");self.input_text.insert("1.0","\n".join(self.project.input_items));self.output_text.delete("1.0","end");self.output_text.insert("1.0","\n".join(self.project.output_items))

    def _refresh_lists(self):
        self.layer_list.delete(0,"end")
        for i,l in enumerate(self.project.layers):self.layer_list.insert("end",f"{i:02d}  {l.kind}: {l.name}  [{l.visible_nodes}/{l.size or '—'}]  {l.shape}")
        self.group_list.delete(0,"end")
        for i,g in enumerate(self.project.groups):self.group_list.insert("end",f"{i:02d}  {g.title}  L{g.start_layer}–L{g.end_layer}  {g.style}")
        self.ann_list.delete(0,"end")
        for i,a in enumerate(self.project.annotations):self.ann_list.insert("end",f"{i:02d}  {a.kind}: {(a.text[:28] if a.text else Path(a.image_path).name if a.image_path else '')}")

    def _estimate_export_aspect(self):
        try:
            p = copy.deepcopy(self.project)
            fig = plt.Figure(figsize=(p.figure_width, p.figure_height), dpi=110, facecolor=p.background)
            self.renderer.draw(fig, p, preview=False)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = fig.get_tightbbox(renderer).expanded(1.01, 1.01)
            w = max(1e-6, float(bbox.width))
            h = max(1e-6, float(bbox.height))
            plt.close(fig)
            return max(0.4, min(4.0, w / h))
        except Exception:
            return max(0.1, float(self.project.figure_width) / max(0.1, float(self.project.figure_height)))

    def _sync_preview_figure_size(self):
        try:
            widget = self.canvas.get_tk_widget()
            avail_w = max(640, widget.winfo_width())
            avail_h = max(420, widget.winfo_height())
            aspect = self._estimate_export_aspect()
            if avail_w / avail_h > aspect:
                target_h = avail_h
                target_w = int(target_h * aspect)
            else:
                target_w = avail_w
                target_h = int(target_w / aspect)
            dpi = 100
            self.fig.set_dpi(dpi)
            self.fig.set_size_inches(max(3.0, target_w / dpi), max(2.4, target_h / dpi), forward=True)
        except Exception:
            pass

    def _render_preview_bytes(self, p: ProjectSpec, dpi: int = 120) -> bytes:
        q = copy.deepcopy(p)
        q.dpi = int(max(90, min(200, dpi)))
        fig = plt.Figure(figsize=(q.figure_width, q.figure_height), dpi=q.dpi, facecolor=q.background)
        self.renderer.draw(fig, q, preview=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=q.dpi, bbox_inches="tight", pad_inches=.05, facecolor=q.background)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def redraw(self):
        try:
            self._read_ui_into_project()
            avail_w = max(420, int(self.preview_frame.winfo_width()) - 12) if hasattr(self, "preview_frame") else 1100
            avail_h = max(300, int(self.preview_frame.winfo_height()) - 12) if hasattr(self, "preview_frame") else 680
            png_bytes = self._render_preview_bytes(self.project, dpi=125)
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            img.thumbnail((avail_w, avail_h), Image.Resampling.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.configure(image=self._preview_photo)
            self.status_var.set(f"{len(self.project.layers)} layers • {len(self.project.annotations)} canvas objects • {len(getattr(self.project, 'highlight_paths', []))} path highlight(s) • preview updated")
        except Exception as exc:
            self.status_var.set(f"Preview error: {exc}")
            print(traceback.format_exc())

    def pick_color(self,key):
        c=colorchooser.askcolor(self.vars[key].get(),parent=self)[1]
        if c:self.vars[key].set(c);self.redraw()

    # Layers
    def _sel(self,lb):
        s=lb.curselection();return s[0] if s else None
    def add_layer(self):
        d=LayerDialog(self);self.wait_window(d)
        if d.result:
            i=self._sel(self.layer_list);self.project.layers.insert((i+1) if i is not None else len(self.project.layers),d.result);self._refresh_lists();self.redraw()
    def edit_layer(self):
        i=self._sel(self.layer_list)
        if i is None:return
        d=LayerDialog(self,self.project.layers[i]);self.wait_window(d)
        if d.result:self.project.layers[i]=d.result;self._refresh_lists();self.layer_list.selection_set(i);self.redraw()
    def duplicate_layer(self):
        i=self._sel(self.layer_list)
        if i is None:return
        x=copy.deepcopy(self.project.layers[i]);x.name+=" copy";self.project.layers.insert(i+1,x);self._refresh_lists();self.redraw()
    def delete_layer(self):
        i=self._sel(self.layer_list)
        if i is None:return
        del self.project.layers[i]
        # Keep remaining index-based metadata consistent after deletion.
        pairs=[]
        for s in self.project.pair_styles:
            if s.source_layer==i or s.target_layer==i:
                continue
            if s.source_layer>i: s.source_layer-=1
            if s.target_layer>i: s.target_layer-=1
            pairs.append(s)
        self.project.pair_styles=pairs
        groups=[]
        for g in self.project.groups:
            if g.start_layer>i: g.start_layer-=1
            if g.end_layer>i: g.end_layer-=1
            if g.start_layer<=g.end_layer and g.end_layer<len(self.project.layers): groups.append(g)
        self.project.groups=groups
        self._refresh_lists();self.redraw()
    def move_layer(self,d):
        i=self._sel(self.layer_list)
        if i is None:return
        j=i+d
        if not(0<=j<len(self.project.layers)):return
        self.project.layers[i],self.project.layers[j]=self.project.layers[j],self.project.layers[i]
        # Clear layer-index-aware styles after reordering to prevent incorrect pair mappings.
        if self.project.pair_styles:
            self.project.pair_styles=[];self.status_var.set("Layer reordered; pair-specific styles cleared for safety.")
        self._refresh_lists();self.layer_list.selection_set(j);self.redraw()

    # Groups
    def add_group(self):
        d=GroupDialog(self,len(self.project.layers));self.wait_window(d)
        if d.result:self.project.groups.append(d.result);self._refresh_lists();self.redraw()
    def edit_group(self):
        i=self._sel(self.group_list)
        if i is None:return
        d=GroupDialog(self,len(self.project.layers),self.project.groups[i]);self.wait_window(d)
        if d.result:self.project.groups[i]=d.result;self._refresh_lists();self.group_list.selection_set(i);self.redraw()
    def delete_group(self):
        i=self._sel(self.group_list)
        if i is not None:del self.project.groups[i];self._refresh_lists();self.redraw()

    # Annotations
    def add_annotation(self,kind="Text"):
        d=AnnotationDialog(self,force_kind=kind);self.wait_window(d)
        if d.result:self.project.annotations.append(d.result);self._refresh_lists();self.redraw()
    def add_image(self):
        fn=filedialog.askopenfilename(parent=self,title="Import image",filetypes=[("Images","*.png *.jpg *.jpeg *.webp *.svg"),("All files","*.*")])
        if not fn:return
        a=AnnotationSpec(kind="Image",image_path=fn,x=0,y=0,image_scale=.5,zorder=20)
        d=AnnotationDialog(self,a);self.wait_window(d)
        if d.result:self.project.annotations.append(d.result);self._refresh_lists();self.redraw()
    def edit_annotation(self):
        i=self._sel(self.ann_list)
        if i is None:return
        d=AnnotationDialog(self,self.project.annotations[i]);self.wait_window(d)
        if d.result:self.project.annotations[i]=d.result;self._refresh_lists();self.ann_list.selection_set(i);self.redraw()
    def duplicate_annotation(self):
        i=self._sel(self.ann_list)
        if i is None:return
        a=copy.deepcopy(self.project.annotations[i]);a.x+=.15;a.y+=.15;self.project.annotations.insert(i+1,a);self._refresh_lists();self.redraw()
    def delete_annotation(self):
        i=self._sel(self.ann_list)
        if i is not None:del self.project.annotations[i];self._refresh_lists();self.redraw()

    # Pair styles manager
    def manage_pair_styles(self):
        w=tk.Toplevel(self);w.title("Layer-pair connection styles");w.geometry("620x410");w.transient(self)
        lb=tk.Listbox(w,exportselection=False);lb.pack(fill="both",expand=True,padx=10,pady=10)
        def refresh():
            lb.delete(0,"end")
            for i,s in enumerate(self.project.pair_styles):lb.insert("end",f"{i:02d}  L{s.source_layer} → L{s.target_layer}  {s.mode}/{s.render}  {s.color or 'inherit'}")
        def add():
            d=PairStyleDialog(w,len(self.project.layers));w.wait_window(d)
            if d.result:self.project.pair_styles.append(d.result);refresh();self.redraw()
        def edit():
            sel=lb.curselection()
            if not sel:return
            i=sel[0];d=PairStyleDialog(w,len(self.project.layers),self.project.pair_styles[i]);w.wait_window(d)
            if d.result:self.project.pair_styles[i]=d.result;refresh();self.redraw()
        def dele():
            sel=lb.curselection()
            if sel:del self.project.pair_styles[sel[0]];refresh();self.redraw()
        bb=ttk.Frame(w);bb.pack(fill="x",padx=10,pady=(0,10));ttk.Button(bb,text="+ Add",command=add).pack(side="left",padx=2);ttk.Button(bb,text="Edit",command=edit).pack(side="left",padx=2);ttk.Button(bb,text="Delete",command=dele).pack(side="left",padx=2);ttk.Button(bb,text="Close",command=w.destroy).pack(side="right")
        lb.bind("<Double-1>",lambda e:edit());refresh()

    # Canvas dragging for annotations.
    def _canvas_press(self,event):
        if event.inaxes is None:return
        for artist,idx in reversed(self.renderer.annotation_artists):
            try:
                contains,_=artist.contains(event)
                if contains and not self.project.annotations[idx].locked:
                    self._drag_ann_index=idx;self._drag_last=(event.xdata,event.ydata);self.ann_list.selection_clear(0,"end");self.ann_list.selection_set(idx);return
            except Exception:pass
    def _canvas_motion(self,event):
        if self._drag_ann_index is None or event.xdata is None or event.ydata is None or self._drag_last is None:return
        dx=event.xdata-self._drag_last[0];dy=event.ydata-self._drag_last[1];a=self.project.annotations[self._drag_ann_index];a.x+=dx;a.y+=dy
        if a.kind.lower() in ("arrow","line","divider"):a.x2+=dx;a.y2+=dy
        self._drag_last=(event.xdata,event.ydata);self.renderer.draw(self.fig,self.project,preview=True);self.canvas.draw_idle();self.dirty=True
    def _canvas_release(self,event):
        if self._drag_ann_index is not None:self._refresh_lists()
        self._drag_ann_index=None;self._drag_last=None

    # Presets/reference
    def apply_preset(self,_event=None):
        if self.dirty and not messagebox.askyesno("Apply preset","Replace the current project with the selected preset?"):return
        self.project=PRESETS[self.preset_var.get()]();self._load_project_into_ui();self.dirty=False;self.redraw()
    def show_reference(self):
        ref=""
        if not ref:
            messagebox.showinfo("Reference image","This classic/custom preset has no external reference image.");return
        path=Path(ref);path=path if path.is_absolute() else BASE_DIR/path
        if not path.exists():messagebox.showerror("Reference image",f"Not found:\n{path}");return
        w=tk.Toplevel(self);w.title("Reference image — " + path.name);w.geometry("1000x700")
        from PIL import ImageTk
        img=Image.open(path).convert("RGB");img.thumbnail((960,650));photo=ImageTk.PhotoImage(img);lab=ttk.Label(w,image=photo);lab.image=photo;lab.pack(expand=True)

    # Auto-fit
    def autofit(self):
        try:
            self._read_ui_into_project();p=self.project;n=max(1,len(p.layers));maxnodes=max(l.visible_nodes for l in p.layers)
            top_caption_lines=max((1 + int(bool((l.note or '').strip())) + (1 if (p.show_nominal_size and l.size>0) else 0) + (1 if (p.show_activation and l.activation) else 0)) for l in p.layers)
            group_bonus=0.55 if p.groups else 0.0
            has_boxes=(p.show_input_items and p.input_items) or (p.show_output_items and p.output_items)
            # Width fit
            content_layers=max(1,n-1)
            p.layer_spacing=max(1.0,min(1.95,9.4/max(2,content_layers)))
            # Height fit
            p.vertical_spacing=max(0.24,min(0.58,3.6/max(4,maxnodes-1)))
            # Node size fit
            if n>=10 or maxnodes>=8:p.node_radius=min(max(p.node_radius,0.09),0.105)
            elif n>=7 or maxnodes>=6:p.node_radius=min(max(p.node_radius,0.10),0.118)
            else:p.node_radius=min(max(p.node_radius,0.11),0.135)
            # Slightly tighter margins unless annotations need room.
            p.margins=0.28 if not p.annotations else max(0.35,p.margins)
            # Keep the export aspect closer to the true content aspect.
            width_units=(content_layers*p.layer_spacing) + (1.2 if has_boxes else 0.0) + 2*p.margins + 1.2
            height_units=max(1.6,(maxnodes-1)*p.vertical_spacing + 1.35 + 0.30*top_caption_lines + group_bonus + 2*p.margins)
            desired_aspect=max(1.15,min(3.0,width_units/max(1e-6,height_units)))
            p.figure_height=max(5.8,min(9.2,p.figure_height))
            p.figure_width=max(8.5,min(16.0,p.figure_height*desired_aspect))
            for k in ["layer_spacing","vertical_spacing","node_radius","margins","figure_width","figure_height"]:self.vars[k].set(str(getattr(p,k)))
            self.redraw();self.status_var.set("Smart auto-fit applied")
        except Exception as exc:self._show_error(exc)

    # Project I/O
    def new_project(self):
        if self.dirty and not messagebox.askyesno("New project","Discard unsaved changes?"):return
        self.project=blank_custom();self.preset_var.set("Blank / Custom");self._load_project_into_ui();self.dirty=False;self.redraw()
    def open_project(self):
        fn=filedialog.askopenfilename(title="Open Neural Architecture Designer project",filetypes=[("NAD project package","*.nadproj"),("JSON project","*.json")])
        if not fn:return
        try:
            if Path(fn).suffix.lower()==".nadproj":p=self._load_package(fn)
            else:p=project_from_dict(json.loads(Path(fn).read_text(encoding="utf-8")))
            self.project=p;self._load_project_into_ui();self.dirty=False;self.redraw();self.status_var.set(f"Opened {Path(fn).name}")
        except Exception as exc:self._show_error(exc)
    def save_project(self):
        return self.save_project_as()
    def save_project_as(self):
        try:self._read_ui_into_project()
        except Exception as exc:self._show_error(exc);return
        fn=filedialog.asksaveasfilename(title="Save project package",defaultextension=".nadproj",filetypes=[("NAD project package","*.nadproj"),("JSON project","*.json")],initialfile="Neural_Architecture_Project.nadproj")
        if not fn:return
        try:
            if Path(fn).suffix.lower()==".json":Path(fn).write_text(json.dumps(asdict(self.project),indent=2,ensure_ascii=False),encoding="utf-8")
            else:self._save_package(fn)
            self.dirty=False;self.status_var.set(f"Saved {Path(fn).name}")
        except Exception as exc:self._show_error(exc)
    def _save_package(self,fn):
        p=copy.deepcopy(self.project);assets=[]
        for i,a in enumerate(p.annotations):
            if a.kind.lower()!="image" or not a.image_path:continue
            src=Path(a.image_path);src=src if src.is_absolute() else BASE_DIR/src
            if not src.exists():continue
            safe=f"asset_{i:03d}{src.suffix.lower()}";assets.append((src,safe));a.image_path=f"assets/{safe}"
        with zipfile.ZipFile(fn,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("project.json",json.dumps(asdict(p),indent=2,ensure_ascii=False))
            for src,safe in assets:z.write(src,f"assets/{safe}")
    def _load_package(self,fn):
        td=tempfile.mkdtemp(prefix="nadproj_");self.package_tempdir=td
        with zipfile.ZipFile(fn,"r") as z:z.extractall(td)
        p=project_from_dict(json.loads(Path(td,"project.json").read_text(encoding="utf-8")))
        for a in p.annotations:
            if a.kind.lower()=="image" and a.image_path and not Path(a.image_path).is_absolute():a.image_path=str((Path(td)/a.image_path).resolve())
        return p

    # Export
    def _export_fig(self,for_eps=False):
        self._read_ui_into_project();p=copy.deepcopy(self.project)
        if for_eps:
            # EPS does not support transparency. Blend global transparency into colors and render opaque.
            def blend(c,a,bg):
                try:
                    cr,cg,cb,_=to_rgba(c);br,bg_,bb,_=to_rgba(bg);return (cr*a+br*(1-a),cg*a+bg_*(1-a),cb*a+bb*(1-a),1)
                except Exception:return c
            p.connection_color=blend(p.connection_color,p.connection_alpha,p.background);p.connection_alpha=1.0
            for s in p.pair_styles:
                if s.color:
                    aa=p.connection_alpha if s.alpha<0 else s.alpha;s.color=blend(s.color,aa,p.background);s.alpha=1.0
            for l in p.layers:
                if l.fill_alpha<1:l.color=blend(l.color,l.fill_alpha,p.background);l.fill_alpha=1.0
            for a in p.annotations:a.alpha=1.0
        fig=plt.Figure(figsize=(p.figure_width,p.figure_height),dpi=p.dpi,facecolor=p.background);self.renderer.draw(fig,p,preview=False);return fig,p
    def export_figure(self):
        fn=filedialog.asksaveasfilename(title="Export figure",defaultextension=".png",filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("SVG","*.svg"),("PDF","*.pdf"),("EPS","*.eps")],initialfile="Neural_Network_Architecture.png")
        if not fn:return
        try:
            ext=Path(fn).suffix.lower();fig,p=self._export_fig(for_eps=ext==".eps");kw=dict(bbox_inches="tight",pad_inches=.05,facecolor=p.background)
            if ext in (".png",".jpg",".jpeg"):kw["dpi"]=p.dpi
            fig.savefig(fn,**kw);plt.close(fig);self.status_var.set(f"Exported {Path(fn).name}")
        except Exception as exc:self._show_error(exc)
    def export_all(self):
        folder=filedialog.askdirectory(title="Choose output folder")
        if not folder:return
        try:
            self._read_ui_into_project();base="Neural_Network_Architecture"
            for ext in [".png",".jpg",".svg",".pdf",".eps"]:
                fig,p=self._export_fig(for_eps=ext==".eps");kw=dict(bbox_inches="tight",pad_inches=.05,facecolor=p.background)
                if ext in (".png",".jpg"):kw["dpi"]=p.dpi
                fig.savefig(Path(folder)/(base+ext),**kw);plt.close(fig)
            Path(folder,(base+".json")).write_text(json.dumps(asdict(self.project),indent=2,ensure_ascii=False),encoding="utf-8")
            self._save_package(str(Path(folder)/(base+".nadproj")))
            self.status_var.set("Exported PNG/JPG/SVG/PDF/EPS + JSON + NADPROJ")
        except Exception as exc:self._show_error(exc)

    def _show_error(self,exc):
        traceback.print_exc();messagebox.showerror("Error",f"{type(exc).__name__}: {exc}")
    def on_close(self):
        if self.dirty and not messagebox.askyesno("Exit","Exit with unsaved changes?"):return
        try:
            if self.package_tempdir:shutil.rmtree(self.package_tempdir,ignore_errors=True)
        finally:self.destroy()


def main():
    App().mainloop()

if __name__=="__main__":
    main()
