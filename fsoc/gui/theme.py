"""Mission-Control visual identity, typography, and QSS stylesheet.

Aesthetic Guidelines:
- Radar console / telemetry software / ground station interface.
- Near-black dark background, muted low-saturation surfaces, 1px hairline borders.
- Monospace font for all numeric readouts, coordinates, and status tokens.
- Restrained semantic color palette:
  * Green: Locked / Tracking
  * Amber: Degraded / Coasting
  * Red: Lost / Active Spiral Search
  * Cyan: Predicted State (Kalman state, not measured)
- Sharp corners (2-3px radius max), no gradients, no bubbly drop shadows.
"""

from PySide6.QtGui import QColor, QFont


# Semantic Palette Constants
COLOR_BG_DARK = "#090C10"       # Deep dark background
COLOR_SURFACE_1 = "#121721"     # Primary surface
COLOR_SURFACE_2 = "#18202D"     # Secondary panel surface
COLOR_SURFACE_INPUT = "#0D121B" # Input box background
COLOR_BORDER = "#212B3B"        # Hairline 1px border
COLOR_BORDER_LIGHT = "#313F55"  # Active border

COLOR_TEXT_PRIMARY = "#E6EDF3"   # Monospace / High contrast text
COLOR_TEXT_MUTED = "#8B949E"     # Secondary labels
COLOR_TEXT_DIM = "#505A69"       # Muted annotations

# Semantic Status Colors
COLOR_LOCKED = "#2ECC71"        # Emerald Green: Locked / Tracking
COLOR_DEGRADED = "#F39C12"      # Amber: Degraded / Coasting
COLOR_LOST = "#E74C3C"          # Red: Lost / Search
COLOR_PREDICTED = "#00D2FF"     # Cyan: Kalman predicted state
COLOR_HUD_RETICLE = "#3D4F66"   # Muted HUD crosshairs
COLOR_HUD_ACCENT = "#58A6FF"    # Accent indicator

# Font Helper Functions
def get_mono_font(size_pt: int = 10, bold: bool = False) -> QFont:
    """Return monospace font for telemetry and numeric readouts."""
    font = QFont("Consolas", size_pt)
    font.setStyleHint(QFont.Monospace)
    font.setBold(bold)
    return font


def get_label_font(size_pt: int = 9, bold: bool = False) -> QFont:
    """Return clean sans-serif font for labels and headings."""
    font = QFont("Segoe UI", size_pt)
    font.setStyleHint(QFont.SansSerif)
    font.setBold(bold)
    return font


# Global Mission-Control QSS Stylesheet
MISSION_CONTROL_STYLESHEET = f"""
QWidget {{
    background-color: {COLOR_BG_DARK};
    color: {COLOR_TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 11px;
}}

/* Top-level panels and frames */
QFrame#PanelFrame {{
    background-color: {COLOR_SURFACE_1};
    border: 1px solid {COLOR_BORDER};
    border-radius: 2px;
}}

QFrame#SectionFrame {{
    background-color: {COLOR_SURFACE_2};
    border: 1px solid {COLOR_BORDER};
    border-radius: 2px;
    padding: 6px;
}}

/* Headers and section titles */
QLabel#HeaderTitle {{
    color: {COLOR_TEXT_PRIMARY};
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
    font-family: 'Consolas', monospace;
}}

QLabel#SectionHeader {{
    color: {COLOR_TEXT_MUTED};
    font-size: 10px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding-bottom: 2px;
    border-bottom: 1px solid {COLOR_BORDER};
    margin-bottom: 4px;
}}

/* Push Buttons */
QPushButton {{
    background-color: {COLOR_SURFACE_2};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER_LIGHT};
    border-radius: 2px;
    padding: 5px 12px;
    font-family: 'Consolas', monospace;
    font-size: 11px;
    font-weight: bold;
}}

QPushButton:hover {{
    background-color: #202B3D;
    border-color: {COLOR_HUD_ACCENT};
}}

QPushButton:pressed {{
    background-color: #101620;
    border-color: {COLOR_LOCKED};
}}

QPushButton:disabled {{
    background-color: #0E131C;
    color: {COLOR_TEXT_DIM};
    border-color: {COLOR_BORDER};
}}

QPushButton#BtnRun {{
    background-color: #12281E;
    color: {COLOR_LOCKED};
    border: 1px solid #1E4632;
}}
QPushButton#BtnRun:hover {{
    background-color: #183729;
    border-color: {COLOR_LOCKED};
}}

QPushButton#BtnPause {{
    background-color: #2B2313;
    color: {COLOR_DEGRADED};
    border: 1px solid #4D3B18;
}}
QPushButton#BtnPause:hover {{
    background-color: #382D18;
    border-color: {COLOR_DEGRADED};
}}

QPushButton#BtnReset {{
    background-color: #281616;
    color: {COLOR_LOST};
    border: 1px solid #4A2222;
}}
QPushButton#BtnReset:hover {{
    background-color: #381C1C;
    border-color: {COLOR_LOST};
}}

/* ComboBox */
QComboBox {{
    background-color: {COLOR_SURFACE_INPUT};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 2px;
    padding: 3px 8px;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}}

QComboBox:hover {{
    border-color: {COLOR_BORDER_LIGHT};
}}

QComboBox::drop-down {{
    border: none;
    width: 18px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLOR_SURFACE_1};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER_LIGHT};
    selection-background-color: #1E2838;
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 4px;
    background: {COLOR_SURFACE_INPUT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 1px;
}}

QSlider::sub-page:horizontal {{
    background: {COLOR_HUD_ACCENT};
}}

QSlider::handle:horizontal {{
    background: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER_LIGHT};
    width: 10px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLOR_LOCKED};
}}

/* CheckBoxes */
QCheckBox {{
    color: {COLOR_TEXT_PRIMARY};
    font-size: 11px;
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 12px;
    height: 12px;
    background-color: {COLOR_SURFACE_INPUT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 2px;
}}

QCheckBox::indicator:checked {{
    background-color: {COLOR_HUD_ACCENT};
    border-color: {COLOR_HUD_ACCENT};
}}

QCheckBox::indicator:hover {{
    border-color: {COLOR_HUD_ACCENT};
}}

/* SpinBoxes */
QSpinBox, QDoubleSpinBox {{
    background-color: {COLOR_SURFACE_INPUT};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 2px;
    padding: 3px 6px;
    font-family: 'Consolas', monospace;
    font-size: 11px;
}}

/* ScrollBars */
QScrollBar:vertical {{
    background: {COLOR_BG_DARK};
    width: 6px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {COLOR_BORDER_LIGHT};
    min-height: 20px;
    border-radius: 2px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""
