"""FSOC Coarse-Alignment Virtual Tracking System — Research Demo.

Smart India Hackathon (SIH) Technical Submission for ISRO / Dept. of Space.
Simulates coarse-pointing optical beacon tracking for Free-Space Optical
Communication (FSOC) terminal prior to fine-pointing / FSM stages.

Usage:
    python main.py
"""

import sys
import os

# Ensure project root is in Python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from fsoc.gui.main_window import MainWindow


def main():
    """Main application entry point."""
    # Configure high-DPI display attributes
    app = QApplication(sys.argv)
    app.setApplicationName("FSOC Virtual Tracking System — ISRO Technical Demo")
    app.setOrganizationName("ISRO-SIH")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
