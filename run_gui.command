#!/bin/bash
# Double-click launcher for the cavidetect GUI (macOS).
# Opens the embolism (colour-shape) desktop window.
cd "$(dirname "$0")" || exit 1
exec python3 -m cavidetect.gui
