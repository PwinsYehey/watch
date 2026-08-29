#!/usr/bin/env python
"""Predict eye state (open/closed) for both eyes in a face photo. Usage: python cli/predict.py path/to/image.jpg"""

from frames.inference.predict import main

if __name__ == "__main__":
    main()
