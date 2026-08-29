#!/usr/bin/env python
"""Audit raw datasets before splitting/training. Run this first after downloading data into data/raw/."""

from frames.data.inspect import main

if __name__ == "__main__":
    main()
