#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Shared test set-up. The reaction inputs under reaction/input are not shipped
with the code, so a test that reads one is marked with the reaction it needs
and is skipped when that reaction folder is absent.

Created on: Tue Nov 03 2026

@author: Chien-Chai Chang (Francis)
@Reference: tReax_expedition SWS Monte-Carlo sampler

"""

#%% 0. Import required libraries
import os

import pytest

from treax_expedition.io_xyz import INPUT_ROOT

#%% 1. Reaction marker
def pytest_configure(config):

    config.addinivalue_line(
        "markers", "reaction(name): needs reaction/input/<name>; skipped when it is absent")

def pytest_runtest_setup(item):

    for Mark in item.iter_markers(name="reaction"):
        for Name in Mark.args:
            if not os.path.isdir(os.path.join(INPUT_ROOT, Name)):
                pytest.skip("reaction/input/%s is not present" % Name)
