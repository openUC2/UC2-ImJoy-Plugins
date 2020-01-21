#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:33:59 2020

@author: bene
"""
import numpy as np
import imagej

import imagej
ij = imagej.init('/Applications/Fiji.app')
#ij = imagej.init('sc.fiji:fiji:2.0.0-pre-10')
ij.getVersion()

#%%
macro = """
open("/Users/bene/Dropbox/Dokumente/Promotion/PROJECTS/STORMoChip/2020_01_13-EColi_TetraSpecs100_P20_Video_RAW_Fluct/E.Coli_2_my_raw_result-1.tif - drift corrected.tif");
run("Slice Keeper", "first=1 last=100 increment=1");

run("Estimate Drift", "time=1 max=50 reference=[first frame (default, better for fixed)] show_drift_plot choose=/Users/bene/Downloads/test.njt");
close();
close();
close();
run("Correct Drift", "choose=/Users/bene/Downloads/testDriftTable.njt");

run("SRRF Analysis", "ring=1 radiality_magnification=5 axes=6 frames_per_time-point=0 start=0 end=0 max=100 preferred=0");
saveAs("Tiff", "/Users/bene/Downloads/result.tif");
"""

result = ij.py.run_macro(macro)
