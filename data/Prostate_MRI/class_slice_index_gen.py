import numpy as np
import os
import glob
import SimpleITK as sitk
import sys
import json
import niftiio as nio

IMG_BNAME="./prostate_MR_normalized/image_*.nii.gz"
SEG_BNAME="./prostate_MR_normalized/label_*.nii.gz"

imgs = glob.glob(IMG_BNAME)
segs = glob.glob(SEG_BNAME)
imgs = [ fid for fid in sorted(imgs, key = lambda x: int(x.split("_")[-1].split(".nii.gz")[0])  ) ]
segs = [ fid for fid in sorted(segs, key = lambda x: int(x.split("_")[-1].split(".nii.gz")[0])  ) ]

classmap = {}   # bladder, bone, obturator internus, transition zone, central gland, rectum, seminal vesicle and neurovascular bundle
LABEL_NAME = ["BG", "Bladder", "Bone", "OI", "TZ", "CG", "Rectum", "SV", "NB"]


MIN_TP = 1 # minimum number of positive label pixels to be recorded. Use >100 when training with manual annotations for more stable training

fid = f'./prostate_MR_normalized/classmap_{MIN_TP}.json' # name of the output file.
for _lb in LABEL_NAME:
    classmap[_lb] = {}
    for _sid in segs:
        pid = _sid.split("_")[-1].split(".nii.gz")[0]
        classmap[_lb][pid] = []

for seg in segs:
    pid = seg.split("_")[-1].split(".nii.gz")[0]
    lb_vol = nio.read_nii_bysitk(seg)
    n_slice = lb_vol.shape[0]
    for slc in range(n_slice):
        for cls in range(len(LABEL_NAME)):
            if cls in lb_vol[slc, ...]:
                if np.sum( lb_vol[slc, ...]) >= MIN_TP:
                    classmap[LABEL_NAME[cls]][str(pid)].append(slc)
    print(f'pid {str(pid)} finished!')
    
with open(fid, 'w') as fopen:
    json.dump(classmap, fopen)
    fopen.close()  
    

with open(fid, 'w') as fopen:
    json.dump(classmap, fopen)
    fopen.close()



