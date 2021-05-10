import os
import numpy as np
import intensity_normalization.normalize as norm
domain_name = 'Utrecht'
domain_dir = 'dataset/Utrecht'
split_path = 'path/'
patient_paths = os.listdir(domain_dir)

ws = getattr(norm, 'whitestripe')

# local_dir = '0064KW_flair_ws.nii.gz'
# local_dir2 = '0064KW_FL_preproc_25-29_ws.nii.gz'
for cp in patient_paths:
    current_patient = cp
    cp_dir = os.path.join(domain_dir, cp)
    output_dir = str(cp_dir) + '/pre'
    flair_dir = str(cp_dir) + '/pre/FLAIR.nii.gz'
    T1_dir = str(cp_dir) + '/pre/T1.nii.gz'
    ws.ws_normalize(flair_dir, 'FLAIR', output_dir=output_dir)
    ws.ws_normalize(T1_dir, 'T1', output_dir=output_dir)
