import os
import numpy as np

domain_name = 'Local'
domain_dir = 'dataset/Local'
split_path = 'path/'
patient_paths = os.listdir(domain_dir)
patient_paths.sort()

for cp in patient_paths:
    current_patient = cp
    cp_dir = os.path.join(domain_dir, cp)
    all_subject_path = os.listdir(cp_dir)

    split_dir = os.path.join(split_path, domain_name)
    s = cp_dir + ' ' + str(current_patient) + '\n'
    if np.random.random() < 0.8:
        split_train_dir = str(split_dir) + '/train.txt'
        with open(split_train_dir, 'a') as out:
            out.write(s)
    else:
        split_val_dir = str(split_dir) + '/val.txt'
        with open(split_val_dir, 'a') as out:
            out.write(s)
    split_test_dir = str(split_dir) + '/test.txt'
    with open(split_test_dir, 'a') as out:
        out.write(s)
