#pylint: disable=C0103, C0411
"""
gen_sinos
"""
import numpy as np
import os
from sino_utils import get_op, space
from utils import list_files, get_image


ANGLES = 48
PROJS = 800


out_path_test = '/home/hzyuan/CT/AAPM_dataset/sinos_test'
files_test = list_files('/home/hzyuan/CT/AAPM_test/', name='*.npy')
out_path = '/home/hzyuan/CT/AAPM_dataest/sinos'
files = list_files('/home/hzyuan/CT/AAPM_dataset/', name='*.npy')

if not os.path.exists(out_path):
    os.makedirs(out_path)
if not os.path.exists(out_path_test):
    os.makedirs(out_path)

def gen_sinos(fs, op):
    """
    Get the projection sinogram image of given phantom files.
    Args:
        fs: .npy files containing the CT image.
        op: output path.
    Returns:
        None
    """
    count = 0
    o1, _ = get_op(ANGLES, PROJS)
    o2, _ = get_op(ANGLES * 2, PROJS)
    o3, _ = get_op(ANGLES * 4, PROJS)
    for file in fs:
        if 'L286' in file:
            continue
        print(count)
        name = file.split('/')[-1].split('_')[0].split('-')[-1]
        sino1 = o1(space.element(np.load(file)/1000.0-1.0))
        sino2 = o2(space.element(get_image(file)/1000.0-1.0))
        sino3 = o3(space.element(np.load(file)/1000.0-1.0))
        name1 = os.path.join(op, '%d_%s_%d.npy'%(ANGLES, name, count))
        name2 = os.path.join(op, '%d_%s_%d.npy'%(ANGLES * 2, name, count))
        name3 = os.path.join(op, '%d_%s_%d.npy'%(ANGLES * 4, name, count))
        np.save(name1, sino1)
        np.save(name2, sino2)
        np.save(name3, sino3)
        count += 1
        print(file)

if __name__ == "__main__":
    gen_sinos(files, out_path)
    gen_sinos(files_test, out_path_test)
