import os
import pathlib
import sys

project_root = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
if project_root not in sys.path: sys.path.append(project_root)
print(f'project root: {project_root}')
print(os.getcwd())
os.chdir(project_root)
print(os.getcwd())

data_path = 'data/images'
print(os.path.isdir(data_path))

import fid
inception = fid.get_inception_model()