import pathlib
import sys
project_root = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
if project_root not in sys.path: sys.path.append(project_root)
print(sys.path)

import fid

import os
test_path = '/data/images'
print(test_path)
print(os.path.isdir(test_path))
print(os.path.abspath(test_path))
print(os.path.isdir(os.path.abspath(test_path)))

print(os.getcwd())
os.chdir('../')
print(os.getcwd())