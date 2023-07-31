import sys
print(sys.path)
#sys.path.append('../')
#print(sys.path)
import os
print(f'cwd: {os.getcwd()}')
print(f'cwd abspath: {os.path.abspath(os.getcwd())}')
cwd_abspath = os.path.abspath(os.getcwd())
#sys.path.append(cwd_abspath)
import pathlib
print(f'file dir: {pathlib.Path(__file__).parent.resolve()}')
print(f'file dir parent: {pathlib.Path(__file__).parent.parent.resolve()}')
file_parent_dir = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(file_parent_dir)
print(sys.path)
#import fid