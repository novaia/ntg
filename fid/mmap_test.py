import os
import numpy as np
import mmap

filename = 'mmap_file'
dtype = 'float32'

f = open(filename, 'w+b')
num_items = 50
dtype_size = np.dtype(dtype).itemsize
file_size = dtype_size * num_items
f.write(b"\0" * file_size)
print(f'Create file of size: {file_size}')

with open(filename, "r+b") as f:
    mm = mmap.mmap(f.fileno(), file_size)

    for i in range(num_items):
        mm[i * dtype_size : (i + 1) * dtype_size] = np.array([float(i)], dtype=dtype).tobytes()
    
    for i in range(num_items):
        arr = np.frombuffer(mm[i * dtype_size : (i + 1) * dtype_size], dtype=dtype)
        print(arr[0])