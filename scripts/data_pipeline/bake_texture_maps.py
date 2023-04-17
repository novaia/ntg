"""
Opens a .blend file and executes code to bake texture maps from procedural material.
"""

import bpy
import os

# Define the path of the blend file to open
blend_file = os.path.join(os.path.dirname(bpy.data.filepath), 'example.blend')

# Define the path of the script to execute
script_file = os.path.join(os.path.dirname(bpy.data.filepath), 'bake_export.py')

# Define a function that executes the script after opening the blend file
def execute_script(scene):
    # Unregister the handler
    bpy.app.handlers.load_post.remove(execute_script)
    # Execute the script
    exec(compile(open(script_file).read(), script_file, 'exec'))

# Register the function as a persistent handler
bpy.app.handlers.load_post.append(execute_script)

# Open the blend file
bpy.ops.wm.open_mainfile(filepath=blend_file)