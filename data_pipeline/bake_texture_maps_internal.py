import bpy
import os

# Get the active object
obj = bpy.context.active_object

# Get the material of the object
mat = obj.data.materials[0]

# Enable nodes for the material
mat.use_nodes = True

# Create a texture node and link it to the material output
tex_node = mat.node_tree.nodes.new('ShaderNodeTexImage')
mat_output = mat.node_tree.nodes['Material Output']
mat.node_tree.links.new(tex_node.outputs['Color'], mat_output.inputs['Surface'])

# Create a new image with the desired resolution and assign it to the texture node
img = bpy.data.images.new(name=obj.name + '_Diffuse', width=4096, height=4096)
tex_node.image = img

# Select the texture node
tex_node.select = True
mat.node_tree.nodes.active = tex_node

# Bake the diffuse map
bpy.ops.object.bake(type='DIFFUSE', margin=16)

# Save the image to a file in the same folder as the blend file
img.filepath_raw = os.path.join(os.path.dirname(bpy.data.filepath), obj.name + '_Diffuse.png')
img.file_format = 'PNG'
img.save()

# Create a new image with the desired resolution and assign it to the texture node
img = bpy.data.images.new(name=obj.name + '_Normal', width=4096, height=4096)
tex_node.image = img

# Bake the normal map
bpy.ops.object.bake(type='NORMAL', margin=16)

# Save the image to a file in the same folder as the blend file
img.filepath_raw = os.path.join(os.path.dirname(bpy.data.filepath), obj.name + '_Normal.png')
img.file_format = 'PNG'
img.save()