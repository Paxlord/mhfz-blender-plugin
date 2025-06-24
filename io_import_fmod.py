bl_info = {
    "name": "FMOD Importer",
    "blender": (3, 0, 0),
    "category": "Import-Export",
    "author": "Pax",
    "version": (1, 0),
    "description": "Import FMOD Mhfz model format into Blender",
    "location": "File > Import > FMOD (.fmod)",
}

import bpy
import bpy_extras.io_utils
import bmesh
import struct
import os
from dataclasses import dataclass
from mathutils import Matrix, Vector, Euler
import math
import random

def read_uint32_le(buf, off): 
    return struct.unpack_from("<I", buf, off)[0]

def read_uint16_le(buf, off):
    return struct.unpack_from("<H", buf, off)[0]

def read_uint8(buf, off):
    return struct.unpack_from("<B", buf, off)[0]

def read_float_le(buf, off):
    return struct.unpack_from("<f", buf, off)[0]

def write_uint32_le(value: int) -> bytes:
    return struct.pack("<I", value)

def write_uint16_le(value: int) -> bytes:
    return struct.pack("<H", value)

def write_uint8(value: int) -> bytes:
    return struct.pack("<B", value)

def write_float_le(value: float) -> bytes:
    return struct.pack("<f", value)



@dataclass
class DirectoryHeader:
    type: int
    file_count: int
    directory_size: int 

@dataclass
class Vertex:
    pos: tuple[float, float, float]
    normal: tuple[float, float, float]
    uv: tuple[float, float]
    color: tuple[float, float, float, float]

@dataclass
class Mesh:
    name: str = "DefaultMesh"
    vertices: list[Vertex] | None = None
    indices: list[int] | None = None
    material_indices: list[int] | None = None

@dataclass
class ParsedMeshData:
    faces: list[list[int]] | None = None
    materials: list[int] | None = None
    vertices: list[tuple[float, float, float]] | None = None
    normals: list[tuple[float, float, float]] | None = None
    uvs: list[tuple[float, float]] | None = None
    colors: list[tuple[float, float, float, float]] | None = None
    material_list: list[int] | None = None
    material_indices: list[int] | None = None
    weights: list[list[tuple[int, int]]] | None = None
    bones_list: list[int] | None = None
    tpn_vec: list[tuple[float, float, float, float]] | None = None

@dataclass
class ParsedTextureData:
    image_idx: int | None = None #image index in the image folder
    width: int | None = None
    height: int | None = None
    data: bytes | None = None #always 244 bytes of zeros

@dataclass
class ParsedMaterialData:
    ambient_rgba: tuple[float, float, float, float] | None = None
    diffuse_rgba: tuple[float, float, float, float] | None = None
    specular_rgba: tuple[float, float, float, float] | None = None
    specular_str: float | None = None
    texture_count: int | None = None
    unknown_data: bytes | None = None
    texture_diffuse: int | None = None
    texture_normal: int | None = None
    texture_specular: int | None = None

@dataclass
class ParsedFMODData:
    meshes: list[ParsedMeshData] | None = None
    textures: list[ParsedTextureData] | None = None
    materials: list[ParsedMaterialData] | None = None

@dataclass
class ParsedBoneData:
    node_id: int | None = None
    parent_id: int | None = None
    child_id: int | None = None
    sibling_id: int | None = None
    scale: tuple[float, float, float] | None = None #Always 1.0,1.0,1.0
    rotation: tuple[float, float, float] | None = None #Always 1.0,0.0,0.0
    ik_influence_weight: float | None = None #Always 0.0
    ik_blend_param: float | None = None #always 1.0
    translation: tuple[float, float, float] | None = None #Relative to the parent bone
    transform_flags: float | None = None #Always 1.0
    unk_int: int | None = None #Always 0xFFFFFFFF
    unknown_bone_param: int | None = None #can be 0,1,2,3 not sure what it does
    unk_data: bytes | None = None #zeroes

@dataclass
class ParsedFSKLData:
    root_indices: list[int] | None = None
    bones: list[ParsedBoneData] | None = None

def find_texture_folder(fmod_path: str) -> str:
    # Look if we have any directory present in the fmod path folder
    fmod_dir = os.path.dirname(fmod_path)
    items = os.listdir(fmod_dir)
    directories = [item for item in items if os.path.isdir(os.path.join(fmod_dir, item))]
    if directories:
        # If we have directories, return the one that start with "0003"
        for directory in directories:
            if directory.startswith("0003"):
                return os.path.join(fmod_dir, directory)
    
    # If there's not check if the fmod directory name starts with "0001"
    if os.path.basename(fmod_dir).startswith("0001"):
        # if so look one directory up for a directory that starts with "0002"
        parent_dir = os.path.dirname(fmod_dir)
        parent_items = os.listdir(parent_dir)
        parent_directories = [item for item in parent_items if os.path.isdir(os.path.join(parent_dir, item))]
        for directory in parent_directories:
            if directory.startswith("0002"):
                return os.path.join(parent_dir, directory)
    
    # If the fmod directory starts with "0004" instead look for a directory that starts with "0005"
    if os.path.basename(fmod_dir).startswith("0004"):
        parent_dir = os.path.dirname(fmod_dir)
        parent_items = os.listdir(parent_dir)
        parent_directories = [item for item in parent_items if os.path.isdir(os.path.join(parent_dir, item))]
        for directory in parent_directories:
            if directory.startswith("0005"):
                return os.path.join(parent_dir, directory)

    return None

def find_fskl_file(fmod_path: str) -> str:
    #look for a .fskl file in the same directory as the fmod file
    fmod_dir = os.path.dirname(fmod_path)
    items = os.listdir(fmod_dir)
    for item in items:
        if item.endswith(".fskl"):
            return os.path.join(fmod_dir, item)
    return None

def triangulate_strips(strips: list[list[int]], material_indices: list[int], material_list: list[int]) -> list[int]:
    triangle_indices = []
    triangle_materials = []

    for strip_idx, strip in enumerate(strips):
        if len(strip) < 3:
            continue
        
        #material from the mesh palette
        material_idx = material_indices[strip_idx]
        global_material_idx = material_list[material_idx]

        for i in range(2, len(strip)):
            v0 = strip[i - 2]
            v1 = strip[i - 1]
            v2 = strip[i]

            if v0 == v1 or v1 == v2 or v0 == v2:
                continue

            if i%2 == 0:
                triangle_indices.extend([v0, v1, v2])
            else:
                triangle_indices.extend([v0, v2, v1])

            triangle_materials.append(global_material_idx)
    
    return triangle_indices, triangle_materials

def create_optimized_strips(triangles: list[list[int]], material_indices: list[int], material_list: list[int]) -> tuple[list[list[int]], list[int]]:
    """
    Convert triangle faces to optimized strips for FMOD export.
    Tries to create longer strips where possible.
    
    Args:
        triangles: List of triangles, each containing 3 vertex indices
        material_indices: List of material indices for each triangle
        material_list: The material palette to reference
        
    Returns:
        tuple: (strips, strip_material_indices)
            - strips: List of strips, where each strip is a list of vertex indices
            - strip_material_indices: List of material indices for each strip (indices into material_list)
    """
    # Group triangles by material
    material_groups: dict[int, list[tuple[int, list[int]]]] = {}
    for i, tri in enumerate(triangles):
        mat_idx = material_indices[i]
        if mat_idx not in material_groups:
            material_groups[mat_idx] = []
        material_groups[mat_idx].append((i, tri))
    
    strips: list[list[int]] = []
    strip_material_indices: list[int] = []
    
    # For each material, create strips
    for mat_idx, tri_list in material_groups.items():
        # Find or add the material to the palette
        if mat_idx in material_list:
            palette_idx = material_list.index(mat_idx)
        else:
            material_list.append(mat_idx)
            palette_idx = len(material_list) - 1
        
        # Build adjacency information - which triangles share edges
        adjacency: dict[int, list[tuple[int, set[int]]]] = {}
        for i, (idx1, tri1) in enumerate(tri_list):
            adjacency[idx1] = []
            
            for j, (idx2, tri2) in enumerate(tri_list):
                if idx1 == idx2:
                    continue
                
                # Check how many vertices they share
                shared: set[int] = set(tri1) & set(tri2)
                if len(shared) == 2:  # They share an edge
                    adjacency[idx1].append((idx2, shared))
        
        # Track which triangles have been used
        used_triangles: set[int] = set()
        
        # Process triangles until all are used
        while len(used_triangles) < len(tri_list):
            # Find unused triangle with the most adjacent triangles to start
            max_adj: int = -1
            best_start: int | None = None
            
            for idx, _ in tri_list:
                if idx not in used_triangles:
                    adj_count: int = sum(1 for adj_idx, _ in adjacency[idx] if adj_idx not in used_triangles)
                    if adj_count > max_adj:
                        max_adj = adj_count
                        best_start = idx
            
            if best_start is None:
                break  # All triangles used
            
            # Get the start triangle
            start_tri: list[int] = next(tri for idx, tri in tri_list if idx == best_start)
            
            # Start a new strip with this triangle
            strip: list[int] = list(start_tri)
            used_triangles.add(best_start)
            
            # Try to extend the strip as long as possible
            current_idx: int = best_start
            while True:
                # Find best next triangle (one with most unused neighbors)
                best_next: int | None = None
                best_next_shared: set[int] | None = None
                max_next_adj: int = -1
                
                for adj_idx, shared in adjacency[current_idx]:
                    if adj_idx not in used_triangles:
                        adj_count: int = sum(1 for next_idx, _ in adjacency[adj_idx] 
                                          if next_idx not in used_triangles and next_idx != current_idx)
                        
                        if adj_count > max_next_adj:
                            max_next_adj = adj_count
                            best_next = adj_idx
                            best_next_shared = shared
                
                if best_next is None:
                    break  # No more adjacent triangles
                
                # Add the new vertex to the strip
                next_tri: list[int] = next(tri for idx, tri in tri_list if idx == best_next)
                new_vertex: int = next(v for v in next_tri if v not in best_next_shared)
                
                # Get the two vertices from the shared edge
                shared_list: list[int] = list(best_next_shared)
                
                # Check if the last two vertices in the strip match the shared edge
                if set(strip[-2:]) == best_next_shared:
                    # Just add the new vertex
                    strip.append(new_vertex)
                else:
                    # Need to handle special case - this usually means we have a complex connection
                    # For simplicity, we'll just start a new strip
                    break
                
                used_triangles.add(best_next)
                current_idx = best_next
            
            # Add the completed strip
            strips.append(strip)
            strip_material_indices.append(palette_idx)
    
    # Try to merge strips with the same material
    merged = True
    while merged:
        merged = False
        
        for i in range(len(strips)):
            if merged:
                break
                
            for j in range(i + 1, len(strips)):
                if strip_material_indices[i] != strip_material_indices[j]:
                    continue
                
                # Check if strips can be merged
                if set(strips[i][-2:]) == set(strips[j][:2]):
                    # Merge j into i
                    strips[i].extend(strips[j][2:])
                    strips.pop(j)
                    strip_material_indices.pop(j)
                    merged = True
                    break
                
                elif set(strips[j][-2:]) == set(strips[i][:2]):
                    # Merge i into j
                    strips[j].extend(strips[i][2:])
                    strips[i] = strips[j]
                    strips.pop(j)
                    strip_material_indices.pop(j)
                    merged = True
                    break
    
    return strips, strip_material_indices

def parent_mesh_to_armature(mesh_obj, armature_obj):
    """Parent a mesh object to an armature object and add an armature modifier."""
    # Add an armature modifier
    modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    modifier.object = armature_obj
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False
    
    # Parent the mesh to the armature
    mesh_obj.parent = armature_obj
    
    return mesh_obj

def create_blender_mesh(obj_name: str, mesh_name: str, parsed_data: ParsedMeshData, blender_materials: list[bpy.types.Material] = None, scale_factor=0.01):
    # (Existing code to create the mesh)
    if not parsed_data.faces:
        print("No faces found in the mesh data")
        return None
    if not parsed_data.vertices:
        print("No vertices found in the mesh data")
        return None
    
    vertices = parsed_data.vertices
    indices, material_indices = triangulate_strips(parsed_data.faces, parsed_data.material_indices, parsed_data.material_list)
    uvs = parsed_data.uvs
    normals = parsed_data.normals
    colors = parsed_data.colors

    mesh_data = bpy.data.meshes.new(mesh_name)
    if len(indices) % 3 != 0:
        print("Indices count is not a multiple of 3, cannot create mesh")
        return None
    num_faces = len(indices) // 3
    faces = [indices[i*3:i*3 + 3] for i in range(num_faces)]

    mesh_data.from_pydata(vertices, [], faces)

    if blender_materials:
        for mat in blender_materials:
            mesh_data.materials.append(mat)
        for face_idx, poly in enumerate(mesh_data.polygons):
            if face_idx < len(material_indices):
                mat_idx = material_indices[face_idx]
                if mat_idx < len(blender_materials):
                    poly.material_index = mat_idx

    if uvs:
        uv_layer = mesh_data.uv_layers.new(name="UVMap")
        for poly in mesh_data.polygons:
            for loop_index in poly.loop_indices:
                vertex_index = mesh_data.loops[loop_index].vertex_index
                if vertex_index < len(uvs):
                    uv_layer.data[loop_index].uv = uvs[vertex_index]

    if colors and len(colors) == len(vertices):
        color_layer = mesh_data.vertex_colors.new(name="VertexColor")
        for poly in mesh_data.polygons:
            for loop_index in poly.loop_indices:
                vertex_index = mesh_data.loops[loop_index].vertex_index
                if vertex_index < len(colors):
                    color_tuple = colors[vertex_index]
                    if len(color_tuple) == 3:
                        color_tuple = (color_tuple[0], color_tuple[1], color_tuple[2], 1.0)
                    elif len(color_tuple) != 4:
                        print(f"Warning: Invalid color tuple size for vert {vertex_index}. Skipping.")
                        continue
                    color_layer.data[loop_index].color = color_tuple
    else:
        print("Skipping vertex colors.")

    if normals and len(normals) == len(vertices):
        try:
            safe_normals = [n for n in normals]
            mesh_data.vertices.foreach_set("normal", [co for n in safe_normals for co in n])
        except:
            mesh_data.use_auto_smooth = False
            mesh_data.validate(verbose=False)
            mesh_data.update(calc_edges=True)
    else:
        mesh_data.validate(verbose=False)
        mesh_data.update(calc_edges=True)

    print(f"Creating object '{obj_name}' with mesh '{mesh_name}'")
    obj = bpy.data.objects.new(obj_name, mesh_data)
    bpy.context.scene.collection.objects.link(obj)
    
        # Modified weight handling section
    if parsed_data.weights:
        print(f"Applying weights to mesh '{mesh_name}' with {len(parsed_data.weights)} weighted vertices")
        
        if parsed_data.bones_list:
            # Case 1: We have both weights and a bone_list, use the bone_list to map indices
            print(f"Using bone_list mapping with {len(parsed_data.bones_list)} bones")
            
            # Create vertex groups for each bone ID in the bone_list
            for bone_id in parsed_data.bones_list:
                bone_name = f"Bone_{bone_id}"
                obj.vertex_groups.new(name=bone_name)
            
            # Assign weights to vertex groups using the bone_list mapping
            for vertex_idx, weight_data in enumerate(parsed_data.weights):
                for bone_idx, weight_value in weight_data:
                    if bone_idx < len(parsed_data.bones_list):
                        bone_id = parsed_data.bones_list[bone_idx]
                        bone_name = f"Bone_{bone_id}"
                        # Normalize weight from 0-100 to 0-1
                        normalized_weight = weight_value / 100.0
                        obj.vertex_groups[bone_name].add([vertex_idx], normalized_weight, 'REPLACE')
        else:
            # Case 2: We have weights but no bone_list, use bone_idx directly as the global ID
            print("No bone_list found, using bone indices directly as global bone IDs")
            
            # Find all unique bone indices used in the weights
            used_bone_indices = set()
            for weight_data in parsed_data.weights:
                for bone_idx, _ in weight_data:
                    used_bone_indices.add(bone_idx)
            
            # Create vertex groups for each unique bone index
            for bone_idx in used_bone_indices:
                bone_name = f"Bone_{bone_idx}"
                obj.vertex_groups.new(name=bone_name)
            
            # Assign weights to vertex groups using bone_idx directly
            for vertex_idx, weight_data in enumerate(parsed_data.weights):
                for bone_idx, weight_value in weight_data:
                    bone_name = f"Bone_{bone_idx}"
                    # Normalize weight from 0-100 to 0-1
                    normalized_weight = weight_value / 100.0
                    obj.vertex_groups[bone_name].add([vertex_idx], normalized_weight, 'REPLACE')

    return obj

def create_blender_material(mat_name: str, parsed_material: ParsedMaterialData, texture_dic: dict[int, bpy.types.Image]):
    mat_name = f"Mat_{mat_name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    mat.blend_method = 'HASHED'
    mat.shadow_method = 'HASHED'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)
    
    principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled_bsdf.location = (0, 0)

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)

    links.new(principled_bsdf.outputs['BSDF'], output.inputs['Surface'])

    if parsed_material.diffuse_rgba:
        principled_bsdf.inputs['Base Color'].default_value = parsed_material.diffuse_rgba
    
    # if parsed_material.specular_rgba:
    #     principled_bsdf.inputs['Tint'].default_value = parsed_material.specular_rgba
    
    # if parsed_material.specular_str:
    #     principled_bsdf.inputs['Roughness'].default_value = parsed_material.specular_str / 100.0

    # if parsed_material.ambient_rgba:
    #     principled_bsdf.inputs['Emission'].default_value = parsed_material.ambient_rgba

    # handling diffuse texture
    if parsed_material.texture_diffuse is not None:
        diffuse_texture = nodes.new(type='ShaderNodeTexImage')
        diffuse_texture.location = (-400, 0)
        diffuse_texture.image = texture_dic.get(parsed_material.texture_diffuse)
        links.new(diffuse_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])

    # handling normal texture
    if parsed_material.texture_normal is not None:
        normal_texture = nodes.new(type='ShaderNodeTexImage')
        normal_texture.location = (-400, -200)
        normal_texture.image = texture_dic.get(parsed_material.texture_normal)
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-200, -200)
        normal_texture.image.colorspace_settings.name = 'Non-Color'
        links.new(normal_texture.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled_bsdf.inputs['Normal'])

    # handling specular texture
    if parsed_material.texture_specular is not None:
        specular_texture = nodes.new(type='ShaderNodeTexImage')
        specular_texture.location = (-400, -400)
        specular_texture.image = texture_dic.get(parsed_material.texture_specular)
        specular_texture.image.colorspace_settings.name = 'Non-Color'
        
        if 'Specular' in principled_bsdf.inputs:
            links.new(specular_texture.outputs['Color'], principled_bsdf.inputs['Specular'])
        elif 'Specular Tint' in principled_bsdf.inputs:
            links.new(specular_texture.outputs['Color'], principled_bsdf.inputs['Specular Tint'])
        elif 'Specular IOR Level' in principled_bsdf.inputs:
            links.new(specular_texture.outputs['Color'], principled_bsdf.inputs['Specular IOR Level'])
        
        invert = nodes.new(type='ShaderNodeInvert')
        invert.location = (-200, -400)
        links.new(specular_texture.outputs['Color'], invert.inputs['Color'])
        links.new(invert.outputs['Color'], principled_bsdf.inputs['Roughness'])

    return mat

def create_armature(armature_name: str, skeleton_data: ParsedFSKLData, scale_factor=0.01):
    # Create a new armature and object
    arm_data = bpy.data.armatures.new(armature_name)
    armature_obj = bpy.data.objects.new(armature_name, arm_data)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bone_length = 100 # constant length for all bones

    # Build a mapping for quick lookup by node_id
    bone_map = {bone.node_id: bone for bone in skeleton_data.bones}
    global_positions = {}

    def get_global_position(bone):
        if bone.node_id in global_positions:
            return global_positions[bone.node_id]
        local = Vector(bone.translation) if bone.translation else Vector((0, 0, 0))
        if bone.parent_id is not None and bone.parent_id != -1:
            parent_bone = bone_map.get(bone.parent_id)
            if parent_bone:
                pos = get_global_position(parent_bone) + local
            else:
                pos = local
        else:
            pos = local
        global_positions[bone.node_id] = pos
        return pos

    # Create edit bones using computed global positions
    edit_bones = {}
    for bone in skeleton_data.bones:
        global_head = get_global_position(bone)
        edit_bone = arm_data.edit_bones.new(f"Bone_{bone.node_id}")
        edit_bone.head = global_head
        edit_bone.tail = global_head + Vector((0, bone_length, 0))
        edit_bones[bone.node_id] = edit_bone

    # Set up parent-child relationships
    for bone in skeleton_data.bones:
        if bone.parent_id is not None and bone.parent_id != -1:
            parent_edit_bone = edit_bones.get(bone.parent_id)
            if parent_edit_bone:
                edit_bones[bone.node_id].parent = parent_edit_bone

    bpy.ops.object.mode_set(mode='OBJECT')
    
    for bone in skeleton_data.bones:
        bone_name = f"Bone_{bone.node_id}"
        armature_bone = armature_obj.data.bones.get(bone_name)
        if armature_bone:
            armature_bone["unknown_bone_param"] = bone.unknown_bone_param

    # Apply rotation (-90° on X) and scaling
    #armature_obj.rotation_euler = Euler((math.radians(90), 0, 0), 'XYZ')
    #armature_obj.scale = (scale_factor, scale_factor, scale_factor)
    
    return armature_obj

def parse_directory_header(buf, off):
    type = read_uint32_le(buf, off)
    file_count = read_uint32_le(buf, off + 4)
    directory_size = read_uint32_le(buf, off + 8)
    return DirectoryHeader(type, file_count, directory_size)

def parse_face_block(buf, off):
    face_block_header = parse_directory_header(buf, off)
    strips = []
    cur_off = off + 12
    for i in range(face_block_header.file_count):
        strips_block_header = parse_directory_header(buf, cur_off)
        cur_off += 12
        for j in range(strips_block_header.file_count):
            strip_size = read_uint32_le(buf, cur_off) & 0xFFFFFFF
            cur_off += 4
            strip = []
            for k in range(strip_size):
                strip.append(read_uint32_le(buf, cur_off))
                cur_off += 4
            strips.append(strip)

    return strips

def parse_material_list_block(buf, off):
    material_list_block_header = parse_directory_header(buf, off)
    materials = []
    cur_off = off + 12
    for i in range(material_list_block_header.file_count):
        material = read_uint32_le(buf, cur_off)
        materials.append(material)
        cur_off += 4
    return materials

def parse_material_indices_block(buf, off):
    material_indices_block_header = parse_directory_header(buf, off)
    material_indices = []
    cur_off = off + 12
    for i in range(material_indices_block_header.file_count):
        material_index = read_uint32_le(buf, cur_off)
        material_indices.append(material_index)
        cur_off += 4
    return material_indices

def parse_vertex_block(buf, off):
    vertex_block_header = parse_directory_header(buf, off)
    vertices = []
    cur_off = off + 12
    for i in range(vertex_block_header.file_count):
        vertex = (
            read_float_le(buf, cur_off),
            read_float_le(buf, cur_off + 4),
            read_float_le(buf, cur_off + 8)
        )
        vertices.append(vertex)
        cur_off += 12
    return vertices

def parse_normal_block(buf, off):
    normal_block_header = parse_directory_header(buf, off)
    normals = []
    cur_off = off + 12
    for i in range(normal_block_header.file_count):
        normal = (
            read_float_le(buf, cur_off),
            read_float_le(buf, cur_off + 4),
            read_float_le(buf, cur_off + 8)
        )
        normals.append(normal)
        cur_off += 12
    return normals

def parse_uv_block(buf, off):
    uv_block_header = parse_directory_header(buf, off)
    uvs = []
    cur_off = off + 12
    for i in range(uv_block_header.file_count):
        uv = (
            read_float_le(buf, cur_off),
            1.0-read_float_le(buf, cur_off + 4)
        )
        uvs.append(uv)
        cur_off += 8
    return uvs

def parse_color_block(buf, off):
    color_block_header = parse_directory_header(buf, off)
    colors = []
    cur_off = off + 12
    for i in range(color_block_header.file_count):
        color = (
            read_float_le(buf, cur_off),
            read_float_le(buf, cur_off + 4),
            read_float_le(buf, cur_off + 8),
            read_float_le(buf, cur_off + 12)
        )
        colors.append(color)
        cur_off += 16
    return colors

def parse_vertex_weight_block(buf, off):
    vertex_weight_block_header = parse_directory_header(buf, off)
    weights = []
    cur_off = off + 12
    for i in range(vertex_weight_block_header.file_count):
        weight_count = read_uint32_le(buf, cur_off)
        cur_off += 4
        weight_list = []
        for j in range(weight_count):
            bone_index = read_uint32_le(buf, cur_off)
            weight_value = read_float_le(buf, cur_off + 4)
            weight_list.append((bone_index, weight_value))
            cur_off += 8
        weights.append(weight_list)
    return weights

def parse_bone_list_block(buf, off):
    bone_list_block_header = parse_directory_header(buf, off)
    bones = []
    cur_off = off + 12
    for i in range(bone_list_block_header.file_count):
        bone = read_uint32_le(buf, cur_off)
        bones.append(bone)
        cur_off += 4
    return bones

def parse_tpn_vec_block(buf, off):
    tpn_vec_block_header = parse_directory_header(buf, off)
    tpn_vec = []
    cur_off = off + 12
    for i in range(tpn_vec_block_header.file_count):
        vec = (
            read_float_le(buf, cur_off),
            read_float_le(buf, cur_off + 4),
            read_float_le(buf, cur_off + 8),
            read_float_le(buf, cur_off + 12)
        )
        tpn_vec.append(vec)
        cur_off += 16
    return tpn_vec

def parse_mesh_directory(buf, off):
    mesh_directory = parse_directory_header(buf, off)
    if mesh_directory.type != 0x4:
        print("Invalid mesh directory structure")
    
    file_count = mesh_directory.file_count
    block_start = off + 12

    parsed_mesh_data = ParsedMeshData()

    for i in range(file_count):
        mesh_data_block_header = parse_directory_header(buf, block_start)
        if mesh_data_block_header.type == 5:
            parsed_mesh_data.faces = parse_face_block(buf, block_start)

        if mesh_data_block_header.type == 0x50000:
            parsed_mesh_data.material_list = parse_material_list_block(buf, block_start)

        if mesh_data_block_header.type == 0x60000:
            parsed_mesh_data.material_indices = parse_material_indices_block(buf, block_start)

        if mesh_data_block_header.type == 0x70000:
            parsed_mesh_data.vertices = parse_vertex_block(buf, block_start)

        if mesh_data_block_header.type == 0x80000:
            parsed_mesh_data.normals = parse_normal_block(buf, block_start)

        if mesh_data_block_header.type == 0xA0000:
            parsed_mesh_data.uvs = parse_uv_block(buf, block_start)

        if mesh_data_block_header.type == 0xB0000:
            parsed_mesh_data.colors = parse_color_block(buf, block_start)

        if mesh_data_block_header.type == 0x100000:
            parsed_mesh_data.bones_list = parse_bone_list_block(buf, block_start)

        if mesh_data_block_header.type == 0xC0000:
            print(f"Parsing vertex weights for mesh {i}")
            parsed_mesh_data.weights = parse_vertex_weight_block(buf, block_start)
            print(f"Parsed {len(parsed_mesh_data.weights)} vertex weights")

        

        if mesh_data_block_header.type == 0x120000:
            parsed_mesh_data.tpn_vec = parse_tpn_vec_block(buf, block_start)

        block_start += mesh_data_block_header.directory_size

    return parsed_mesh_data

def parse_material_directory(buf, off):
    material_directory_header = parse_directory_header(buf, off)
    print(f"Material Directory: Type: {material_directory_header.type}, File Count: {material_directory_header.file_count}, Size: {material_directory_header.directory_size}")
    materials = []
    directory_start = off + 12
    for i in range(material_directory_header.file_count):
        material_block_header = parse_directory_header(buf, directory_start)
        print(f"Material Block {i}: Type: {material_block_header.type}, File Count: {material_block_header.file_count}, Size: {material_block_header.directory_size}")
        ambient_rgba = (
            read_float_le(buf, directory_start + 12),
            read_float_le(buf, directory_start + 16),
            read_float_le(buf, directory_start + 20),
            read_float_le(buf, directory_start + 24)
        )
        print(f"Material {i}: Ambient RGBA: {ambient_rgba}")
        diffuse_rgba = (
            read_float_le(buf, directory_start + 28),
            read_float_le(buf, directory_start + 32),
            read_float_le(buf, directory_start + 36),
            read_float_le(buf, directory_start + 40)
        )
        print(f"Material {i}: Diffuse RGBA: {diffuse_rgba}")
        specular_rgba = (
            read_float_le(buf, directory_start + 44),
            read_float_le(buf, directory_start + 48),
            read_float_le(buf, directory_start + 52),
            read_float_le(buf, directory_start + 56)
        )
        print(f"Material {i}: Specular RGBA: {specular_rgba}")
        specular_str = read_float_le(buf, directory_start + 60)
        print(f"Material {i}: Specular Strength: {specular_str}")
        texture_count = read_uint32_le(buf, directory_start + 64)
        print(f"Material {i}: Texture Count: {texture_count}")
        unknown_data = buf[directory_start + 68:directory_start + 68 + 200] #unk data buffer is 200 bytes long

        texture_indices = [None, None, None]
        for j in range(texture_count):
            texture_indices[j] = read_uint32_le(buf, directory_start + 268 + j * 4)

        
        parsed_material_data = ParsedMaterialData(
            ambient_rgba=ambient_rgba,
            diffuse_rgba=diffuse_rgba,
            specular_rgba=specular_rgba,
            specular_str=specular_str,
            texture_count=texture_count,
            unknown_data=unknown_data,
            texture_diffuse=texture_indices[0],
            texture_normal=texture_indices[1],
            texture_specular=texture_indices[2]
        )
        materials.append(parsed_material_data)
        directory_start += material_block_header.directory_size
    
    return materials

def parse_texture_directory(buf, off):
    texture_directory_header = parse_directory_header(buf, off)
    directory_start = off + 12
    textures = []
    for i in range(texture_directory_header.file_count):
        image_block_header = parse_directory_header(buf, directory_start)
        image_idx = read_uint32_le(buf, directory_start + 12)
        width = read_uint32_le(buf, directory_start + 16)
        height = read_uint32_le(buf, directory_start + 20)
        data = buf[directory_start + 24:image_block_header.directory_size - 12]
        parsed_texture_data = ParsedTextureData(image_idx=image_idx, width=width, height=height, data=data)
        textures.append(parsed_texture_data)
        directory_start += image_block_header.directory_size

    return textures

def parse_fskl(buf):
    print("Parsing FSKL data...")
    skeleton_data = ParsedFSKLData(root_indices=[], bones=[])
    offset = 0

    top_directory_header = parse_directory_header(buf, offset)
    offset += 12

    #Hierarchy block
    hierarchy_block_header = parse_directory_header(buf, offset)
    offset += 12

    for i in range(hierarchy_block_header.file_count):
        root_index = read_uint32_le(buf, offset)
        skeleton_data.root_indices.append(root_index)
        offset += 4

    # Read the bone blocks 
    for i in range(top_directory_header.file_count - 1):
        bone_block_header = parse_directory_header(buf, offset)
        offset += 12

        node_id = read_uint32_le(buf, offset)
        offset += 4
        parent_id = read_uint32_le(buf, offset)
        offset += 4
        child_id = read_uint32_le(buf, offset)
        offset += 4
        sibling_id = read_uint32_le(buf, offset)
        offset += 4
        
        scale_x = read_float_le(buf, offset)
        offset += 4
        scale_y = read_float_le(buf, offset)
        offset += 4
        scale_z = read_float_le(buf, offset)
        offset += 4

        # Rotation (XYZ) in radians
        rot_x = read_float_le(buf, offset)
        offset += 4
        rot_y = read_float_le(buf, offset)
        offset += 4
        rot_z = read_float_le(buf, offset)
        offset += 4

        # IK influence weight
        ik_influence_weight = read_float_le(buf, offset)
        offset += 4
        # IK blend parameter
        ik_blend_param = read_float_le(buf, offset)
        offset += 4

        # Translation (XYZ)
        trans_x = read_float_le(buf, offset)
        offset += 4
        trans_y = read_float_le(buf, offset)
        offset += 4
        trans_z = read_float_le(buf, offset)
        offset += 4

        transform_flags = read_float_le(buf, offset)
        offset += 4

        unk_int = read_uint32_le(buf, offset)
        offset += 4

        unk_bone_param = read_uint32_le(buf, offset)
        offset += 4

        remaining_size = bone_block_header.directory_size - 84
        unk_data = buf[offset:offset + remaining_size]
        offset += remaining_size

        if parent_id == 0xFFFFFFFF:
            parent_id = -1
        if child_id == 0xFFFFFFFF:
            child_id = -1
        if sibling_id == 0xFFFFFFFF:
            sibling_id = -1

        bone = ParsedBoneData(
            node_id=node_id,
            parent_id=parent_id,
            child_id=child_id,
            sibling_id=sibling_id,
            scale=(scale_x, scale_y, scale_z),
            rotation=(rot_x, rot_y, rot_z),
            ik_influence_weight=ik_influence_weight,
            ik_blend_param=ik_blend_param,
            translation=(trans_x,trans_y,trans_z),
            transform_flags=transform_flags,
            unk_int=unk_int,
            unknown_bone_param=unk_bone_param,    
            unk_data=unk_data
        )
        skeleton_data.bones.append(bone)

    return skeleton_data

def parse_directory(buf, off, fmod_data: ParsedFMODData):
    directory_header = parse_directory_header(buf, off)
    directory_data_start = off + 12
    if directory_header.type == 0x20000:
        directory_data_start += directory_header.directory_size - 12
        pass

    elif directory_header.type == 2:
        file_count = directory_header.file_count
        for i in range(file_count):
            cur_mesh_header = parse_directory_header(buf, directory_data_start)
            mesh_data = parse_mesh_directory(buf, directory_data_start)
            fmod_data.meshes.append(mesh_data)
            directory_data_start += cur_mesh_header.directory_size

    elif directory_header.type == 0x9:
        print("Material Block found")
        file_count = directory_header.file_count
        materials = parse_material_directory(buf, off)
        fmod_data.materials.extend(materials)

    elif directory_header.type == 0xA:
        file_count = directory_header.file_count
        texture_data = parse_texture_directory(buf, off)
        fmod_data.textures.extend(texture_data)

class ImportFMOD(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    """Import FMOD model"""
    bl_idname = "import_scene.fmod"
    bl_label = "Import FMOD"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: bpy.props.StringProperty(
        default="*.fmod",
        options={'HIDDEN'},
        maxlen=255,
    )

    def execute(self, context):
        # Get the file path
        filepath = self.filepath
        texture_folder = find_texture_folder(filepath)
        skeleton_path = find_fskl_file(filepath)

        if not texture_folder:
            self.report({'ERROR'}, "Texture folder not found. Please ensure the texture folder is in the same directory as the FMOD file.")
            return {'CANCELLED'}

        # Import the FMOD model
        try:
            print("Importing FMOD model from testestest:", filepath)

            parsed_fmod_data = ParsedFMODData(meshes=[], textures=[], materials=[])
            with open(filepath, 'rb') as file:
                buf = file.read()
                file.seek(0)
                root_dir = parse_directory_header(buf, 0)
                print(f"Root Directory: Type={root_dir.type}, File Count={root_dir.file_count}, Size={root_dir.directory_size}")
                start_offset = 12
                for i in range(root_dir.file_count):
                    file.seek(start_offset)
                    block_header = parse_directory_header(buf, start_offset)
                    parse_directory(buf, start_offset, parsed_fmod_data)
                    start_offset += block_header.directory_size

            parsed_skeleton_data = None
            if skeleton_path:
                parsed_skeleton_data = ParsedFSKLData(root_indices=[], bones=[])
                with open(skeleton_path, 'rb') as fskl_file:
                    fskl_buf = fskl_file.read()
                    fskl_file.seek(0)
                    parsed_skeleton_data = parse_fskl(fskl_buf)

            print("Parsed FMOD data successfully")
            print(f"Parsed meshes: {len(parsed_fmod_data.meshes)}")
            print(f"Parsed materials: {len(parsed_fmod_data.materials)}")
            print(f"Parsed textures: {len(parsed_fmod_data.textures)}")
            if parsed_skeleton_data:
                print(f"Parsed skeleton data: {len(parsed_skeleton_data.bones)} bones")

            
            texture_dic = {}
            files = sorted([f for f in os.listdir(texture_folder) if os.path.isfile(os.path.join(texture_folder, f))])
            for i, texture in enumerate(parsed_fmod_data.textures):
                file_path = os.path.join(texture_folder, files[texture.image_idx])
                try:
                    img = bpy.data.images.load(file_path)
                    texture_dic[i] = img
                except Exception as e:
                    print(f"Failed to load texture {file_path}: {e}")
                    continue
            print("Loaded textures successfully")

            print(f"Texture dictionary: {texture_dic}")

            blender_materials = []
            for i, material in enumerate(parsed_fmod_data.materials):
                mat_name = f"Material_{i}"
                blender_material = create_blender_material(mat_name, material, texture_dic)
                if blender_material:
                    blender_materials.append(blender_material)
                    print(f"Created material '{mat_name}'")
                else:
                    print(f"Failed to create material '{mat_name}'")
            print("Created materials successfully")
            print(f"Blender materials: {blender_materials}")

            armature_obj = None
            if parsed_skeleton_data:
                armature_name = os.path.basename(skeleton_path).replace(".fskl", "")
                armature_obj = create_armature(armature_name, parsed_skeleton_data)
                if armature_obj:
                    print(f"Created armature '{armature_name}'")
                else:
                    print(f"Failed to create armature '{armature_name}'")
                print("Created armature successfully")
            
            mesh_objects = []
            for i, mesh_data in enumerate(parsed_fmod_data.meshes):
                mesh_name = f"Mesh_{i}"
                mesh_obj = create_blender_mesh(mesh_name, mesh_name, mesh_data, blender_materials)
                if mesh_obj:
                    print(f"Created object '{mesh_name}' with mesh '{mesh_name}'")
                    mesh_objects.append(mesh_obj)
                else:
                    print(f"Failed to create object for mesh '{mesh_name}'")

            # Parent meshes to armature
            if armature_obj and mesh_objects:
                for mesh_obj in mesh_objects:
                    parent_mesh_to_armature(mesh_obj, armature_obj)
                print("Parented meshes to armature successfully")

        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        return {'FINISHED'}

def collect_scene_texture(context):
    texture_dict = {}
    texture_idx = 0

    for obj in context.scene.objects:
        if not hasattr(obj.data, "materials") or not obj.data.materials:
            continue

        for mat_slots in obj.material_slots:
            if not mat_slots.material or not mat_slots.material.use_nodes:
                continue

            for node in mat_slots.material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    if node.image not in texture_dict:
                        texture_dict[node.image] = texture_idx
                        texture_idx += 1
    
    return texture_dict

def texture_dic_to_namelist(texture_dic: dict[bpy.types.Image, int]) -> list[str]:
    """ Take a texture dicitonary and return a namelist of texture names """
    namelist = []
    for image, idx in texture_dic.items():
        namelist.append(image.name)
    return namelist

def collect_scene_materials(context):
    material_dict = {}
    material_names = []
    material_idx = 0

    for obj in context.scene.objects:
        if not hasattr(obj.data, "materials") or not obj.data.materials:
            continue

        for mat_slots in obj.material_slots:
            if not mat_slots.material or not mat_slots.material.use_nodes:
                continue

            if mat_slots.material.name in material_names:
                continue

            if mat_slots.material not in material_dict:
                material_dict[mat_slots.material] = material_idx
                material_names.append(mat_slots.material.name)
                material_idx += 1
    
    return material_dict

def collect_scene_meshes(context):
    mesh_dict = {}
    mesh_idx = 0

    for obj in context.scene.objects:
        if obj.type != 'MESH':
            continue
            
        if not obj.data or not isinstance(obj.data, bpy.types.Mesh):
            continue

        mesh_dict[obj] = mesh_idx
        mesh_idx += 1
        
    return mesh_dict

def collect_scene_armature(context):
    armature_dict = {}
    armature_idx = 0

    for obj in context.scene.objects:
        if obj.type != 'ARMATURE':
            continue
            
        if not obj.data or not isinstance(obj.data, bpy.types.Armature):
            continue

        armature_dict[obj] = armature_idx
        armature_idx += 1
        
    return armature_dict

def texture_data_from_image(image: bpy.types.Image, idx: int) -> ParsedTextureData:
    width = image.size[0]
    height = image.size[1]
    zeros = bytes([0] * 244)

    return ParsedTextureData(
        image_idx=idx,
        width=width,
        height=height,
        data=zeros
    )

def material_data_from_materials(material: bpy.types.Material, texture_names: list[str]) -> ParsedMaterialData:
    diffuse_textre = None
    normal_texture = None
    specular_texture = None

    principled_node = None

    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled_node = node
            break

    if principled_node:
        for link in material.node_tree.links:
            if link.to_node == principled_node:
                input_name = link.to_socket.name
                from_node = link.from_node

                if from_node.type == 'TEX_IMAGE':
                    texture_index = texture_names.index(from_node.image.name)

                    if input_name == 'Base Color':
                        diffuse_textre = texture_index
                    elif input_name == 'Specular' or input_name == 'Specular Tint' or input_name == 'Specular IOR Level' or input_name == 'Roughness':
                        specular_texture = texture_index
                    elif input_name == 'Normal':
                        normal_texture = texture_index

                    elif input_name == 'Normal' and from_node.type == 'NORMAL_MAP':
                        for normal_link in material.node_tree.links:
                            if normal_link.from_node == from_node and normal_link.from_node.type == 'TEX_IMAGE':
                                if normal_link.from_node.image and normal_link.from_node.image.name in texture_names:
                                    normal_texture = texture_names.index(normal_link.from_node.image.name)

    # Create a ParsedMaterialData object
    # not checking from the material for now using mostly default values (TODO)
    diffuse_rgba = (1.0,1.0,1.0, 1.0)
    specular_rgba = (1.0, 1.0, 1.0, 0.0)
    ambient_rgba = (0.3,0.3,0.3, 0.0)
    specular_str = 50.0

    texture_count = 0
    if diffuse_textre is not None:
        texture_count += 1
    if normal_texture is not None:
        texture_count += 1
    if specular_texture is not None:
        texture_count += 1

    parsed_material_data = ParsedMaterialData(
        ambient_rgba=ambient_rgba,
        diffuse_rgba=diffuse_rgba,
        specular_rgba=specular_rgba,
        specular_str=specular_str,
        texture_count=texture_count,
        unknown_data=bytes([0] * 200),
        texture_diffuse=diffuse_textre,
        texture_normal=normal_texture,
        texture_specular=specular_texture
    )

    return parsed_material_data

def mesh_data_from_mesh(mesh: bpy.types.Mesh, material_names: list[str]) -> ParsedMeshData:
    mesh_data = ParsedMeshData()

    temp_mesh = mesh.data.copy()
    temp_mesh.transform(Matrix.Identity(4))

    #triangulate
    bm = bmesh.new()
    bm.from_mesh(temp_mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()
    bm.to_mesh(temp_mesh)
    bm.free()

    temp_mesh.update(calc_edges=True)

    # Vertices
    mesh_data.vertices = [(v.co.x, v.co.y, v.co.z) for v in temp_mesh.vertices]
    print(f"Mesh vertices: {len(mesh_data.vertices)}")

    # Normals
    mesh_data.normals = [(v.normal.x, v.normal.y, v.normal.z) for v in temp_mesh.vertices]
    print(f"Mesh normals: {len(mesh_data.normals)}")

    # UVs
    if temp_mesh.uv_layers and len(temp_mesh.uv_layers) > 0:
        vert_to_uv = {}
        active_uv_layer = temp_mesh.uv_layers.active
        for poly in temp_mesh.polygons:
            for loop_index in poly.loop_indices:
                vert_idx = temp_mesh.loops[loop_index].vertex_index
                uv = active_uv_layer.data[loop_index].uv
                vert_to_uv[vert_idx] = (uv.x, uv.y)

        mesh_data.uvs = [vert_to_uv.get(i, (0.0, 0.0)) for i in range(len(temp_mesh.vertices))]
    else:
        mesh_data.uvs = [(0.0, 0.0) for _ in range(len(temp_mesh.vertices))]
    print(f"Mesh UVs: {len(mesh_data.uvs)}")

    # Colors
    if temp_mesh.vertex_colors and len(temp_mesh.vertex_colors) > 0:
        vert_to_color = {}
        for poly in temp_mesh.polygons:
            for loop_index in poly.loop_indices:
                vert_idx = temp_mesh.loops[loop_index].vertex_index
                color = temp_mesh.vertex_colors.active.data[loop_index].color
                vert_to_color[vert_idx] = (color[0], color[1], color[2], color[3] if len(color) == 4 else 1.0)
            
        mesh_data.colors = [vert_to_color.get(i, (0.0, 0.0, 0.0, 1.0)) for i in range(len(temp_mesh.vertices))]
    else:
        mesh_data.colors = [(1.0, 1.0, 1.0, 1.0) for _ in range(len(temp_mesh.vertices))]
    print(f"Mesh colors: {len(mesh_data.colors)}")

    triangles = []
    face_materials = []

    for poly in temp_mesh.polygons:

        tri = [temp_mesh.loops[i].vertex_index for i in poly.loop_indices]
        triangles.append(tri)

        if poly.material_index < len(material_names):
            face_materials.append(poly.material_index)
        else:
            face_materials.append(0)

    material_list = list(set(face_materials))

    strips = []
    strip_material_indices = []

    strips, strip_material_indices = create_optimized_strips(triangles, face_materials, material_list)

    mesh_data.faces = strips
    mesh_data.material_list = material_list
    mesh_data.material_indices = strip_material_indices 

    # Extract vertex weights and bone indices if mesh is rigged
    if mesh.vertex_groups:
        print(f"Mesh '{mesh.name}' has vertex groups, checking for armature parenting...")
        # Find an armature that might be parenting this mesh
        armature = None
        if mesh.parent and mesh.parent.type == 'ARMATURE':
            armature = mesh.parent
            print(f"Found armature '{armature.name}' as parent of mesh '{mesh.name}'")
        else:
            print(f"Mesh '{mesh.name}' has no armature parent, checking for armature modifier...")
        
        if armature:
            print(f"Found armature '{armature.name}' for mesh '{mesh.name}'")
            
            # Map vertex groups to bone indices
            bone_ids = []  # Final list of bone IDs
            vgroup_to_bone_idx = {}  # Maps vertex group index to bone index in bone_ids list
            
            for vg in mesh.vertex_groups:
                # Look for vertex groups that follow the "Bone_X" pattern
                if vg.name.startswith("Bone_"):
                    try:
                        bone_id = int(vg.name[5:])  # Extract number from "Bone_X"
                        bone_idx = len(bone_ids)
                        bone_ids.append(bone_id)
                        vgroup_to_bone_idx[vg.index] = bone_idx
                    except ValueError:
                        # Not a bone vertex group or invalid format
                        pass
            
            if bone_ids:
                print(f"Found {len(bone_ids)} bones referenced in vertex groups")
                
                # Extract weights for each vertex
                weights = []
                for vert in temp_mesh.vertices:
                    vert_weights = []
                    
                    for vg in vert.groups:
                        if vg.group in vgroup_to_bone_idx and vg.weight > 0:
                            bone_idx = vgroup_to_bone_idx[vg.group]
                            # Convert weight from 0-1 to 0-100 scale as expected by FMOD
                            weight_value = vg.weight * 100.0
                            vert_weights.append((bone_idx, weight_value))
                    
                    # Sort by weight (highest first) and limit to 4 weights if needed
                    vert_weights.sort(key=lambda x: x[1], reverse=True)
                    if len(vert_weights) > 4:
                        vert_weights = vert_weights[:4]
                    
                    # Normalize weights to sum to 100 if we have weights
                    if vert_weights:
                        total_weight = sum(w for _, w in vert_weights)
                        if total_weight > 0:
                            vert_weights = [(idx, (w / total_weight) * 100.0) for idx, w in vert_weights]
                    
                    weights.append(vert_weights)
                
                mesh_data.weights = weights
                mesh_data.bones_list = bone_ids
                print(f"Extracted weights for {len(weights)} vertices referencing {len(bone_ids)} bones")

    bpy.data.meshes.remove(temp_mesh)
    return mesh_data

def fskl_data_from_armature(armature: bpy.types.Object) -> ParsedFSKLData:
    fskl_data = ParsedFSKLData(root_indices=[], bones=[])

    root_bones = [bone for bone in armature.data.bones if bone.parent is None]

    bone_name_to_node_id = {}

    for bone in armature.data.bones:
        bone_name: str = bone.name

        if bone_name.startswith("Bone_"):
            extracted_id = int(bone_name[5:])
            bone_name_to_node_id[bone_name] = extracted_id
        else:
            print(f"Bone '{bone_name}' does not follow the 'Bone_X' naming convention, skipping...")
            return None
    
    for root_bone in root_bones:
        fskl_data.root_indices.append(bone_name_to_node_id[root_bone.name])
    
    for bone in armature.data.bones:
        bone_name: str = bone.name
        node_id: int = bone_name_to_node_id[bone_name]

        parent_id = -1
        if bone.parent:
            parent_id = bone_name_to_node_id[bone.parent.name]
        
        left_child_id = -1
        if bone.children:
            left_child_id = bone_name_to_node_id[bone.children[0].name]

        right_sibling_id = -1
        if bone.parent and bone.parent.children and len(bone.parent.children) > 1:
            cur_bone_index = list(bone.parent.children).index(bone)
            if cur_bone_index + 1 < len(bone.parent.children):
                right_sibling_id = bone_name_to_node_id[bone.parent.children[cur_bone_index + 1].name]

        #computing position
        if bone.parent:
            position_vec = bone.head_local - bone.parent.head_local
        else:
            position_vec = bone.head_local

        unknown_bone_param = 0
        if "unknown_bone_param" in bone:
            unknown_bone_param = bone["unknown_bone_param"]

        bone_data = ParsedBoneData(
            node_id=node_id,
            parent_id=parent_id,
            child_id=left_child_id,
            sibling_id=right_sibling_id,
            scale=(1.0, 1.0, 1.0),
            rotation=(1.0, 0.0, 0.0),
            ik_influence_weight=0.0,
            ik_blend_param=1.0,
            translation=(position_vec.x, position_vec.y, position_vec.z),
            transform_flags=1.0,
            unk_int=0xFFFFFFFF,
            unknown_bone_param=unknown_bone_param,
            unk_data=bytes([0] * 184)
        )
        fskl_data.bones.append(bone_data)
        print(f"Bone '{bone_name}' added with node ID {node_id}, parent ID {parent_id}, left child ID {left_child_id}, right sibling ID {right_sibling_id}")

    return fskl_data

def write_directory_header(type: int, file_count: int, directory_size: int) -> bytes:
    return (
        write_uint32_le(type) +
        write_uint32_le(file_count) +
        write_uint32_le(directory_size)
    )

def write_vertex_block(vertices: list[tuple[float, float, float]]) -> bytes:
    block_size = 12 + (len(vertices) * 12)
    header = write_directory_header(0x70000, len(vertices), block_size)

    data = bytearray()
    for vertex in vertices:
        data.extend(write_float_le(vertex[0]))
        data.extend(write_float_le(vertex[1]))
        data.extend(write_float_le(vertex[2]))
    
    return header + data

def write_normal_block(normals: list[tuple[float, float, float]]) -> bytes:
    block_size = 12 + (len(normals) * 12)
    header = write_directory_header(0x80000, len(normals), block_size)

    data = bytearray()
    for normal in normals:
        data.extend(write_float_le(normal[0]))
        data.extend(write_float_le(normal[1]))
        data.extend(write_float_le(normal[2]))
    
    return header + data

def write_uv_block(uvs: list[tuple[float, float]]) -> bytes:
    block_size = 12 + (len(uvs) * 8)
    header = write_directory_header(0xA0000, len(uvs), block_size)

    data = bytearray()
    for uv in uvs:
        data.extend(write_float_le(uv[0]))
        data.extend(write_float_le(1.0 - uv[1]))
    
    return header + data

def write_color_block(colors: list[tuple[float, float, float, float]]) -> bytes:
    block_size = 12 + (len(colors) * 16)
    header = write_directory_header(0xB0000, len(colors), block_size)

    data = bytearray()
    for color in colors:
        data.extend(write_float_le(color[0]*255.0))
        data.extend(write_float_le(color[1]*255.0))
        data.extend(write_float_le(color[2]*255.0))
        data.extend(write_float_le(color[3]*255.0))
    
    return header + data

def write_material_list_block(materials: list[int]) -> bytes:
    block_size = 12 + (len(materials) * 4)
    header = write_directory_header(0x50000, len(materials), block_size)

    data = bytearray()
    for material in materials:
        data.extend(write_uint32_le(material))
    
    return header + data

def write_material_indices_block(material_indices: list[int]) -> bytes:
    block_size = 12 + (len(material_indices) * 4)
    header = write_directory_header(0x60000, len(material_indices), block_size)

    data = bytearray()
    for material_index in material_indices:
        data.extend(write_uint32_le(material_index))
    
    return header + data

def write_face_block(faces: list[list[int]]) -> bytes:
    #each strip has a count and a list of indices
    strip_size = [len(strip) * 4 + 4 for strip in faces]
    block_size = 12 + sum(strip_size)
    header = write_directory_header(5, 1, block_size + 12) #outer face block header
    strips_header = write_directory_header(0x30000, len(faces), block_size) #inner strips block container

    data = bytearray()
    for strip in faces:
        data.extend(write_uint32_le(len(strip))) #count
        for index in strip:
            data.extend(write_uint32_le(index))
    
    return header + strips_header + data

def write_weight_block(weights: list[list[tuple[int, float]]]) -> bytes:
    """Write a vertex weight block with its header."""
    # Calculate size: 12 for header + sum of (4 + weight_count * 8) for each vertex
    vertex_sizes = [4 + (len(weight_list) * 8) for weight_list in weights]
    block_size = 12 + sum(vertex_sizes)
    
    header = write_directory_header(0xC0000, len(weights), block_size)
    
    data = bytearray()
    for weight_list in weights:
        # Write weight count
        data.extend(write_uint32_le(len(weight_list)))
        
        # Write weights (bone index + weight value)
        for bone_idx, weight_value in weight_list:
            data.extend(write_uint32_le(bone_idx))
            data.extend(write_float_le(weight_value))
    
    return header + data

def write_bone_list_block(bones: list[int]) -> bytes:
    """Write a bone list block with its header."""
    block_size = 12 + (len(bones) * 4)  # 12 bytes for header, 4 bytes per bone index
    header = write_directory_header(0x100000, len(bones), block_size)
    
    data = bytearray()
    for bone in bones:
        data.extend(write_uint32_le(bone))
    
    return header + data

def write_mesh_directory(mesh_data: ParsedMeshData) -> bytes:
    blocks = []

    if mesh_data.faces:
        blocks.append(write_face_block(mesh_data.faces))
    
    if mesh_data.material_list:
        blocks.append(write_material_list_block(mesh_data.material_list))

    if mesh_data.material_indices:
        blocks.append(write_material_indices_block(mesh_data.material_indices))
    
    if mesh_data.vertices:
        blocks.append(write_vertex_block(mesh_data.vertices))   

    if mesh_data.normals:
        blocks.append(write_normal_block(mesh_data.normals))

    if mesh_data.uvs:
        blocks.append(write_uv_block(mesh_data.uvs))
    
    if mesh_data.colors:
        blocks.append(write_color_block(mesh_data.colors))
    
    if mesh_data.weights:
        blocks.append(write_weight_block(mesh_data.weights))

    if mesh_data.bones_list:
        blocks.append(write_bone_list_block(mesh_data.bones_list))

    total_size = 12 + sum(len(block) for block in blocks)

    header = write_directory_header(0x4, len(blocks), total_size)
    result = bytearray(header)
    for block in blocks:
        result.extend(block)

    return result

def write_material(material_data: ParsedMaterialData) -> bytes:
    block_size = 0x110
    print(f"Material data: {material_data}")
    if material_data.texture_count > 1:
        block_size += 0x4 * material_data.texture_count

    header = write_directory_header(2, 1, block_size)
    data = bytearray()
    data.extend(header)
    data.extend(write_float_le(material_data.ambient_rgba[0]))
    data.extend(write_float_le(material_data.ambient_rgba[1]))
    data.extend(write_float_le(material_data.ambient_rgba[2]))
    data.extend(write_float_le(material_data.ambient_rgba[3]))
    data.extend(write_float_le(material_data.diffuse_rgba[0]))
    data.extend(write_float_le(material_data.diffuse_rgba[1]))
    data.extend(write_float_le(material_data.diffuse_rgba[2]))
    data.extend(write_float_le(material_data.diffuse_rgba[3]))
    data.extend(write_float_le(material_data.specular_rgba[0]))
    data.extend(write_float_le(material_data.specular_rgba[1]))
    data.extend(write_float_le(material_data.specular_rgba[2]))
    data.extend(write_float_le(material_data.specular_rgba[3]))
    data.extend(write_float_le(material_data.specular_str))
    data.extend(write_uint32_le(material_data.texture_count))
    data.extend(material_data.unknown_data)
    data.extend(write_uint32_le(material_data.texture_diffuse))
    if material_data.texture_count > 1:
        data.extend(write_uint32_le(material_data.texture_normal))
    if material_data.texture_count > 2:
        data.extend(write_uint32_le(material_data.texture_specular))    
    return data

    bloc

def write_texture(texture_data: ParsedTextureData) -> bytes:
    header = write_directory_header(0, 1, 0x10C); 
    data = bytearray()
    data.extend(header)
    data.extend(write_uint32_le(texture_data.image_idx))
    data.extend(write_uint32_le(texture_data.width))
    data.extend(write_uint32_le(texture_data.height))
    data.extend(texture_data.data)
    return data

def write_bone_block(bone_data: ParsedBoneData) -> bytes:
    header = write_directory_header(0x40000001, 1, 0x10C)
    data = bytearray()
    data.extend(header)
    data.extend(write_uint32_le(bone_data.node_id))
    data.extend(write_uint32_le(0xFFFFFFFF if bone_data.parent_id == -1 else bone_data.parent_id))
    data.extend(write_uint32_le(0xFFFFFFFF if bone_data.child_id == -1 else bone_data.child_id))
    data.extend(write_uint32_le(0xFFFFFFFF if bone_data.sibling_id == -1 else bone_data.sibling_id))
    data.extend(write_float_le(bone_data.scale[0]))
    data.extend(write_float_le(bone_data.scale[1]))
    data.extend(write_float_le(bone_data.scale[2]))
    data.extend(write_float_le(bone_data.rotation[0]))
    data.extend(write_float_le(bone_data.rotation[1]))
    data.extend(write_float_le(bone_data.rotation[2]))
    data.extend(write_float_le(bone_data.ik_influence_weight))
    data.extend(write_float_le(bone_data.ik_blend_param))
    data.extend(write_float_le(bone_data.translation[0]))
    data.extend(write_float_le(bone_data.translation[1]))
    data.extend(write_float_le(bone_data.translation[2]))
    data.extend(write_float_le(bone_data.transform_flags))
    data.extend(write_uint32_le(bone_data.unk_int))
    data.extend(write_uint32_le(bone_data.unknown_bone_param))
    data.extend(bone_data.unk_data)
    return data

def write_fskl(parsed_skeleton_data: ParsedFSKLData) -> bytes:
    root_block_header = write_directory_header(0, len(parsed_skeleton_data.root_indices), len(parsed_skeleton_data.root_indices) * 4 + 12)
    root_block = bytearray(root_block_header)
    for root_index in parsed_skeleton_data.root_indices:
        root_block.extend(write_uint32_le(root_index))
    
    skeleton_blocks = bytearray()
    for bone in parsed_skeleton_data.bones:
        bone_block = write_bone_block(bone)
        skeleton_blocks.extend(bone_block)

    main_file_header = write_directory_header(0xC0000000, 1 + len(parsed_skeleton_data.bones), len(skeleton_blocks) + len(root_block) + 12)
    main_file_data = bytearray(main_file_header)
    main_file_data.extend(root_block)
    main_file_data.extend(skeleton_blocks)
    return main_file_data

def write_fmod(parsed_data: ParsedFMODData) -> None:
    meshes_block = []
    materials_block = []
    textures_block = []

    if parsed_data.meshes:
        for mesh in parsed_data.meshes:
            mesh_data = write_mesh_directory(mesh)
            meshes_block.append(mesh_data)

    if parsed_data.materials:
        for material in parsed_data.materials:
            material_data = write_material(material)
            materials_block.append(material_data)

    if parsed_data.textures:
        for texture in parsed_data.textures:
            texture_data = write_texture(texture)
            textures_block.append(texture_data)

    main_file_data = bytearray()
    init_block_header = write_directory_header(0x20000, 1, 0x10);
    main_file_data.extend(init_block_header)
    main_file_data.extend(write_uint32_le(random.randint(0, 0xFFFFFFFF)))

    mesh_block_size = sum(len(block) for block in meshes_block) + 12
    mesh_block_header = write_directory_header(2, len(meshes_block), mesh_block_size)
    main_file_data.extend(mesh_block_header)
    for block in meshes_block:
        main_file_data.extend(block)

    material_block_size = sum(len(block) for block in materials_block) + 12
    material_block_header = write_directory_header(0x9, len(materials_block), material_block_size)
    main_file_data.extend(material_block_header)
    for block in materials_block:
        main_file_data.extend(block)
    
    texture_block_size = sum(len(block) for block in textures_block) + 12
    texture_block_header = write_directory_header(0xA, len(textures_block), texture_block_size)
    main_file_data.extend(texture_block_header)
    for block in textures_block:
        main_file_data.extend(block)

    main_file_header = write_directory_header(1, 4, len(main_file_data) + 12)
    main_file_data = bytearray(main_file_header) + main_file_data
    return main_file_data

class ExportFMOD(bpy.types.Operator, bpy_extras.io_utils.ExportHelper):
    """Export model, textures, materials and skeleton to FMOD"""
    bl_idname = "export_scene.fmod"
    bl_label = "Export FMOD"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ""

    filter_glob: bpy.props.StringProperty(
        default="*.fmod",
        options={'HIDDEN'},
        maxlen=255,
    )

    write_log_files: bpy.props.BoolProperty(
        name="Write Log File",
        description="Write a log file with the export process",
        default=False,
    )

    def execute(self, context):
        
        textures = collect_scene_texture(context)
        print(f"Collected {len(textures)} textures from the scene")
        texture_names = [image.name for image, _ in sorted(textures.items(), key=lambda item: item[1])]
        print(f"Texture names: {texture_names}")
        texture_data = []
        for image, idx in textures.items():
            texture_data.append(texture_data_from_image(image, idx))
        print(f"Generated texture data for {len(texture_data)} textures")

        materials = collect_scene_materials(context)
        print(f"Collected {len(materials)} materials from the scene")
        material_names = [material.name for material, _ in sorted(materials.items(), key=lambda item: item[1])]
        print(f"Material names: {material_names}")
        materials_data = []
        for material, idx in materials.items():
            materials_data.append(material_data_from_materials(material, texture_names))
        print(f"Generated material data for {len(materials_data)} materials")

        meshes = collect_scene_meshes(context)
        print(f"Collected {len(meshes)} meshes from the scene")
        mesh_names = [mesh.name for mesh, _ in sorted(meshes.items(), key=lambda item: item[1])]
        print(f"Mesh names: {mesh_names}")
        meshes_data = []
        for mesh, idx in meshes.items():
            meshes_data.append(mesh_data_from_mesh(mesh, material_names))
        print(f"Generated mesh data for {len(meshes_data)} meshes")

        # Create the FMOD data structure
        parsed_fmod_data = ParsedFMODData(meshes=meshes_data, textures=texture_data, materials=materials_data)
        print("Parsed FMOD data successfully")

        # Check if an armature is present in the scene
        armature = collect_scene_armature(context)
        if armature:
            armature_obj = list(armature.keys())[0]
            print(f"Found armature '{armature_obj.name}' in the scene")
            parsed_fskl_data = fskl_data_from_armature(armature_obj)
            print("Parsed skeleton data successfully")
        else:
            parsed_fskl_data = None
            print("No armature found in the scene")

        # Generate FMOD buffer 
        fmod_buffer = write_fmod(parsed_fmod_data)
        print("Generated FMOD buffer successfully")

        fskl_buffer = None
        if parsed_fskl_data:
            fskl_buffer = write_fskl(parsed_fskl_data)
            print("Generated FSKL buffer successfully")

        # Extract the file name from the filepath
        file_name = os.path.basename(self.filepath)
        # Create the directory if it doesn't exist
        directory = os.path.dirname(self.filepath)
        export_directory = os.path.join(directory, file_name)
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
            print(f"Created directory: {export_directory}")

        # Save the FMOD data to the directory
        fmod_name = "0001_0000001C.fmod"
        fmod_name_no_ext = fmod_name.split(".")[0]
        self.filepath = os.path.join(export_directory, fmod_name)

        # Write the FMOD data to the file
        with open(self.filepath, 'wb') as fmod_file:
            fmod_file.write(fmod_buffer)
            print(f"Exported FMOD data to {self.filepath}")

        #Generate random hex string between 0x300000 and 0xA000000 left padded to 8 digits
        texture_unique_id = hex(random.randint(0x300000, 0xA000000))[2:].upper()
        texture_unique_id = texture_unique_id.zfill(8)
        # Create the texture directory name
        texture_directory_name = f"0003_{texture_unique_id}"
        # Create the directory for the textures
        texture_directory_path = os.path.join(export_directory, texture_directory_name)
        if not os.path.exists(texture_directory_path):
            os.makedirs(texture_directory_path)
            print(f"Created directory: {texture_directory_path}")

        # Save the textures to the directory
        texture_file_names = []
        for texture, index in textures.items():
            padded_index = str(index+1).zfill(4)
            random_hex = hex(random.randint(0x00000000, 0xFFFFFFFF))[2:].upper()
            padded_hex = random_hex.zfill(8)
            texture_name = f"{padded_index}_{padded_hex}.png"
            texture_path = os.path.join(texture_directory_path, texture_name)
            try:
                texture.save_render(filepath=texture_path, scene=context.scene)
                texture_file_names.append(texture_name)
                print(f"Exported texture {texture.name} to {texture_path}")
            except Exception as e:
                print(f"Failed to export texture {texture.name}: {e}")

        # Handling the skeleton file
        fskl_name = None
        if parsed_fskl_data:
            fskl_rand_id = hex(random.randint(0x00000000, 0xFFFFFFFF))[2:].upper()
            fskl_rand_id = fskl_rand_id.zfill(8)
            fskl_name = f"0002_{fskl_rand_id}"
            fskl_file_name = f"{fskl_name}.fskl"
            fskl_path = os.path.join(export_directory, fskl_file_name)
            with open(fskl_path, 'wb') as fskl_file:
                fskl_file.write(fskl_buffer)
                print(f"Exported skeleton data to {fskl_path}")

        
        # if write log files is enabled, write the log files
        # writing the texture log files
        if self.write_log_files:
            with open(os.path.join(texture_directory_path, f"{texture_directory_name}.log"), 'w') as log_file:
                log_file.write("SimpleArchive\n")
                log_file.write(f"{texture_directory_name}.bin\n")
                log_file.write(f"{len(texture_file_names)}\n")
                for texture in texture_file_names:
                    log_file.write(f"{texture_name},12,13526,1196314761\n") #magic numbers for now

            #writing the whole fmod log file
            with open(os.path.join(export_directory, f"{file_name}.log"), 'w') as log_file:
                log_file.write("SimpleArchive\n")
                log_file.write(f"{file_name}.bin\n")
                log_file.write(f"3\n")
                log_file.write(f"{fmod_name},12,13526,1196314761\n") #magic numbers for now
                if fskl_name:
                    log_file.write(f"{fskl_name}.fskl,12,13526,1196314761\n") #Placeholder for skeleton file
                log_file.write(f"{texture_directory_name}.bin,12,13526,1196314761\n")

        return {'FINISHED'}

def menu_func_import(self, context):
    self.layout.operator(ImportFMOD.bl_idname, text="FMOD (.fmod)")

def menu_func_export(self, context):
    self.layout.operator(ExportFMOD.bl_idname, text="FMOD (.fmod)")

def register():
    bpy.utils.register_class(ImportFMOD)
    bpy.utils.register_class(ExportFMOD)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ImportFMOD)
    bpy.utils.unregister_class(ExportFMOD)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()
