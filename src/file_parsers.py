import bpy #type: ignore
import os

from .binary_utils import *
from .data_classes import *

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
    print(f"Mesh Directory: Type: {mesh_directory.type}, File Count: {mesh_directory.file_count}, Size: {mesh_directory.directory_size}")

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
        unknown_data = buf[directory_start + 68:directory_start + 68 + 200] 

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

    
    hierarchy_block_header = parse_directory_header(buf, offset)
    offset += 12

    for i in range(hierarchy_block_header.file_count):
        root_index = read_uint32_le(buf, offset)
        skeleton_data.root_indices.append(root_index)
        offset += 4

    
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

        
        rot_x = read_float_le(buf, offset)
        offset += 4
        rot_y = read_float_le(buf, offset)
        offset += 4
        rot_z = read_float_le(buf, offset)
        offset += 4

        
        ik_influence_weight = read_float_le(buf, offset)
        offset += 4
        
        ik_blend_param = read_float_le(buf, offset)
        offset += 4

        
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
        print(f"Mesh directory found: {directory_header.file_count} meshes found")
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

def load_texture_no_dupes(parsed_fmod_data, texture_folder):
    image_extensions = {'.png', '.jpg', '.jpeg', '.tga', '.dds', '.bmp', '.tiff'}
    files = sorted([f for f in os.listdir(texture_folder) 
                   if os.path.isfile(os.path.join(texture_folder, f)) 
                   and os.path.splitext(f.lower())[1] in image_extensions])
    
    file_to_image = {}
    texture_dic = {}
    
    for i, texture in enumerate(parsed_fmod_data.textures):
        if texture.image_idx >= len(files):
            print(f"Warning: texture.image_idx {texture.image_idx} is out of range for {len(files)} image files")
            continue
            
        file_path = os.path.join(texture_folder, files[texture.image_idx])
        
        try:
            if file_path not in file_to_image:
                img = bpy.data.images.load(file_path)
                file_to_image[file_path] = img
                print(f"Loaded new texture: {img.name} from {file_path}")
            else:
                img = file_to_image[file_path]
                print(f"Reusing texture: {img.name} for slot {i}")
            
            texture_dic[i] = img
            
        except Exception as e:
            print(f"Failed to load texture {file_path}: {e}")
            continue
    
    print(f"Texture summary:")
    print(f"  - {len(texture_dic)} texture slots")
    print(f"  - {len(file_to_image)} unique texture files")
    print(f"  - {len(set(texture_dic.values()))} unique image objects")
    print(f"  - Found {len(files)} image files in folder")
    
    return texture_dic