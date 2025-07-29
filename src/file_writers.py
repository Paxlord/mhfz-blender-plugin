import bpy #type: ignore
import bmesh #type: ignore
from mathutils import Matrix #type: ignore

from .data_classes import *
from .model_utils import *
from .binary_utils import *
import random

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

def mesh_data_from_mesh(mesh: bpy.types.Mesh, material_names: list[str], include_bone_list: bool = True,force_empty_attributes: bool = False, include_tangeants: bool = False) -> ParsedMeshData:
    mesh_data = ParsedMeshData()

    if force_empty_attributes:
        empty_block = [65536] + [0] * 17
        empty_block[1] = 1
        empty_block[11] = 2
        empty_block[12] = 3
        mesh_data.attributes = empty_block

    temp_mesh = mesh.data.copy()
    temp_mesh.transform(Matrix.Identity(4))

    
    bm = bmesh.new()
    bm.from_mesh(temp_mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.normal_update()
    bm.to_mesh(temp_mesh)
    bm.free()

    temp_mesh.update(calc_edges=True)
    
    mesh_data.vertices = [(v.co.x, v.co.y, v.co.z) for v in temp_mesh.vertices]
    print(f"Mesh vertices: {len(mesh_data.vertices)}")
    
    mesh_data.normals = [(v.normal.x, v.normal.y, v.normal.z) for v in temp_mesh.vertices]
    print(f"Mesh normals: {len(mesh_data.normals)}")
    
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

    if temp_mesh.uv_layers.active:
        temp_mesh.calc_tangents(uvmap=temp_mesh.uv_layers.active.name)

    if include_tangeants:
        print("Extracting TPN data for export...")
        tpn_data = extract_tpn_for_export(mesh)
        if tpn_data:
            mesh_data.tpn_vec = tpn_data
            print(f"Extracted {len(tpn_data)} TPN vectors for export")

    slot_to_global_map = {}
    for i, mat in enumerate(temp_mesh.materials):
        if mat and mat.name in material_names:
            slot_to_global_map[i] = material_names.index(mat.name)
        else:
            slot_to_global_map[i] = 0 

    triangles = []
    face_materials = []

    mesh_to_global_material_map = {}
    
    for slot_idx, material_slot in enumerate(temp_mesh.materials):
        if material_slot and material_slot.name in material_names:
            global_mat_idx = material_names.index(material_slot.name)
            mesh_to_global_material_map[slot_idx] = global_mat_idx
        else:
            mesh_to_global_material_map[slot_idx] = 0

    print(f"Mesh materials: {[mat.name if mat else 'None' for mat in temp_mesh.materials]}")
    print(f"Mesh to global material mapping: {mesh_to_global_material_map}")

    actually_used_materials = set()
    
    for poly in temp_mesh.polygons:
        tri = [temp_mesh.loops[i].vertex_index for i in poly.loop_indices]
        triangles.append(tri)

        local_mat_idx = poly.material_index
        global_mat_idx = mesh_to_global_material_map.get(local_mat_idx, 0)
        face_materials.append(global_mat_idx)
        actually_used_materials.add(global_mat_idx)

    material_list = sorted(list(actually_used_materials))
    
    print(f"Final material_list for mesh: {material_list}")
    print(f"Face materials (global indices): {set(face_materials)}")

    strips = []
    strip_material_indices = []

    strips, strip_material_indices = create_sgi_strips(triangles, face_materials, material_list)

    mesh_data.faces = strips
    mesh_data.material_list = material_list
    mesh_data.material_indices = strip_material_indices  

    
    if mesh.vertex_groups:
        print(f"Mesh '{mesh.name}' has vertex groups, checking for armature parenting...")
        
        armature = None
        if mesh.parent and mesh.parent.type == 'ARMATURE':
            armature = mesh.parent
            print(f"Found armature '{armature.name}' as parent of mesh '{mesh.name}'")
        else:
            print(f"Mesh '{mesh.name}' has no armature parent, checking for armature modifier...")
        
        if armature:
            print(f"Found armature '{armature.name}' for mesh '{mesh.name}'")
            
            if include_bone_list:
                
                bone_ids = []  
                vgroup_to_bone_idx = {}  
                
                for vg in mesh.vertex_groups:
                    
                    if vg.name.startswith("Bone_"):
                        try:
                            bone_id = int(vg.name[5:])  
                            bone_idx = len(bone_ids)
                            bone_ids.append(bone_id)
                            vgroup_to_bone_idx[vg.index] = bone_idx
                        except ValueError:
                            
                            pass
                
                if bone_ids:
                    print(f"Found {len(bone_ids)} bones referenced in vertex groups")
                    
                    
                    weights = []
                    for vert in mesh.data.vertices:
                        vert_weights = []
                        
                        for vg in vert.groups:
                            if vg.group in vgroup_to_bone_idx and vg.weight > 0:
                                bone_idx = vgroup_to_bone_idx[vg.group]
                                
                                weight_value = vg.weight * 100.0
                                vert_weights.append((bone_idx, weight_value))
                        
                        
                        vert_weights.sort(key=lambda x: x[1], reverse=True)
                        if len(vert_weights) > 4:
                            vert_weights = vert_weights[:4]
                        
                        
                        if vert_weights:
                            total_weight = sum(w for _, w in vert_weights)
                            if total_weight > 0:
                                vert_weights = [(idx, (w / total_weight) * 100.0) for idx, w in vert_weights]
                        
                        weights.append(vert_weights)
                    
                    mesh_data.weights = weights
                    mesh_data.bones_list = bone_ids
                    print(f"Extracted weights for {len(weights)} vertices referencing {len(bone_ids)} bones")
            else :
                
                print("Using direct bone ID approach - weights reference global bone IDs")
                
                
                vgroup_to_bone_id = {}  
                
                for vg in mesh.vertex_groups:
                    
                    if vg.name.startswith("Bone_"):
                        try:
                            bone_id = int(vg.name[5:])  
                            vgroup_to_bone_id[vg.index] = bone_id
                        except ValueError:
                            
                            pass
                
                if vgroup_to_bone_id:
                    print(f"Found {len(vgroup_to_bone_id)} bone vertex groups")
                    
                    
                    weights = []
                    for vert in mesh.data.vertices:
                        vert_weights = []
                        
                        for vg in vert.groups:
                            if vg.group in vgroup_to_bone_id and vg.weight > 0:
                                bone_id = vgroup_to_bone_id[vg.group]  
                                
                                weight_value = vg.weight * 100.0
                                print(f"bone_id: {bone_id}, weight_value: {weight_value}")
                                vert_weights.append((bone_id, weight_value))
                        
                        
                        vert_weights.sort(key=lambda x: x[1], reverse=True)
                        if len(vert_weights) > 4:
                            vert_weights = vert_weights[:4]
                        
                        
                        if vert_weights:
                            total_weight = sum(w for _, w in vert_weights)
                            if total_weight > 0:
                                vert_weights = [(idx, (w / total_weight) * 100.0) for idx, w in vert_weights]
                        
                        weights.append(vert_weights)
                    
                    mesh_data.weights = weights
                    mesh_data.bones_list = None  
                    print(f"Extracted weights for {len(weights)} vertices using direct bone IDs")


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

def write_attribute_block(attributes: list[int]) -> bytes:
    block_size = 12 + (len(attributes) * 4)
    header = write_directory_header(0xF0000, 1, block_size)
    
    data = bytearray()
    for value in attributes:
        data.extend(write_uint32_le(value))
        
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

def write_tpn_vec_block(tpn_vectors: list[tuple[float, float, float, float]]) -> bytes:
    block_size = 12 + (len(tpn_vectors) * 16)
    header = write_directory_header(0x120000, len(tpn_vectors), block_size)

    data = bytearray()
    for vec in tpn_vectors:
        data.extend(write_float_le(vec[0]))
        data.extend(write_float_le(vec[1]))
        data.extend(write_float_le(vec[2]))
        data.extend(write_float_le(vec[3]))
    
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
    
    strip_size = [len(strip) * 4 + 4 for strip in faces]
    block_size = 12 + sum(strip_size)
    header = write_directory_header(5, 1, block_size + 12) 
    strips_header = write_directory_header(0x30000, len(faces), block_size) 

    data = bytearray()
    for strip in faces:
        data.extend(write_uint32_le(len(strip))) 
        for index in strip:
            data.extend(write_uint32_le(index))
    
    return header + strips_header + data

def write_weight_block(weights: list[list[tuple[int, float]]]) -> bytes:
    """Write a vertex weight block with its header."""
    
    vertex_sizes = [4 + (len(weight_list) * 8) for weight_list in weights]
    block_size = 12 + sum(vertex_sizes)
    
    header = write_directory_header(0xC0000, len(weights), block_size)
    
    data = bytearray()
    for weight_list in weights:
        
        data.extend(write_uint32_le(len(weight_list)))
        
        
        for bone_idx, weight_value in weight_list:
            data.extend(write_uint32_le(bone_idx))
            data.extend(write_float_le(weight_value))
    
    return header + data

def write_bone_list_block(bones: list[int]) -> bytes:
    """Write a bone list block with its header."""
    block_size = 12 + (len(bones) * 4)  
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
    
    if mesh_data.tpn_vec:
        blocks.append(write_tpn_vec_block(mesh_data.tpn_vec))

    if mesh_data.attributes:
        blocks.append(write_attribute_block(mesh_data.attributes))

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
        block_size += 0x4 * (material_data.texture_count - 1)

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

def material_and_texture_data_from_material(material: bpy.types.Material, image_dict: dict[bpy.types.Image, int], global_texture_count: int, resize_option: str) -> tuple[ParsedMaterialData, list[ParsedTextureData]]:
    """Create material data and associated texture data blocks for a single material"""
    
    
    material_images = {
        'diffuse': None,
        'normal': None, 
        'specular': None
    }

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

                if from_node.type == 'TEX_IMAGE' and from_node.image:
                    if input_name == 'Base Color':
                        material_images['diffuse'] = from_node.image
                    elif input_name in ['Specular', 'Specular Tint', 'Specular IOR Level', 'Roughness']:
                        material_images['specular'] = from_node.image
                
                
                elif from_node.type == 'NORMAL_MAP':
                    for normal_link in material.node_tree.links:
                        if (normal_link.to_node == from_node and 
                            normal_link.from_node.type == 'TEX_IMAGE' and 
                            normal_link.from_node.image):
                            material_images['normal'] = normal_link.from_node.image

    
    texture_blocks = []
    global_texture_indices = {}
    
    current_global_idx = global_texture_count
    
    for tex_type, image in material_images.items():
        if image and image in image_dict:
            image_idx = image_dict[image]

            
            width, height = image.size[0], image.size[1]
            if resize_option != 'NONE':
                size = int(resize_option)
                width, height = size, size
            
            
            texture_data = ParsedTextureData(
                image_idx=image_idx,
                width=width,
                height=height,
                data=bytes([0] * 244)  
            )
            
            texture_blocks.append(texture_data)
            
            global_texture_indices[tex_type] = current_global_idx
            current_global_idx += 1

    
    diffuse_rgba = (1.0, 1.0, 1.0, 1.0)
    specular_rgba = (1.0, 1.0, 1.0, 0.0)
    ambient_rgba = (0.3, 0.3, 0.3, 0.0)
    specular_str = 50.0

    texture_count = len(texture_blocks)

    parsed_material_data = ParsedMaterialData(
        ambient_rgba=ambient_rgba,
        diffuse_rgba=diffuse_rgba,
        specular_rgba=specular_rgba,
        specular_str=specular_str,
        texture_count=texture_count,
        unknown_data=bytes([0] * 200),
        texture_diffuse=global_texture_indices.get('diffuse'),
        texture_normal=global_texture_indices.get('normal'),
        texture_specular=global_texture_indices.get('specular')
    )

    return parsed_material_data, texture_blocks