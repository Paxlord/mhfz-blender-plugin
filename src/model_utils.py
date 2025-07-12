import bpy #type: ignore
from mathutils import Matrix, Vector #type: ignore

from .sgi_strips import *
from .data_classes import *


def triangulate_strips(strips: list[list[int]], material_indices: list[int], material_list: list[int]) -> list[int]:
    triangle_indices = []
    triangle_materials = []

    for strip_idx, strip in enumerate(strips):
        if len(strip) < 3:
            continue
        
        
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

def create_simple_strips(triangles: list[list[int]], material_indices: list[int], material_list: list[int]) -> tuple[list[list[int]], list[int]]:
    strips = []
    strip_material_indices = []
    
    
    for mat_idx in material_indices:
        if mat_idx not in material_list:
            material_list.append(mat_idx)
    
    for i, triangle in enumerate(triangles):
        
        strips.append(triangle)
        
        
        mat_idx = material_indices[i]
        palette_idx = material_list.index(mat_idx)
        strip_material_indices.append(palette_idx)
    
    return strips, strip_material_indices

def create_sgi_strips(triangles: list[list[int]], material_indices: list[int], material_list: list[int]) -> tuple[list[list[int]], list[int]]:
    """
    Creates optimized triangle strips using the SGI algorithm, respecting material boundaries.
    
    Args:
        triangles: List of all triangles in the mesh.
        material_indices: List of material indices per triangle.
        material_list: The material palette to be populated or referenced.
        
    Returns:
        tuple: (list_of_strips, list_of_strip_material_indices)
    """
    final_strips = []
    final_strip_materials = []

    
    material_groups = {}
    print("Grouping triangles by mats")
    for i, tri in enumerate(triangles):
        mat_idx = material_indices[i]
        if mat_idx not in material_groups:
            material_groups[mat_idx] = []
        material_groups[mat_idx].append(tri)

    print(f"Generated {len(material_groups)} groups")

    
    for mat_idx, tris_for_material in material_groups.items():
        print("Generating strips for material")
        if not tris_for_material:
            continue

        
        if mat_idx not in material_list:
            material_list.append(mat_idx)
        palette_idx = material_list.index(mat_idx)

        
        striper = Striper()
        
        
        print("Stripifying")
        material_strips = striper.stripify(
            faces=tris_for_material, 
            sgi_algorithm=True, 
            one_sided=True, 
            connect_all_strips=False
        )
        
        final_strips.extend(material_strips)
        final_strip_materials.extend([palette_idx] * len(material_strips))

    return final_strips, final_strip_materials

def parent_mesh_to_armature(mesh_obj, armature_obj):
    """Parent a mesh object to an armature object and add an armature modifier."""
    
    modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    modifier.object = armature_obj
    modifier.use_vertex_groups = True
    modifier.use_bone_envelopes = False
    
    
    mesh_obj.parent = armature_obj
    
    return mesh_obj

def create_blender_mesh(obj_name: str, mesh_name: str, parsed_data: ParsedMeshData, blender_materials: list[bpy.types.Material] = None, scale_factor=0.01):
    
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
    
        
    if parsed_data.weights:
        print(f"Applying weights to mesh '{mesh_name}' with {len(parsed_data.weights)} weighted vertices")
        
        if parsed_data.bones_list:
            
            print(f"Using bone_list mapping with {len(parsed_data.bones_list)} bones")
            
            
            for bone_id in parsed_data.bones_list:
                bone_name = f"Bone_{bone_id}"
                obj.vertex_groups.new(name=bone_name)
            
            
            for vertex_idx, weight_data in enumerate(parsed_data.weights):
                for bone_idx, weight_value in weight_data:
                    if bone_idx < len(parsed_data.bones_list):
                        bone_id = parsed_data.bones_list[bone_idx]
                        bone_name = f"Bone_{bone_id}"
                        
                        normalized_weight = weight_value / 100.0
                        obj.vertex_groups[bone_name].add([vertex_idx], normalized_weight, 'REPLACE')
        else:
            
            print("No bone_list found, using bone indices directly as global bone IDs")
            
            
            used_bone_indices = set()
            for weight_data in parsed_data.weights:
                for bone_idx, _ in weight_data:
                    used_bone_indices.add(bone_idx)
            
            
            for bone_idx in used_bone_indices:
                bone_name = f"Bone_{bone_idx}"
                obj.vertex_groups.new(name=bone_name)
            
            
            for vertex_idx, weight_data in enumerate(parsed_data.weights):
                for bone_idx, weight_value in weight_data:
                    bone_name = f"Bone_{bone_idx}"
                    
                    normalized_weight = weight_value / 100.0
                    obj.vertex_groups[bone_name].add([vertex_idx], normalized_weight, 'REPLACE')

    return obj

def create_blender_material(mat_name: str, parsed_material: ParsedMaterialData, texture_dic: dict[int, bpy.types.Image]):
    mat_name = f"Mat_{mat_name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    mat.blend_method = 'HASHED'

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
    
    
    
    
    
    

    
    

    
    if parsed_material.texture_diffuse is not None:
        diffuse_texture = nodes.new(type='ShaderNodeTexImage')
        diffuse_texture.location = (-400, 0)
        diffuse_texture.image = texture_dic.get(parsed_material.texture_diffuse)
        links.new(diffuse_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])

    
    if parsed_material.texture_normal is not None:
        normal_texture = nodes.new(type='ShaderNodeTexImage')
        normal_texture.location = (-400, -200)
        normal_texture.image = texture_dic.get(parsed_material.texture_normal)
        normal_map = nodes.new(type='ShaderNodeNormalMap')
        normal_map.location = (-200, -200)
        normal_texture.image.colorspace_settings.name = 'Non-Color'
        links.new(normal_texture.outputs['Color'], normal_map.inputs['Color'])
        links.new(normal_map.outputs['Normal'], principled_bsdf.inputs['Normal'])

    
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
    
    arm_data = bpy.data.armatures.new(armature_name)
    armature_obj = bpy.data.objects.new(armature_name, arm_data)
    bpy.context.collection.objects.link(armature_obj)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bone_length = 100 

    
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

    
    edit_bones = {}
    for bone in skeleton_data.bones:
        global_head = get_global_position(bone)
        edit_bone = arm_data.edit_bones.new(f"Bone_{bone.node_id}")
        edit_bone.head = global_head
        edit_bone.tail = global_head + Vector((0, bone_length, 0))
        edit_bones[bone.node_id] = edit_bone

    
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

    
    
    
    
    return armature_obj

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

def collect_unique_images(context):
    """Collect all unique images used in materials across the scene"""
    unique_images = {}
    image_idx = 0

    for obj in context.scene.objects:
        if not hasattr(obj.data, "materials") or not obj.data.materials:
            continue

        for mat_slots in obj.material_slots:
            if not mat_slots.material or not mat_slots.material.use_nodes:
                continue

            for node in mat_slots.material.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.image:
                    if node.image not in unique_images:
                        unique_images[node.image] = image_idx
                        image_idx += 1
    
    return unique_images