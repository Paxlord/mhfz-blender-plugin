import bpy
from numpy import mat #type: ignore
from mathutils import Matrix, Vector #type: ignore
from collections import defaultdict, deque

from .sgi_strips import *
from .data_classes import *

CHANNEL_NAME_TO_FLAG_MAP = {
    "scale.x": 0x01, "scale.y": 0x02, "scale.z": 0x04,
    "rotation_euler.x": 0x08, "rotation_euler.y": 0x10, "rotation_euler.z": 0x20,
    "location.x": 0x40, "location.y": 0x80, "location.z": 0x100,
}

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

    if parsed_data.tpn_vec:
        print(f"Applying custom tangents to mesh '{mesh_name}'")
        
        if not mesh_data.uv_layers:
            print(f"Mesh '{mesh_name}' has TPN data but no UVs. Tangents cannot be set.")
        else:
            try:
                mesh_data.calc_tangents(uvmap=mesh_data.uv_layers.active.name)
                
                tangent_attr = mesh_data.attributes.get('tangent')
                bitangent_sign_attr = mesh_data.attributes.get('bitangent_sign')

                vertex_to_tpn = {i: data for i, data in enumerate(parsed_data.tpn_vec)}

                num_loops = len(mesh_data.loops)
                new_tangents = [0.0] * (num_loops * 3)
                new_signs = [0.0] * num_loops

                for i, loop in enumerate(mesh_data.loops):
                    vert_idx = loop.vertex_index
                    if vert_idx in vertex_to_tpn:
                        tpn_data = vertex_to_tpn[vert_idx]
                        
                        tangent = (tpn_data[0], tpn_data[1], tpn_data[2])
                        sign = tpn_data[3]
                        
                        new_tangents[i*3 : i*3 + 3] = tangent
                        new_signs[i] = sign
                
                tangent_attr.data.foreach_set('vector', new_tangents)
                bitangent_sign_attr.data.foreach_set('value', new_signs)

                print(f"Successfully applied {len(parsed_data.tpn_vec)} custom tangents.")
                
            except Exception as e:
                print(f"Error applying custom tangents: {e}")


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

    mat.blend_method = 'CLIP'
    mat.alpha_threshold = 0.5

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
        links.new(diffuse_texture.outputs['Alpha'], principled_bsdf.inputs['Alpha'])
        if diffuse_texture.image:
            diffuse_texture.image.alpha_mode = 'CHANNEL_PACKED'
    
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
            armature_bone["part_id"] = bone.part_id
    
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

def extract_tpn_for_export(mesh_obj: bpy.types.Object) -> list[tuple[float, float, float, float]]:
    mesh_data = mesh_obj.data
    
    if not mesh_data.uv_layers:
        print(f"Warning: No UV layers on {mesh_obj.name}, cannot generate tangents")
        return []
    
    mesh_data.calc_tangents(uvmap=mesh_data.uv_layers.active.name)
    
    vertex_tangents = {}
    vertex_signs = {}
    vertex_counts = {}
    
    for poly in mesh_data.polygons:
        for loop_index in poly.loop_indices:
            loop = mesh_data.loops[loop_index]
            vert_idx = loop.vertex_index
            
            if vert_idx not in vertex_tangents:
                vertex_tangents[vert_idx] = [0.0, 0.0, 0.0]
                vertex_signs[vert_idx] = 0.0
                vertex_counts[vert_idx] = 0
            
            vertex_tangents[vert_idx][0] += loop.tangent[0]
            vertex_tangents[vert_idx][1] += loop.tangent[1] 
            vertex_tangents[vert_idx][2] += loop.tangent[2]
            vertex_signs[vert_idx] += loop.bitangent_sign
            vertex_counts[vert_idx] += 1
    
    tpn_data = []
    for i in range(len(mesh_data.vertices)):
        if i in vertex_tangents and vertex_counts[i] > 0:
            count = vertex_counts[i]
            avg_tangent = [
                vertex_tangents[i][0] / count,
                vertex_tangents[i][1] / count, 
                vertex_tangents[i][2] / count
            ]
            avg_sign = vertex_signs[i] / count
            
            length = (avg_tangent[0]**2 + avg_tangent[1]**2 + avg_tangent[2]**2)**0.5
            if length > 0:
                avg_tangent = [avg_tangent[0]/length, avg_tangent[1]/length, avg_tangent[2]/length]
            
            tpn_data.append((avg_tangent[0], avg_tangent[1], avg_tangent[2], -avg_sign))
        else:
            tpn_data.append((1.0, 0.0, 0.0, -1.0))
    
    return tpn_data

def apply_animation_to_armature(armature_obj: bpy.types.Object, aan_package: AANPackage, scale_factor: float, animation_mode: str = "monster", apply_filter: bool = True):
    context = bpy.context

    if not armature_obj or armature_obj.type != 'ARMATURE':
        print("Target object is not an armature.")
        return

    for pose_bone in armature_obj.pose.bones:
        pose_bone.rotation_mode = 'XYZ'

    if not armature_obj.animation_data:
        armature_obj.animation_data_create()

    context.scene.render.fps = 30
    print("Set scene framerate to 30 FPS.")

    bone_buckets = defaultdict(list)
    sorted_pose_bones = sorted(armature_obj.pose.bones, key=lambda b: int(b.name.split('_')[1]))

    max_part_id = -1
    for pose_bone in sorted_pose_bones:
        bone_data = armature_obj.data.bones.get(pose_bone.name)
        if bone_data and "part_id" in bone_data:
            part_id = bone_data["part_id"]
            bone_buckets[part_id].append(pose_bone)
            max_part_id = max(max_part_id, part_id)
        else:
            bone_buckets[0].append(pose_bone)
            max_part_id = max(max_part_id, 0)

    print(f"Max part ID found: {max_part_id}")
    print("Created Bone Buckets:")
    for part_id, bones in bone_buckets.items():
        print(f"  Bucket {part_id}: {len(bones)} bones - {[b.name for b in bones]}")

    EULER_CONVERSION_FACTOR = 0.00038349521
    MODEL_IMPORT_SCALE = scale_factor

    CHANNEL_TO_INDEX_MAP = {
        "location.x": 0, "location.y": 1, "location.z": 2,
        "rotation_euler.x": 0, "rotation_euler.y": 1, "rotation_euler.z": 2,
        "scale.x": 0, "scale.y": 1, "scale.z": 2
    }

    motions_by_part = defaultdict(dict)  
    for motion in aan_package.motions:
        motions_by_part[motion.part_index][motion.motion_slot_index] = motion

    print(f"Found motions in parts: {list(motions_by_part.keys())}")
    
    if animation_mode == "player":
        print("Using PLAYER animation mode")
        apply_player_animations(motions_by_part, bone_buckets, EULER_CONVERSION_FACTOR, MODEL_IMPORT_SCALE, CHANNEL_TO_INDEX_MAP, armature_obj, context, apply_filter)
    else:
        print("Using MONSTER animation mode") 
        apply_monster_animations(motions_by_part, bone_buckets, max_part_id, EULER_CONVERSION_FACTOR, MODEL_IMPORT_SCALE, CHANNEL_TO_INDEX_MAP, armature_obj, context, apply_filter)

def apply_player_animations(motions_by_part, bone_buckets, euler_factor, scale_factor, channel_map, armature_obj, context, apply_filter):
    part_pairs = []
    all_parts = sorted(motions_by_part.keys())
    
    for i in range(0, max(all_parts) + 1, 2):
        upper_part = i
        lower_part = i + 1
        if upper_part in motions_by_part or lower_part in motions_by_part:
            part_pairs.append((upper_part, lower_part))
    
    print(f"Processing {len(part_pairs)} part pairs: {part_pairs}")
    
    for upper_part, lower_part in part_pairs:
        upper_slots = set(motions_by_part.get(upper_part, {}).keys())
        lower_slots = set(motions_by_part.get(lower_part, {}).keys())
        all_slots = upper_slots | lower_slots

        
        
        print(f"Part pair ({upper_part},{lower_part}): Found {len(all_slots)} motion slots: {sorted(all_slots)}")
        
        for motion_slot_index in all_slots:
            action_name = f"Motion_{upper_part:02d}_{lower_part:02d}_{motion_slot_index:02d}"
            action = bpy.data.actions.new(name=action_name)
            print(f"\nCreating Action: {action_name}")
            
            main_motion = None
            if upper_part in motions_by_part and motion_slot_index in motions_by_part[upper_part]:
                main_motion = motions_by_part[upper_part][motion_slot_index]
            elif lower_part in motions_by_part and motion_slot_index in motions_by_part[lower_part]:
                main_motion = motions_by_part[lower_part][motion_slot_index]

            if main_motion:
                action["loops"] = main_motion.loops
                action["loop_start_frame"] = main_motion.loop_start_frame
                print(f"  Storing loop info on action: Loops={main_motion.loops}, Start Frame={main_motion.loop_start_frame}")

            min_frame = float('inf')
            max_frame = float('-inf')
            
            if upper_part in motions_by_part and motion_slot_index in motions_by_part[upper_part]:
                upper_motion = motions_by_part[upper_part][motion_slot_index]
                print(f"  Applying upper body motion from part {upper_part} to bucket 0")
                success = apply_motion_to_bucket(upper_motion, bone_buckets[0], action, 
                                               euler_factor, scale_factor, channel_map)
                if success:
                    motion_min, motion_max = get_motion_frame_range(upper_motion)
                    min_frame = min(min_frame, motion_min)
                    max_frame = max(max_frame, motion_max)
            else:
                print(f"  No upper body motion found in part {upper_part}")
            
            if lower_part in motions_by_part and motion_slot_index in motions_by_part[lower_part]:
                lower_motion = motions_by_part[lower_part][motion_slot_index]
                print(f"  Applying lower body motion from part {lower_part} to bucket 1")
                success = apply_motion_to_bucket(lower_motion, bone_buckets[1], action,
                                               euler_factor, scale_factor, channel_map)
                if success:
                    motion_min, motion_max = get_motion_frame_range(lower_motion)
                    min_frame = min(min_frame, motion_min)
                    max_frame = max(max_frame, motion_max)
            else:
                print(f"  No lower body motion found in part {lower_part}")
            
            if min_frame != float('inf') and max_frame != float('-inf'):
                action.frame_range = (min_frame, max_frame)
                print(f"  Set action frame range: {min_frame} to {max_frame}")

                if apply_filter:
                    _apply_euler_filter_to_action(action)
                
                if armature_obj.animation_data.action is None:
                    context.scene.frame_start = int(min_frame)
                    context.scene.frame_end = int(max_frame)
                    context.scene.frame_current = int(min_frame)
                    print(f"  Set scene timeline: {min_frame} to {max_frame}")
                    armature_obj.animation_data.action = action
    
    if armature_obj.animation_data.action is None and bpy.data.actions:
        armature_obj.animation_data.action = bpy.data.actions[-1]

def apply_monster_animations(motions_by_part, bone_buckets, max_part_id, euler_factor, scale_factor, channel_map, armature_obj, context, apply_filter):
    expected_parts = (max_part_id + 1) * 2
    print(f"Expected number of parts in animation file: {expected_parts}")
    
    processed_slots = set()
    
    for base_part in [0, 1]:
        if base_part not in motions_by_part:
            continue
            
        for motion_slot_index, base_motion in motions_by_part[base_part].items():
            if motion_slot_index in processed_slots:
                continue  
                
            action_name = f"Motion_{base_part:02d}_{motion_slot_index:02d}"
            action = bpy.data.actions.new(name=action_name)
            print(f"\nCreating Action: {action_name}")

            action["loops"] = base_motion.loops
            action["loop_start_frame"] = base_motion.loop_start_frame
            print(f"  Storing loop info on action: Loops={base_motion.loops}, Start Frame={base_motion.loop_start_frame}")
            
            min_frame = float('inf')
            max_frame = float('-inf')
            
            current_motion = base_motion
            bucket_index = 0
            part_index = base_part
            
            print(f"  Processing base motion from part {part_index} for bucket {bucket_index}")
            success = apply_motion_to_bucket(current_motion, bone_buckets[bucket_index], action, 
                                           euler_factor, scale_factor, channel_map)
            if success:
                motion_min, motion_max = get_motion_frame_range(current_motion)
                min_frame = min(min_frame, motion_min)
                max_frame = max(max_frame, motion_max)
            
            for bucket_index in range(1, max_part_id + 1):
                extended_part = base_part + (bucket_index * 2)
                
                if extended_part in motions_by_part and motion_slot_index in motions_by_part[extended_part]:
                    extended_motion = motions_by_part[extended_part][motion_slot_index]
                    print(f"  Found extended motion in part {extended_part} for bucket {bucket_index}")
                    
                    success = apply_motion_to_bucket(extended_motion, bone_buckets[bucket_index], action,
                                                   euler_factor, scale_factor, channel_map)
                    if success:
                        motion_min, motion_max = get_motion_frame_range(extended_motion)
                        min_frame = min(min_frame, motion_min)
                        max_frame = max(max_frame, motion_max)
                else:
                    print(f"  No extended motion found in part {extended_part} for bucket {bucket_index}")
            
            if min_frame != float('inf') and max_frame != float('-inf'):
                action.frame_range = (min_frame, max_frame)
                print(f"  Set action frame range: {min_frame} to {max_frame}")

                if apply_filter:
                    _apply_euler_filter_to_action(action)
                
                if armature_obj.animation_data.action is None:
                    context.scene.frame_start = int(min_frame)
                    context.scene.frame_end = int(max_frame)
                    context.scene.frame_current = int(min_frame)
                    print(f"  Set scene timeline: {min_frame} to {max_frame}")
                    armature_obj.animation_data.action = action
            
            processed_slots.add(motion_slot_index)

    if armature_obj.animation_data.action is None and bpy.data.actions:
        armature_obj.animation_data.action = bpy.data.actions[-1]
        
    print(f"Player animation import complete. Created actions for all part pairs.")

    if armature_obj.animation_data.action is None and bpy.data.actions:
        armature_obj.animation_data.action = bpy.data.actions[-1]

def apply_motion_to_bucket(motion, target_bones, action, euler_factor, scale_factor, channel_map):
    if not target_bones:
        return False
        
    applied_any = False
    for track_index, bone_track in motion.bone_tracks.items():
        if track_index >= len(target_bones):
            print(f"    Warning: Track index {track_index} is out of bounds for bucket (size {len(target_bones)}).")
            continue
        
        pose_bone = target_bones[track_index]
        print(f"    Applying track {track_index} to bone {pose_bone.name}")
        
        for component in bone_track.components:
            if not component.keyframes:
                continue

            data_path_base = component.channel_name.split('.')[0]
            array_index = channel_map[component.channel_name]

            fcurve = action.fcurves.new(data_path=f"pose.bones[\"{pose_bone.name}\"].{data_path_base}", index=array_index)
            fcurve.keyframe_points.add(count=len(component.keyframes))
            
            converted_keys = []
            for key in component.keyframes:
                is_short = "Short" in component.type_name
                
                value, time, tan_in, tan_out = 0, 0, 0, 0
                interp_flag = 0x20000

                if "Complex" in component.type_name:
                    interp_flag, value, time, tan_in, tan_out = key.data
                elif "Hermite" in component.type_name:
                    value, time, tan_in, tan_out = key.data
                elif "Linear" in component.type_name:
                    value, time = key.data
                    interp_flag = 0x10000

                frame_number = time

                if is_short:
                    if "location" in component.channel_name:
                        value /= 16.0
                        tan_in /= 16.0
                        tan_out /= 16.0
                    elif "rotation_euler" in component.channel_name:
                        value *= euler_factor
                        tan_in *= euler_factor
                        tan_out *= euler_factor
                    elif "scale" in component.channel_name:
                        value /= 16.0
                        tan_in /= 16.0
                        tan_out /= 16.0
                else: 
                    if "location" in component.channel_name:
                        value *= scale_factor
                        tan_in *= scale_factor
                        tan_out *= scale_factor

                float_precision = 6
                value = round(value, float_precision)
                tan_in = round(tan_in, float_precision)
                tan_out = round(tan_out, float_precision)
                frame_number = round(frame_number)

                converted_keys.append({'time': frame_number, 'value': value, 'tan_in': tan_in, 'tan_out': tan_out, 'interp_flag': interp_flag})
                
            for i, key_data in enumerate(converted_keys):
                kp = fcurve.keyframe_points[i]
                kp.co = key_data['time'], key_data['value']

                if key_data['interp_flag'] == 0x10000:
                    kp.interpolation = 'LINEAR'
                else:
                    kp.interpolation = 'BEZIER'
                    kp.handle_left_type = 'FREE'
                    kp.handle_right_type = 'FREE'

                    current_time = key_data['time']
                    current_value = key_data['value']
                    game_tan_in = key_data['tan_in']
                    game_tan_out = key_data['tan_out']

                    if i > 0:
                        prev_key = converted_keys[i - 1]
                        dt = current_time - prev_key['time']
                        
                        handle_time_offset = dt / 3.0
                        left_handle_x = current_time - handle_time_offset
                        left_handle_y = current_value - (game_tan_in * handle_time_offset)
                        kp.handle_left = (left_handle_x, left_handle_y)
                    else:
                        kp.handle_left = (current_time, current_value)
                    
                    if i < len(converted_keys) - 1:
                        next_key = converted_keys[i + 1]
                        dt = next_key['time'] - current_time
                        
                        handle_time_offset = dt / 3.0
                        right_handle_x = current_time + handle_time_offset
                        right_handle_y = current_value + (game_tan_out * handle_time_offset)
                        kp.handle_right = (right_handle_x, right_handle_y)
                    else:
                        kp.handle_right = (current_time, current_value)

            fcurve.update()
            applied_any = True
            
            for kp in fcurve.keyframe_points:
                if kp.interpolation == 'BEZIER':
                    kp.handle_left_type = 'FREE'
                    kp.handle_right_type = 'FREE'


            fcurve.update()
            applied_any = True
    
    return applied_any

def get_motion_frame_range(motion):
    min_frame = float('inf')
    max_frame = float('-inf')
    
    for bone_track in motion.bone_tracks.values():
        for component in bone_track.components:
            for key in component.keyframes:
                if "Complex" in component.type_name:
                    time = key.data[2]
                elif "Hermite" in component.type_name:
                    time = key.data[1]
                elif "Linear" in component.type_name:
                    time = key.data[1]
                else:
                    continue
                    
                min_frame = min(min_frame, time)
                max_frame = max(max_frame, time)
    
    return min_frame if min_frame != float('inf') else 0, max_frame if max_frame != float('-inf') else 0

def parse_action_name(action_name: str) -> tuple[list[int], int] | None:
    if not action_name.lower().startswith("motion_"):
        return None
    
    parts = action_name.split('_')
    if len(parts) < 3:
        return None
        
    try:
        part_indices = [int(p) for p in parts[1:-1]]
        motion_slot = int(parts[-1])
        return part_indices, motion_slot
    except (ValueError, IndexError):
        return None
    
def _apply_euler_filter_to_action(action: bpy.types.Action):
    if not action or not action.fcurves:
        return

    original_selection = {fcurve: fcurve.select for fcurve in action.fcurves}

    for fcurve in action.fcurves:
        fcurve.select = False
        if "rotation_euler" in fcurve.data_path:
            fcurve.select = True

    try:
        bpy.ops.graph.euler_filter()
        print(f"  Applied Euler Filter to action '{action.name}'.")
    except Exception as e:
        print(f"  Warning: Could not apply Euler Filter to action '{action.name}'. Operator failed: {e}")
    finally:
        # Restore original selection state
        for fcurve, is_selected in original_selection.items():
            if fcurve: # Check if fcurve still exists
                fcurve.select = is_selected

def get_animation_data_from_action(
    action: bpy.types.Action, 
    bone_to_bucket_map: dict[str, tuple[int, int]], 
    model_scale: float
) -> dict[int, dict[int, list[AANAnimComponent]]]:
    wip_data = defaultdict(lambda: defaultdict(dict))

    for fcurve in action.fcurves:
        try:
            bone_name = fcurve.data_path.split('"')[1]
        except IndexError:
            continue

        if bone_name not in bone_to_bucket_map:
            continue
            
        bucket_idx, bone_idx_in_bucket = bone_to_bucket_map[bone_name]

        prop_name = fcurve.data_path.split('.')[-1]
        channel_name = f"{prop_name}.{('x', 'y', 'z')[fcurve.array_index]}"

        if not fcurve.keyframe_points:
            continue

        keyframes = []
        is_linear = all(kp.interpolation == 'LINEAR' for kp in fcurve.keyframe_points)

        for i, kp in enumerate(fcurve.keyframe_points):
            time = kp.co.x
            value = kp.co.y
            
            if 'location' in channel_name:
                value /= model_scale

            if is_linear:
                key_data = (value, time)
                keyframes.append(Keyframe(data=key_data))
            else:
                current_time = kp.co.x
                current_value = kp.co.y 
                
                if i > 0:
                    prev_kp = fcurve.keyframe_points[i - 1]
                    dt = current_time - prev_kp.co.x
                    
                    expected_handle_time = dt / 3.0
                    actual_handle_dy = current_value - kp.handle_left.y
                    
                    if expected_handle_time != 0:
                        tan_in = actual_handle_dy / expected_handle_time
                    else:
                        tan_in = 0.0
                else:
                    tan_in = 0.0

                if i < len(fcurve.keyframe_points) - 1:
                    next_kp = fcurve.keyframe_points[i + 1]
                    dt = next_kp.co.x - current_time
                    
                    expected_handle_time = dt / 3.0
                    actual_handle_dy = kp.handle_right.y - current_value
                    
                    if expected_handle_time != 0:
                        tan_out = actual_handle_dy / expected_handle_time
                    else:
                        tan_out = 0.0
                else:
                    tan_out = 0.0
                
                if 'location' in channel_name:
                    tan_in /= model_scale
                    tan_out /= model_scale
                
                tan_in = round(tan_in, 6)
                tan_out = round(tan_out, 6)
                
                key_data = (value, time, tan_in, tan_out)
                keyframes.append(Keyframe(data=key_data))
        
        type_name = 'LinearFloat' if is_linear else 'HermiteFloat'
        component = AANAnimComponent(
            channel_name=channel_name,
            type_name=type_name,
            keyframes=keyframes
        )
        wip_data[bucket_idx][bone_idx_in_bucket][channel_name] = component

    final_data = defaultdict(lambda: defaultdict(list))
    for bucket_idx, bones in wip_data.items():
        for bone_idx, channels in bones.items():
            final_data[bucket_idx][bone_idx] = list(channels.values())
            
    return final_data