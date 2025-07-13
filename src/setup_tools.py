import bpy #type: ignore

def apply_all_transforms(obj):
    print(f"Applying transforms to {obj.name}")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"Transforms applied to {obj.name}")
    return True

def parent_mesh_to_armature(mesh_obj, armature_obj):
    print(f"Parenting {mesh_obj.name} to {armature_obj.name}")
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.select_all(action='DESELECT')
    mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE')
    print(f"{mesh_obj.name} parented to {armature_obj.name}")
    return True

def create_vertex_groups(mesh_obj, armature_obj):
    print(f"Creating vertex groups for {mesh_obj.name}")
    mesh_obj.vertex_groups.clear()
    for bone in armature_obj.data.bones:
        mesh_obj.vertex_groups.new(name=bone.name)
        print(f"Created vertex group: {bone.name}")
    print(f"Created {len(armature_obj.data.bones)} vertex groups")
    return True

def duplicate_material(mesh_obj, nb):
    if not mesh_obj or mesh_obj.type != 'MESH':
        return False
    if not mesh_obj.data.materials or len(mesh_obj.data.materials) == 0:
        return False
    first_material = mesh_obj.data.materials[0]
    if not first_material:
        return False
    for i in range(nb):
        new_material = first_material.copy()
        mesh_obj.data.materials.append(new_material)
    
    return True

def rename_mesh_materials(mesh_obj, names):
    if not mesh_obj or mesh_obj.type != 'MESH':
        return False
    if not mesh_obj.data.materials:
        return False
    for i, name in enumerate(names):
        if i < len(mesh_obj.data.materials) and mesh_obj.data.materials[i]:
            mesh_obj.data.materials[i].name = name
    return True

def assign_faces_to_material(mesh_obj, material_slot):
    if not mesh_obj or mesh_obj.type != 'MESH':
        return False
    if material_slot >= len(mesh_obj.data.materials) or material_slot < 0:
        return False
    for face in mesh_obj.data.polygons:
        face.material_index = material_slot
    return True

def assign_weight_to_bone(mesh_obj, bone_id, weight):
    """Set all vertices of mesh to bone named 'Bone_{bone_id}' with specified weight"""
    if not mesh_obj or mesh_obj.type != 'MESH':
        return False
    
    bone_name = f"Bone_{bone_id}"
    
    if bone_name not in mesh_obj.vertex_groups:
        return False
    
    vertex_group = mesh_obj.vertex_groups[bone_name]
    vertex_indices = [v.index for v in mesh_obj.data.vertices]
    vertex_group.add(vertex_indices, weight, 'REPLACE')
    
    return True


def setup_great_sword(mesh_obj, armature_obj):
    """Complete setup for Great Sword weapons"""
    if not mesh_obj or not armature_obj:
        return False
    
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    if not assign_faces_to_material(mesh_obj, 0):
        print("Failed to assign faces to material 0")
        return False
    
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Great Sword setup completed for {mesh_obj.name}")
    return True

def setup_long_sword(mesh_obj, armature_obj):
    if not mesh_obj or not armature_obj:
        return False
    
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    if not duplicate_material(mesh_obj, 4):
        print("Failed to duplicate materials")
        return False
    
    material_names = ["Blade", "Mat1", "Mat2", "Sheathe1", "Sheathe2"]
    if not rename_mesh_materials(mesh_obj, material_names):
        print("Failed to rename materials")
        return False
    
    if not assign_faces_to_material(mesh_obj, 0):
        print("Failed to assign faces to Blade material")
        return False
    
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Long Sword setup completed for {mesh_obj.name}")
    return True

def setup_sword_and_shield(sword_mesh_obj, shield_mesh_obj, armature_obj):
    """Complete setup for Sword and Shield weapons"""
    if not sword_mesh_obj or not shield_mesh_obj or not armature_obj:
        return False
    
    # Setup sword mesh
    if not apply_all_transforms(sword_mesh_obj):
        print("Failed to apply transforms to sword")
        return False
    
    if not assign_faces_to_material(sword_mesh_obj, 0):
        print("Failed to assign sword faces to material 0")
        return False
    
    if not parent_mesh_to_armature(sword_mesh_obj, armature_obj):
        print("Failed to parent sword to armature")
        return False
    
    if not create_vertex_groups(sword_mesh_obj, armature_obj):
        print("Failed to create vertex groups for sword")
        return False
    
    if not assign_weight_to_bone(sword_mesh_obj, 1, 1.0):
        print("Failed to assign sword weight to Bone_1")
        return False
    
    # Setup shield mesh
    if not apply_all_transforms(shield_mesh_obj):
        print("Failed to apply transforms to shield")
        return False
    
    if not assign_faces_to_material(shield_mesh_obj, 0):
        print("Failed to assign shield faces to material 0")
        return False
    
    if not parent_mesh_to_armature(shield_mesh_obj, armature_obj):
        print("Failed to parent shield to armature")
        return False
    
    if not create_vertex_groups(shield_mesh_obj, armature_obj):
        print("Failed to create vertex groups for shield")
        return False
    
    if not assign_weight_to_bone(shield_mesh_obj, 4, 1.0):
        print("Failed to assign shield weight to Bone_4")
        return False
    
    print(f"Sword and Shield setup completed for {sword_mesh_obj.name} and {shield_mesh_obj.name}")
    return True

def setup_dual_swords(blade1_mesh_obj, blade2_mesh_obj, armature_obj):
    """Complete setup for Dual Swords weapons"""
    if not blade1_mesh_obj or not blade2_mesh_obj or not armature_obj:
        return False
    
    # Setup first blade
    if not apply_all_transforms(blade1_mesh_obj):
        print("Failed to apply transforms to blade 1")
        return False
    
    if not assign_faces_to_material(blade1_mesh_obj, 0):
        print("Failed to assign blade 1 faces to material 0")
        return False
    
    if not parent_mesh_to_armature(blade1_mesh_obj, armature_obj):
        print("Failed to parent blade 1 to armature")
        return False
    
    if not create_vertex_groups(blade1_mesh_obj, armature_obj):
        print("Failed to create vertex groups for blade 1")
        return False
    
    if not assign_weight_to_bone(blade1_mesh_obj, 1, 1.0):
        print("Failed to assign blade 1 weight to Bone_1")
        return False
    
    # Setup second blade
    if not apply_all_transforms(blade2_mesh_obj):
        print("Failed to apply transforms to blade 2")
        return False
    
    if not assign_faces_to_material(blade2_mesh_obj, 0):
        print("Failed to assign blade 2 faces to material 0")
        return False
    
    if not parent_mesh_to_armature(blade2_mesh_obj, armature_obj):
        print("Failed to parent blade 2 to armature")
        return False
    
    if not create_vertex_groups(blade2_mesh_obj, armature_obj):
        print("Failed to create vertex groups for blade 2")
        return False
    
    if not assign_weight_to_bone(blade2_mesh_obj, 4, 1.0):
        print("Failed to assign blade 2 weight to Bone_4")
        return False
    
    print(f"Dual Swords setup completed for {blade1_mesh_obj.name} and {blade2_mesh_obj.name}")
    return True

def setup_lance(lance_mesh_obj, shield_mesh_obj, armature_obj):
    """Complete setup for Lance weapons"""
    if not lance_mesh_obj or not shield_mesh_obj or not armature_obj:
        return False
    
    # Setup lance mesh
    if not apply_all_transforms(lance_mesh_obj):
        print("Failed to apply transforms to lance")
        return False
    
    if not assign_faces_to_material(lance_mesh_obj, 0):
        print("Failed to assign lance faces to material 0")
        return False
    
    if not parent_mesh_to_armature(lance_mesh_obj, armature_obj):
        print("Failed to parent lance to armature")
        return False
    
    if not create_vertex_groups(lance_mesh_obj, armature_obj):
        print("Failed to create vertex groups for lance")
        return False
    
    if not assign_weight_to_bone(lance_mesh_obj, 1, 1.0):
        print("Failed to assign lance weight to Bone_1")
        return False
    
    # Setup shield mesh
    if not apply_all_transforms(shield_mesh_obj):
        print("Failed to apply transforms to shield")
        return False
    
    if not assign_faces_to_material(shield_mesh_obj, 0):
        print("Failed to assign shield faces to material 0")
        return False
    
    if not parent_mesh_to_armature(shield_mesh_obj, armature_obj):
        print("Failed to parent shield to armature")
        return False
    
    if not create_vertex_groups(shield_mesh_obj, armature_obj):
        print("Failed to create vertex groups for shield")
        return False
    
    if not assign_weight_to_bone(shield_mesh_obj, 4, 1.0):
        print("Failed to assign shield weight to Bone_4")
        return False
    
    print(f"Lance setup completed for {lance_mesh_obj.name} and {shield_mesh_obj.name}")
    return True

def setup_gunlance(mesh_obj, armature_obj):
    """Complete setup for Gunlance weapons"""
    if not mesh_obj or not armature_obj:
        return False
    
    # Apply all transformations
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    # Create 5 material slots (duplicate first material 4 times)
    if not duplicate_material(mesh_obj, 4):
        print("Failed to duplicate materials")
        return False
    
    # Rename materials
    material_names = ["Shield", "Base", "Mat2", "Barrel", "WyverFire"]
    if not rename_mesh_materials(mesh_obj, material_names):
        print("Failed to rename materials")
        return False
    
    # Assign all faces to material slot 1 (Base)
    if not assign_faces_to_material(mesh_obj, 1):
        print("Failed to assign faces to Base material")
        return False
    
    # Parent mesh to armature
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    # Create vertex groups for all bones
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    # Set all vertices to Bone_1 with weight 1.0
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Gunlance setup completed for {mesh_obj.name}")
    return True

def setup_hammer(mesh_obj, armature_obj):
    """Complete setup for Hammer weapons"""
    if not mesh_obj or not armature_obj:
        return False
    
    # Apply all transformations
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    # Assign all faces to material 0
    if not assign_faces_to_material(mesh_obj, 0):
        print("Failed to assign faces to material 0")
        return False
    
    # Parent mesh to armature
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    # Create vertex groups for all bones
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    # Set all vertices to Bone_1 with weight 1.0
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Hammer setup completed for {mesh_obj.name}")
    return True

def setup_hunting_horn(mesh_obj, armature_obj):
    """Complete setup for Hunting Horn weapons"""
    if not mesh_obj or not armature_obj:
        return False
    
    # Apply all transformations
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    # Create 2 material slots (duplicate first material 1 time)
    if not duplicate_material(mesh_obj, 1):
        print("Failed to duplicate materials")
        return False
    
    # Rename materials
    material_names = ["Base", "VibratingHorn"]
    if not rename_mesh_materials(mesh_obj, material_names):
        print("Failed to rename materials")
        return False
    
    # Assign all faces to material slot 0 (Base)
    if not assign_faces_to_material(mesh_obj, 0):
        print("Failed to assign faces to Base material")
        return False
    
    # Parent mesh to armature
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    # Create vertex groups for all bones
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    # Set all vertices to Bone_1 with weight 1.0
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Hunting Horn setup completed for {mesh_obj.name}")
    return True

def setup_bow(mesh_obj, armature_obj):
    """Complete setup for Bow weapons"""
    if not mesh_obj or not armature_obj:
        return False
    
    # Apply all transformations
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    # Create 5 material slots (duplicate first material 4 times)
    if not duplicate_material(mesh_obj, 4):
        print("Failed to duplicate materials")
        return False
    
    # Rename materials
    material_names = ["Quiver", "Mat1", "Mat2", "String", "Bow"]
    if not rename_mesh_materials(mesh_obj, material_names):
        print("Failed to rename materials")
        return False
    
    # Assign all faces to material slot 4 (Bow)
    if not assign_faces_to_material(mesh_obj, 4):
        print("Failed to assign faces to Quiver material")
        return False
    
    # Parent mesh to armature
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    # Create vertex groups for all bones
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    # Set all vertices to Bone_1 with weight 1.0
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Bow setup completed for {mesh_obj.name}")
    return True

def setup_light_bowgun(mesh_obj, armature_obj):
    """Complete setup for Light Bowgun weapons"""
    if not mesh_obj or not armature_obj:
        return False
    
    # Apply all transformations
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    # Create 3 material slots (duplicate first material 2 times)
    if not duplicate_material(mesh_obj, 2):
        print("Failed to duplicate materials")
        return False
    
    # Rename materials
    material_names = ["Base", "BarrelInner", "BarrelOuter"]
    if not rename_mesh_materials(mesh_obj, material_names):
        print("Failed to rename materials")
        return False
    
    # Assign all faces to material slot 0 (Base)
    if not assign_faces_to_material(mesh_obj, 0):
        print("Failed to assign faces to Base material")
        return False
    
    # Parent mesh to armature
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    # Create vertex groups for all bones
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    # Set all vertices to Bone_1 with weight 1.0
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Light Bowgun setup completed for {mesh_obj.name}")
    return True

def setup_heavy_bowgun(mesh_obj, armature_obj):
    """Complete setup for Heavy Bowgun weapons"""
    if not mesh_obj or not armature_obj:
        return False
    
    # Apply all transformations
    if not apply_all_transforms(mesh_obj):
        print("Failed to apply transforms")
        return False
    
    # Create 4 material slots (duplicate first material 3 times)
    if not duplicate_material(mesh_obj, 3):
        print("Failed to duplicate materials")
        return False
    
    # Rename materials
    material_names = ["Base", "Barrel", "Shield", "Mat3"]
    if not rename_mesh_materials(mesh_obj, material_names):
        print("Failed to rename materials")
        return False
    
    # Assign all faces to material slot 0 (Base)
    if not assign_faces_to_material(mesh_obj, 0):
        print("Failed to assign faces to Base material")
        return False
    
    # Parent mesh to armature
    if not parent_mesh_to_armature(mesh_obj, armature_obj):
        print("Failed to parent mesh to armature")
        return False
    
    # Create vertex groups for all bones
    if not create_vertex_groups(mesh_obj, armature_obj):
        print("Failed to create vertex groups")
        return False
    
    # Set all vertices to Bone_1 with weight 1.0
    if not assign_weight_to_bone(mesh_obj, 1, 1.0):
        print("Failed to assign weight to Bone_1")
        return False
    
    print(f"Heavy Bowgun setup completed for {mesh_obj.name}")
    return True