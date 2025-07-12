import bpy #type: ignore
import bpy_extras.io_utils #type: ignore
import os

from .data_classes import ParsedFMODData, ParsedFSKLData
from .file_parsers import *
from .model_utils import *

def find_texture_folder(fmod_path: str, manual_path: str = "") -> str:
    def contains_images(folder_path: str) -> bool:
        """Check if a folder contains image files"""
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return False
        
        image_extensions = {'.png', '.jpg', '.jpeg', '.tga', '.dds', '.bmp', '.tiff'}
        try:
            items = os.listdir(folder_path)
            for item in items:
                if os.path.isfile(os.path.join(folder_path, item)):
                    _, ext = os.path.splitext(item.lower())
                    if ext in image_extensions:
                        return True
        except OSError:
            pass
        return False
    
    
    fmod_dir = os.path.dirname(fmod_path)
    fmod_filename = os.path.basename(fmod_path)
    
    
    try:
        fmod_number = int(fmod_filename[:4])
    except (ValueError, IndexError):
        return None
    

    print(f"fmod number {fmod_number}")

    
    target_number = fmod_number + 2
    target_prefix = f"{target_number:04d}"  
    
    print(f"target_prefix {target_prefix}")

    
    try:
        items = os.listdir(fmod_dir)
        for item in items:
            if os.path.isdir(os.path.join(fmod_dir, item)) and item.startswith(target_prefix):
                candidate_path = os.path.join(fmod_dir, item)
                if contains_images(candidate_path):
                    return candidate_path
    except OSError:
        pass
    
    
    fmod_dir_name = os.path.basename(fmod_dir)
    try:
        containing_folder_number = int(fmod_dir_name[:4])
    except (ValueError, IndexError):
        return None
    
    
    target_number = containing_folder_number + 1
    target_prefix = f"{target_number:04d}"  
    
    
    parent_dir = os.path.dirname(fmod_dir)
    try:
        parent_items = os.listdir(parent_dir)
        for item in parent_items:
            if os.path.isdir(os.path.join(parent_dir, item)) and item.startswith(target_prefix):
                candidate_path = os.path.join(parent_dir, item)
                if contains_images(candidate_path):
                    return candidate_path
    except OSError:
        pass
    
    
    return None

def find_fskl_file(fmod_path: str) -> str:
    
    fmod_dir = os.path.dirname(fmod_path)
    items = os.listdir(fmod_dir)
    for item in items:
        if item.endswith(".fskl"):
            return os.path.join(fmod_dir, item)
    return None

class ImportFMOD(bpy.types.Operator, bpy_extras.io_utils.ImportHelper):
    """Import FMOD model"""
    bl_idname = "import_scene.fmod"
    bl_label = "Import FMOD"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: bpy.props.StringProperty(
        default="*.fmod",
        options={'HIDDEN'},
        maxlen=255,
    ) # type: ignore

    def execute(self, context):
        
        filepath = self.filepath
        texture_folder = find_texture_folder(filepath, "")
        skeleton_path = find_fskl_file(filepath)

        if not texture_folder:
            self.report({'ERROR'}, "Texture folder not found. Please ensure the texture folder is in the same directory as the FMOD file.")
            return {'CANCELLED'}

        
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

            
            texture_dic = load_texture_no_dupes(parsed_fmod_data, texture_folder)

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

            
            if armature_obj and mesh_objects:
                for mesh_obj in mesh_objects:
                    parent_mesh_to_armature(mesh_obj, armature_obj)
                print("Parented meshes to armature successfully")

        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        return {'FINISHED'}