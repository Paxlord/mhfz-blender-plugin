import bpy #type: ignore
import bpy_extras.io_utils #type: ignore
import os

from .file_writers import *
from .model_utils import *

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
    ) # type: ignore

    write_log_files: bpy.props.BoolProperty(
        name="Write Log File",
        description="Write a log file with the export process",
        default=False,
    ) # type: ignore

    include_bone_list: bpy.props.BoolProperty(
        name="Include Bone List",
        description="Include bone list block in mesh (weights reference local indices). If disabled, weights reference global bone IDs directly",
        default=True,
    ) # type: ignore

    force_resize_items = [
        ('NONE', "None", "Do not resize textures"),
        ('128', "128x128", "Resize textures to 128x128"),
        ('256', "256x256", "Resize textures to 256x256"),
        ('512', "512x512", "Resize textures to 512x512"),
    ]
    force_texture_resize: bpy.props.EnumProperty(
        name="Force Texture Resize",
        description="Force resize all exported textures to a specific dimension",
        items=force_resize_items,
        default='NONE',
    ) # type: ignore

    export_type_items = [
        ('WEAPON', "Weapon", "Exports with a weapon folder structure"),
        ('ARMOR', "Armor", "Exports with an armor folder structure")
    ]
    export_type: bpy.props.EnumProperty(
        name="Export Type",
        description="Specify the type of folder structure for the exported model",
        items=export_type_items,
        default="WEAPON",
    ) #type: ignore

    def execute(self, context):

        image_dict = collect_unique_images(context)
        print(f"Collected {len(image_dict)} unique images from the scene")

        materials = collect_scene_materials(context)
        print(f"Collected {len(materials)} materials from the scene")

        materials_data = []
        all_texture_data = []
        material_names = []
        global_texture_count = 0

        for material, idx in sorted(materials.items(), key=lambda item: item[1]):
            material_names.append(material.name)
            
            
            material_data, texture_blocks = material_and_texture_data_from_material(material, image_dict, global_texture_count, self.force_texture_resize)
            materials_data.append(material_data)
            
            
            all_texture_data.extend(texture_blocks)

            global_texture_count += len(texture_blocks)
            
            print(f"Material '{material.name}': created {len(texture_blocks)} texture blocks")
    
        print(f"Generated material data for {len(materials_data)} materials")
        print(f"Generated texture data for {len(all_texture_data)} texture blocks")

        meshes = collect_scene_meshes(context)
        print(f"Collected {len(meshes)} meshes from the scene")
        mesh_names = [mesh.name for mesh, _ in sorted(meshes.items(), key=lambda item: item[1])]
        print(f"Mesh names: {mesh_names}")
        meshes_data = []
        for mesh, idx in meshes.items():
            meshes_data.append(mesh_data_from_mesh(mesh, material_names, self.include_bone_list))
        print(f"Generated mesh data for {len(meshes_data)} meshes")

        
        parsed_fmod_data = ParsedFMODData(meshes=meshes_data, textures=all_texture_data, materials=materials_data)
        print("Parsed FMOD data successfully")

        
        armature = collect_scene_armature(context)
        if armature:
            armature_obj = list(armature.keys())[0]
            print(f"Found armature '{armature_obj.name}' in the scene")
            parsed_fskl_data = fskl_data_from_armature(armature_obj)
            print("Parsed skeleton data successfully")
        else:
            parsed_fskl_data = None
            print("No armature found in the scene")

        
        fmod_buffer = write_fmod(parsed_fmod_data)
        print("Generated FMOD buffer successfully")

        fskl_buffer = None
        if parsed_fskl_data:
            fskl_buffer = write_fskl(parsed_fskl_data)
            print("Generated FSKL buffer successfully")

        
        file_name = os.path.basename(self.filepath)
        
        directory = os.path.dirname(self.filepath)
        export_directory = os.path.join(directory, file_name)
        if not os.path.exists(export_directory):
            os.makedirs(export_directory)
            print(f"Created directory: {export_directory}")

        
        fmod_name = "0001_0000001C.fmod"
        fmod_name_no_ext = fmod_name.split(".")[0]
        self.filepath = os.path.join(export_directory, fmod_name)

        
        with open(self.filepath, 'wb') as fmod_file:
            fmod_file.write(fmod_buffer)
            print(f"Exported FMOD data to {self.filepath}")

        
        texture_unique_id = hex(random.randint(0x300000, 0xA000000))[2:].upper()
        texture_unique_id = texture_unique_id.zfill(8)
        
        texture_directory_name = f"0003_{texture_unique_id}"
        
        texture_directory_path = os.path.join(export_directory, texture_directory_name)
        if self.export_type == "ARMOR":
            texture_directory_path = export_directory

        if not os.path.exists(texture_directory_path):
            os.makedirs(texture_directory_path)
            print(f"Created directory: {texture_directory_path}")

        
        texture_file_names = []
        index_off = 1
        if self.export_type == "ARMOR":
            index_off = 3
        for texture, index in image_dict.items():
            padded_index = str(index+index_off).zfill(4)
            random_hex = hex(random.randint(0x00000000, 0xFFFFFFFF))[2:].upper()
            padded_hex = random_hex.zfill(8)
            texture_name = f"{padded_index}_{padded_hex}.png"
            texture_path = os.path.join(texture_directory_path, texture_name)
            
            image_to_save = texture
            is_copy = False
            
            try:
                
                if self.force_texture_resize != 'NONE':
                    size = int(self.force_texture_resize)
                    
                    if texture.size[0] != size or texture.size[1] != size:
                        print(f"Resizing texture '{texture.name}' to {size}x{size}...")
                        image_to_save = texture.copy()
                        image_to_save.scale(size, size)
                        is_copy = True

                
                temp_scene = bpy.data.scenes.new("RGB_Export")
                temp_scene.render.image_settings.file_format = 'PNG'
                temp_scene.render.image_settings.color_mode = 'RGB'
                temp_scene.render.image_settings.color_depth = '8'
                temp_scene.render.image_settings.compression = 15
                
                
                image_to_save.save_render(filepath=texture_path, scene=temp_scene)
                
                
                bpy.data.scenes.remove(temp_scene)
                if is_copy:
                    bpy.data.images.remove(image_to_save)
        
                texture_file_names.append(texture_name)
                print(f"Exported texture {texture.name} to {texture_path}")
            except Exception as e:
                print(f"Failed to export texture {texture.name}: {e}")
                
                if is_copy:
                    bpy.data.images.remove(image_to_save, do_unlink=True)

        
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

        
        
        
        if self.write_log_files:
            with open(os.path.join(texture_directory_path, f"{texture_directory_name}.log"), 'w') as log_file:
                log_file.write("SimpleArchive\n")
                log_file.write(f"{texture_directory_name}.bin\n")
                log_file.write(f"{len(texture_file_names)}\n")
                for texture in texture_file_names:
                    log_file.write(f"{texture_name},12,13526,1196314761\n") 

            
            with open(os.path.join(export_directory, f"{file_name}.log"), 'w') as log_file:
                log_file.write("SimpleArchive\n")
                log_file.write(f"{file_name}.bin\n")
                log_file.write(f"3\n")
                log_file.write(f"{fmod_name},12,13526,1196314761\n") 
                if fskl_name:
                    log_file.write(f"{fskl_name}.fskl,12,13526,1196314761\n") 
                log_file.write(f"{texture_directory_name}.bin,12,13526,1196314761\n")

        return {'FINISHED'}