bl_info = {
    "name": "FMOD Importer",
    "blender": (3, 0, 0),
    "category": "Import-Export",
    "author": "Pax",
    "version": (1, 0),
    "description": "Import AMO/FMOD Mhfz model format into Blender",
    "location": "File > Import > FMOD (.fmod)",
}

import bpy #type: ignore

from .import_operators import *
from .export_operators import *

classes = (
    ImportFMOD,
    ExportFMOD,
)

def menu_func_import(self, context):
    self.layout.operator(import_operators.ImportFMOD.bl_idname, text="FMOD (.fmod)")

def menu_func_export(self, context):
    self.layout.operator(export_operators.ExportFMOD.bl_idname, text="FMOD (.fmod)")
    pass

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()