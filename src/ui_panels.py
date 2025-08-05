import bpy # type: ignore
from bpy.props import EnumProperty, StringProperty # type: ignore
from bpy.types import Panel, PropertyGroup, Operator # type: ignore
from .setup_tools import *
import os

def get_mesh_items(self, context):
    items = [('NONE', 'Select Mesh...', 'No mesh selected')]
    
    for obj in context.scene.objects:
        if obj.type == 'MESH':
            items.append((obj.name, obj.name, f'Mesh object: {obj.name}'))
    
    return items

def get_armature_items(self, context):
    items = [('NONE', 'Select Armature...', 'No armature selected')]
    
    for obj in context.scene.objects:
        if obj.type == 'ARMATURE':
            items.append((obj.name, obj.name, f'Armature object: {obj.name}'))
    
    return items

def update_item_type(self, context):
    self.item_subtype = 'NONE'

class FMOD_SetupProperties(PropertyGroup):
    target_mesh: EnumProperty(
        name="Target Mesh",
        description="Select the mesh to setup for FMOD export",
        items=get_mesh_items,
        default=0
    ) # type: ignore

    target_sub_mesh: EnumProperty(
        name="Target Sub Mesh",
        description="Select the sub mesh (shield) for SNS setup",
        items=get_mesh_items,
        default=0
    ) # type: ignore
    
    target_armature: EnumProperty(
        name="Target Armature", 
        description="Select the armature to parent the mesh to",
        items=get_armature_items,
        default=0
    ) # type: ignore
    
    item_type: EnumProperty(
        name="Item Type",
        description="Select the type of item you're setting up",
        items=[
            ('NONE', 'Select Type...', 'No type selected'),
            ('WEAPONS', 'Weapons', 'Weapon models (GS, LS, DS, SNS, Bow)'),
            ('ARMORS', 'Armors (Not implemented)', 'Armor pieces (Head, Body, Legs, Waist, Arms)'),
            ('OTHER', 'Other (Not implemented)', 'Other model types')
        ],
        default='NONE',
        update=update_item_type
    ) # type: ignore
    
    item_subtype: EnumProperty(
        name="Item Subtype",
        description="Select the specific subtype",
        items=lambda self, context: get_subtype_items(self, context),
        default=0
    ) # type: ignore

def get_subtype_items(self, context):
    setup_props = context.scene.fmod_setup_props
    
    if setup_props.item_type == 'WEAPONS':
        return [
            ('NONE', 'Select Weapon...', 'No weapon type selected'),
            ('GS', 'Great Sword', 'Great Sword setup'),
            ('LS', 'Long Sword', 'Long Sword setup'),
            ('SNS', 'Sword & Shield', 'Sword and Shield setup'),
            ('DS', 'Dual Swords', 'Dual Swords setup'),
            ('HAMMER', 'Hammer', 'Hammer setup'),
            ('HH', 'Hunting Horn', 'Hunting Horn setup'),
            ('LANCE', 'Lance', 'Lance setup'),
            ('GL', 'Gunlance', 'Gunlance setup'),
            ('TONFAS', 'Tonfas (Not implemented)', 'Tonfas setup'),
            ('SA', 'Switch-Axe (Not implemented)', 'Switch-axe setup'),
            ('MS', 'Magnet Spike (Not implemented)', 'Manget Spike setup'),
            ('BOW', 'Bow', 'Bow setup'),
            ('LBG', 'Light bowgun', 'Light bowgun setup'),
            ('HBG', 'Heavy bowgun', 'Heavy bowgun setup')
        ]
    elif setup_props.item_type == 'ARMORS':
        return [
            ('NONE', 'Select Armor...', 'No armor type selected'),
            ('HEAD', 'Head', 'Head armor piece'),
            ('BODY', 'Body', 'Body armor piece'),
            ('LEGS', 'Legs', 'Leg armor piece'),
            ('WAIST', 'Waist', 'Waist armor piece'),
            ('ARMS', 'Arms', 'Arm armor piece')
        ]
    else:
        return [
            ('NONE', 'Select Subtype...', 'No subtype available'),
            ('GENERIC', 'Generic', 'Generic model setup')
        ]

class FMOD_OT_SetupMesh(bpy.types.Operator):
    bl_idname = "fmod.setup_mesh"
    bl_label = "Setup FMOD Mesh"
    bl_description = "Automatically setup the selected mesh for FMOD export"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        setup_props = context.scene.fmod_setup_props
        
        mesh_name = setup_props.target_mesh
        sub_mesh_name = setup_props.target_sub_mesh
        armature_name = setup_props.target_armature

        item_type = setup_props.item_type
        item_subtype = setup_props.item_subtype
        
        # Validation
        if mesh_name == 'NONE':
            self.report({'ERROR'}, "Please select a target mesh")
            return {'CANCELLED'}
            
        if armature_name == 'NONE':
            self.report({'ERROR'}, "Please select a target armature")
            return {'CANCELLED'}
            
        if item_type == 'NONE':
            self.report({'ERROR'}, "Please select an item type")
            return {'CANCELLED'}
            
        if item_subtype == 'NONE':
            self.report({'ERROR'}, "Please select an item subtype")
            return {'CANCELLED'}
        
        mesh_obj = context.scene.objects.get(mesh_name)
        sub_mesh_obj = context.scene.objects.get(sub_mesh_name)
        armature_obj = context.scene.objects.get(armature_name)
        
        if not mesh_obj:
            self.report({'ERROR'}, f"Mesh '{mesh_name}' not found")
            return {'CANCELLED'}
            
        if not armature_obj:
            self.report({'ERROR'}, f"Armature '{armature_name}' not found")
            return {'CANCELLED'}
        
        print(f"Setting up {item_type} {item_subtype} for {mesh_name}")

        if item_subtype == "GS":
            success = setup_great_sword(mesh_obj, armature_obj)
        if item_subtype == "LS":
            success = setup_long_sword(mesh_obj, armature_obj)
        if item_subtype == "SNS":
            success = setup_sword_and_shield(mesh_obj, sub_mesh_obj, armature_obj)
        if item_subtype == "DS":
            success = setup_dual_swords(mesh_obj, sub_mesh_obj, armature_obj)
        if item_subtype == "HAMMER":
            success = setup_hammer(mesh_obj, armature_obj)
        if item_subtype == "HH":
            success = setup_hunting_horn(mesh_obj, armature_obj)
        if item_subtype == "LANCE": 
            success = setup_lance(mesh_obj, sub_mesh_obj, armature_obj)
        if item_subtype == "GL":
            success = setup_gunlance(mesh_obj, armature_obj)
        if item_subtype == "BOW":
            success = setup_bow(mesh_obj, armature_obj)
        if item_subtype == "LBG":
            success = setup_light_bowgun(mesh_obj, armature_obj)
        if item_subtype == "HBG":
            success = setup_heavy_bowgun(mesh_obj, armature_obj)
        
        success = False
        
        if success:
            return {'FINISHED'}
        else:
            return {'CANCELLED'}

class FMOD_PT_SetupPanel(Panel):
    bl_label = "FMOD Setup"
    bl_idname = "FMOD_PT_setup_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FMOD"
    
    def draw(self, context):
        layout = self.layout
        setup_props = context.scene.fmod_setup_props

        box = layout.box()
        box.label(text="Assets:", icon='ASSET_MANAGER')
        box.operator("fmod.import_player_base", text="Import Player Base", icon='ADD')
        
        layout.label(text="Prepare Mesh for FMOD Export", icon='TOOL_SETTINGS')
        layout.separator()
        
        box = layout.box()
        box.label(text="Object Selection:", icon='OBJECT_DATA')
        box.prop(setup_props, "target_mesh", text="Mesh")
        box.prop(setup_props, "target_sub_mesh", text="Sub Mesh (shields, dual blades, tonfas...)")
        box.prop(setup_props, "target_armature", text="Armature")
        
        layout.separator()
        
        box = layout.box()
        box.label(text="Item Configuration:", icon='MODIFIER')
        box.prop(setup_props, "item_type", text="Type")
        
        if setup_props.item_type != 'NONE':
            box.prop(setup_props, "item_subtype", text="Subtype")
        
        layout.separator()
        
        col = layout.column(align=True)
        
        can_setup = (setup_props.target_mesh != 'NONE' and 
                    setup_props.target_armature != 'NONE' and
                    setup_props.item_type != 'NONE' and
                    setup_props.item_subtype != 'NONE')
        
        if can_setup:
            col.operator("fmod.setup_mesh", text="Setup FMOD Mesh", icon='CHECKMARK')
        else:
            col.label(text="Please fill all fields above", icon='INFO')
            col.operator("fmod.setup_mesh", text="Setup FMOD Mesh", icon='X').enabled = False

class FMOD_OT_ImportPlayerBase(Operator):
    bl_idname = "fmod.import_player_base"
    bl_label = "Import Player Base"
    bl_description = "Imports a default player model (armature & meshes) for animation work"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        script_dir = os.path.dirname(__file__)
        blend_file_path = os.path.join(script_dir, "templates", "base_player_male.blend")

        if not os.path.exists(blend_file_path):
            self.report({'ERROR'}, f"Player base file not found. Expected at: {blend_file_path}")
            return {'CANCELLED'}

        collection_name = "PlayerBase"

        try:
            with bpy.data.libraries.load(blend_file_path, link=False, relative=False) as (data_from, data_to):
                if collection_name in data_from.collections:
                    data_to.collections = [collection_name]
                else:
                    self.report({'WARNING'}, f"Collection '{collection_name}' not found. Appending all objects.")
                    data_to.objects = data_from.objects
            
            linked_something = False
            if data_to.collections:
                for coll in data_to.collections:
                    if coll:
                        context.scene.collection.children.link(coll)
                        linked_something = True
                self.report({'INFO'}, f"Imported collection '{collection_name}' from player base model.")
            elif data_to.objects:
                for obj in data_to.objects:
                    if obj:
                        context.scene.collection.objects.link(obj)
                        linked_something = True
                self.report({'INFO'}, "Imported all objects from player base model.")
            
            if not linked_something:
                self.report({'WARNING'}, "No data was imported from the player base model file.")
                return {'CANCELLED'}

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Failed to import player base model: {str(e)}")
            return {'CANCELLED'}

        return {'FINISHED'}

# Registration
classes = (
    FMOD_SetupProperties,
    FMOD_OT_SetupMesh,
    FMOD_OT_ImportPlayerBase, 
    FMOD_PT_SetupPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.fmod_setup_props = bpy.props.PointerProperty(type=FMOD_SetupProperties)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.fmod_setup_props