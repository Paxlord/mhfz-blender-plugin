from dataclasses import dataclass

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
    attributes: list[int] | None = None

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