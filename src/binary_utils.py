import struct

def read_uint32_le(buf, off): 
    return struct.unpack_from("<I", buf, off)[0]

def read_uint16_le(buf, off):
    return struct.unpack_from("<H", buf, off)[0]

def read_uint8(buf, off):
    return struct.unpack_from("<B", buf, off)[0]

def read_float_le(buf, off):
    return struct.unpack_from("<f", buf, off)[0]

def write_uint32_le(value: int) -> bytes:
    return struct.pack("<I", value)

def write_uint16_le(value: int) -> bytes:
    return struct.pack("<H", value)

def write_uint8(value: int) -> bytes:
    return struct.pack("<B", value)

def write_float_le(value: float) -> bytes:
    return struct.pack("<f", value)
