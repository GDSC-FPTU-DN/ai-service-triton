import os
from typing import Union, Any


class TritonEnum:
    '''
    Triton Enum Class.
    Used to present special types in Triton Server config.pbtxt.
    '''

    def __init__(self, value: Union[str, list], mapping_values: dict = None):
        self.__value = value
        self.__mapping_values = mapping_values

    def __get_value(self, value: Any) -> str:
        '''
        Get value from mapping values.
        '''
        if self.__mapping_values:
            return self.__mapping_values.get(value, value)
        return value

    def __str__(self):
        '''
        Convert to string.
        '''
        if isinstance(self.__value, list):
            return f"[{', '.join(self.__get_value(str(value)) for value in self.__value)}]"
        return self.__value


def get_absolute_path(path: str) -> str:
    '''
    Get absolute path from relative path.
    '''
    paths = path.split("/")
    # Check if path is absolute
    if paths[0] == "":
        return path
    # Check if path is relative
    if paths[0] == ".":
        return os.path.join(os.getcwd(), *paths[1:])
    # Check if path is relative to home directory
    if paths[0] == "~":
        return os.path.join(os.path.expanduser("~"), *paths[1:])
    # Check if path is relative to current directory
    if paths[0] == "..":
        back_count = paths.count("..")
        current_directory = os.getcwd()
        for _ in range(back_count):
            current_directory = current_directory[0:current_directory.rfind(
                "\\")]
        return os.path.join(current_directory, *paths[back_count:])
    return os.path.join(os.getcwd(), path)


def dictionary_to_string(dictionary: dict, indent: int = 0, tab: int = 2) -> str:
    '''
    Convert dictionary to pretty string.
    '''
    string = ""
    for key, value in dictionary.items():
        if isinstance(value, str):
            string += f"{' '*indent}{key}: \"{value}\"\n"
        if isinstance(value, int):
            string += f"{' '*indent}{key}: {value}\n"
        if isinstance(value, TritonEnum):
            string += f"{' '*indent}{key}: {value}\n"
        if isinstance(value, list):
            string += f"{' '*indent}{key} [\n"
            for i, item in enumerate(value):
                if isinstance(item, str):
                    string += f"{' '*(indent+tab)}\"{item}\"\n"
                if isinstance(item, int):
                    string += f"{' '*(indent+tab)}{item}\n"
                if isinstance(item, TritonEnum):
                    string += f"{' '*(indent+tab)}{item}\n"
                if isinstance(item, dict):
                    string += f"{' '*(indent+tab)}{{\n"
                    string += dictionary_to_string(item, indent+tab*2, tab)
                    string += f"{' '*(indent+tab)}}}{',' if i != len(value) - 1 else ''}\n"
            string += f"{' '*indent}]\n"
        if isinstance(value, dict):
            string += f"{' '*indent}{key} {{\n"
            string += dictionary_to_string(value, indent+tab, tab)
            string += f"{' '*indent}}}\n"
    return string


def get_backend_string(engine: str) -> str:
    '''
    Get backend string for Triton Server config.pbtxt file.
    '''
    if engine == "onnx":
        return "onnxruntime"
    return ""


def get_kind_instance(kind: str) -> str:
    '''
    Get kind instance string for Triton Server config.pbtxt file.
    '''
    if kind == "gpu":
        return TritonEnum("KIND_GPU")
    return TritonEnum("KIND_CPU")


def get_dtype_string(dtype: str) -> str:
    '''
    Get data type string for Triton Server config.pbtxt file.
    '''
    if dtype == "float32":
        return TritonEnum("TYPE_FP32")
    if dtype == "float64":
        return TritonEnum("TYPE_FP64")
    if dtype == "int32":
        return TritonEnum("TYPE_INT32")
    if dtype == "int64":
        return TritonEnum("TYPE_INT64")
    if dtype == "uint8":
        return TritonEnum("TYPE_UINT8")
    if dtype == "uint16":
        return TritonEnum("TYPE_UINT16")
    if dtype == "uint32":
        return TritonEnum("TYPE_UINT32")
    if dtype == "uint64":
        return TritonEnum("TYPE_UINT64")
    if dtype == "int8":
        return TritonEnum("TYPE_INT8")
    if dtype == "int16":
        return TritonEnum("TYPE_INT16")
    if dtype == "bool":
        return TritonEnum("TYPE_BOOL")
    raise ValueError(f"Unsupported data type: {dtype}")
