# Chroma logger prefix
ERROR_PREFIX = "\033[91mERROR: \033[0m"
SUCCESS_PREFIX = "\033[92mSUCCESS: \033[0m"
WARNING_PREFIX = "\033[93mWARNING: \033[0m"
INFO_PREFIX = "\033[94mINFO: \033[0m"


def get_file_instruction_string(data: dict): return f'''# Welcome to the Triton Server Protobuf Text Format (PBtxt) file.
# This is auto generated file by `trsp` module. Developed by Ming-doan.
# Models: {data['name']}.
# Engine: {data['backend']}.
# ------------------------------

'''
