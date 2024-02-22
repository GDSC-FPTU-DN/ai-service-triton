'''
Triton Server Build Support.
----
Author: Ming-doan
Created: 2024-02-20
----
This module provides support for building Triton Server model repository
and its configuration files.
'''

# Chroma logger prefix
ERROR_PREFIX = "\033[91mERROR: \033[0m"
SUCCESS_PREFIX = "\033[92mSUCCESS: \033[0m"
WARNING_PREFIX = "\033[93mWARNING: \033[0m"
INFO_PREFIX = "\033[94mINFO: \033[0m"


# Constants
TRITON_PRESEVED_KEYWORDS = [
    "model", "config", "triton_python_backend_utils", "pb_utils", "TritonPythonModel"]
BUILD_DIR = "build"
