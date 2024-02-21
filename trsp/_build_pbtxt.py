'''
Triton Server Build Support.
----
Author: Ming-doan
Created: 2024-02-20
----
This module provides support for building Triton Server model repository
and its configuration files.
'''

import os
import onnx
from _utils import (
    get_absolute_path,
    dictionary_to_string,
    get_backend_string,
    get_dtype_string,
    get_kind_instance,
    TritonEnum
)
from _constants import INFO_PREFIX, SUCCESS_PREFIX, get_file_instruction_string


class BuildProtoBufTxt:
    '''
    Build ProtoBufTxt Class.
    Use to build Triton Server model repository and its configuration files.
    '''

    def __init__(self, data: dict):
        self.__data = data
        self.__file_name = "config"
        self.__model_repository = get_absolute_path(
            self.__data["model_repository"])

    def __create_folders(self, name: str, model_config: dict) -> str:
        '''
        Create model folders and subfolders for each version.
        '''
        # Create model directory
        model_path = os.path.join(
            self.__model_repository, name)
        os.makedirs(model_path, exist_ok=True)

        # Create version directories
        for version in model_config["versions"]:
            version_path = os.path.join(
                model_path, str(version["version"]))
            os.makedirs(version_path, exist_ok=True)

        # Return model path
        return model_path

    def __format_config(self, name: str, model_config: dict) -> dict:
        '''
        Format from yaml format to Triton Server config.pbtxt format.
        '''
        # Initialize config
        config = {
            "name": name,
            "backend": get_backend_string(model_config["engine"]),
            "max_batch_size": model_config["max_batch_size"],
            "input": [],
            "output": []
        }

        # Add dynamic batching if enabled
        if "dynamic_batching" in model_config:
            config["dynamic_batching"] = {}
            if "max_queue_delay_microseconds" in model_config:
                config["dynamic_batching"]["max_queue_delay_microseconds"] = model_config["max_queue_delay_microseconds"]

        # Add instance_group if enabled
        if "instance_group" in model_config:
            config["instance_group"] = []
            for group in model_config["instance_group"]:
                instance_group = {
                    "kind": get_kind_instance(group["kind"])
                }
                if group["kind"] == "gpu":
                    if "count" in group:
                        instance_group["count"] = group["count"]
                if "gpus" in group:
                    instance_group["gpus"] = TritonEnum(group["gpus"])
                config["instance_group"].append(instance_group)

        # Add input and output
        config["input"] = model_config["_input"]
        config["output"] = model_config["_output"]

        return config

    def __format_onnx(self, path: str, model_config: dict) -> dict[str, list]:
        '''
        Process ONNX model and generate input and output configs.
        '''
        def __get_onnx_shape(input_layer) -> list:
            '''
            Get ONNX input shape.
            If dynamic batching is enabled, remove batch dimension.
            '''
            # Get tensor shape
            tensor_shape = [
                dim.dim_value for dim in input_layer.type.tensor_type.shape.dim]

            # Remove batch dimension if dynamic batching is enabled
            if "dynamic_batching" in model_config:
                return tensor_shape[1:]

            return tensor_shape

        # Initialize configs ---------------------------------------------------
        configs = {
            "input": [],
            "output": []
        }

        # Process each version -------------------------------------------------
        for version in model_config["versions"]:
            # Load ONNX model
            model_path = get_absolute_path(version["path"])
            onnx_model = onnx.load(model_path)

            # Modify input shape and output if dynamic batching is enabled
            if "dynamic_batching" in model_config:
                # Modify input dimensions param of ONNX model
                for i, input_layer in enumerate(onnx_model.graph.input):
                    input_layer.type.tensor_type.shape.dim[
                        0].dim_param = f"{input_layer.name}_dynamic_axes_{i+1}"

                # Modify output dimensions param of ONNX model
                for i, output_layer in enumerate(onnx_model.graph.output):
                    output_layer.type.tensor_type.shape.dim[
                        0].dim_param = f"{output_layer.name}_dynamic_axes_{i+1}"

            # Get model data type
            model_data_type = model_config["dtype"] if "dtype" in model_config else "float32"

            # Define mapping values for TritonEnum
            # If input or output of the model is a string, replace 0 with -1
            mapping_values = {
                "0": "-1"
            }

            # Add input configs
            for input_layer in onnx_model.graph.input:
                input_config = {
                    "name": input_layer.name,
                    "data_type": get_dtype_string(model_data_type),
                    "dims": TritonEnum(__get_onnx_shape(input_layer), mapping_values=mapping_values)
                }
                configs["input"].append(input_config)
            # Add output configs
            for output_layer in onnx_model.graph.output:
                output_config = {
                    "name": output_layer.name,
                    "data_type": get_dtype_string(model_data_type),
                    "dims": TritonEnum(__get_onnx_shape(output_layer), mapping_values=mapping_values)
                }
                configs["output"].append(output_config)

            # Save ONNX model
            model_save_path = os.path.join(
                path, str(version["version"]), "model.onnx"
            )
            onnx.save(onnx_model, model_save_path)

        # Return input and output configs
        return configs

    def __generate_pbtxt_string(self, data: dict) -> str:
        '''
        Generate config.pbtxt data to string.
        '''
        dict_string = dictionary_to_string(data)
        return get_file_instruction_string(data) + dict_string

    def __write_pbtxt(self, path: str, file_string: str):
        '''
        Write config.pbtxt file to model directory.
        '''
        file_path = os.path.join(path, f"{self.__file_name}.pbtxt")
        with open(file_path, "w") as f:
            f.write(file_string)

    def build(self):
        '''
        Build model repository and its configuration files.
        '''
        # Print info -----------------------------------------------------------
        print(INFO_PREFIX + "Building model repository...")

        # Create model_repository directory if not exists
        os.makedirs(self.__model_repository, exist_ok=True)

        # Process each model ---------------------------------------------------
        for name, model_config in self.__data["models"].items():
            # Create model directory
            model_path = self.__create_folders(name, model_config)

            # Create ONNX model file, if engine is onnx
            if model_config["engine"] == "onnx":
                input_output_configs = self.__format_onnx(
                    model_path, model_config)

            # Add input and output configs to model config
            model_config["_input"] = input_output_configs["input"]
            model_config["_output"] = input_output_configs["output"]

            # Create main config data
            config = self.__format_config(name, model_config)

            # Generate and write config.pbtxt file
            proto_string = self.__generate_pbtxt_string(config)
            self.__write_pbtxt(model_path, proto_string)

        # Print success --------------------------------------------------------
        print(SUCCESS_PREFIX +
              f"Build completed. Model repository: {self.__model_repository}")
