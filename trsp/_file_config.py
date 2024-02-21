'''
Triton Server Build Support.
----
Author: Ming-doan
Created: 2024-02-20
----
This module provides support for building Triton Server model repository
and its configuration files.
'''

import yaml
from typing import Any


class FileConfig:
    '''
    File Configuration Class.
    Used to load and validate configuration file.
    '''

    def __init__(self, file_path: str):
        self.__file_path = file_path
        self.__config = self.__load_config()

    def __load_config(self) -> dict:
        '''
        Load configuration file and validate its fields.
        '''
        # Load configuration file ------------------------------------------------
        with open(self.__file_path, "r") as f:
            configs = yaml.safe_load(f)

        # Validate configuration file ---------------------------------------------
        # Check if fields are present
        assert "model_repository" in configs, "Model repository not found in configuration file."
        assert "models" in configs, "Models not found in configuration file."

        # Check if models field is valid
        for model in configs["models"]:
            assert "engine" in configs["models"][model], "Model `engine` not found in configuration models."
            assert "max_batch_size" in configs["models"][model], "Model `max_batch_size` not found in configuration models."
            assert "versions" in configs["models"][model], "Model `versions` not found in configuration models."

            # Check if versions field is valid
            for version in configs["models"][model]["versions"]:
                assert "version" in version, "Model `version` not found in configuration versions."

                # If engine is onnx, check if onnx field is valid
                if configs["models"][model]["engine"] == "onnx":
                    assert "path" in version, "Model `path` not found in configuration versions."

            # If max_batch_size is present, check if it is valid
            if "max_batch_size" in configs["models"][model]:
                assert isinstance(configs["models"][model]["max_batch_size"],
                                  int), "Model `max_batch_size` should be an integer."

            # If dynamic_batching is present, check if it is valid
            if "dynamic_batching" in configs["models"][model]:
                assert isinstance(configs["models"][model]["dynamic_batching"],
                                  bool), "Model `dynamic_batching` should be a boolean."

                if "dynamic_batching_delay" in configs["models"][model]:
                    assert isinstance(configs["models"][model]["dynamic_batching_delay"],
                                      int), "Model `dynamic_batching_delay` should be an integer."

            # If instance_group is present, check if it is valid
            if "instance_group" in configs["models"][model]:
                for group in configs["models"][model]["instance_group"]:
                    assert "kind" in group, "Model `kind` not found in configuration instance_group."

                    if group["kind"] == "gpu":
                        if "count" in group:
                            assert isinstance(
                                group["count"], int), "Model `count` should be an integer."

                    if "gpus" in group:
                        assert isinstance(
                            group["gpus"], list), "Model `gpus` should be a list."

        # All fields are valid ---------------------------------------------------
        return configs

    def get_config(self) -> dict[str, Any]:
        '''
        Get configuration dictionary.
        '''
        return self.__config
