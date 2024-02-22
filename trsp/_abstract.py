'''
Triton Server Build Support.
----
Author: Ming-doan
Created: 2024-02-20
----
This module provides support for building Triton Server model repository
and its configuration files.
'''

from typing import TypedDict, Dict, List, Optional, Union, Any


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


class TensorShapeConfig(TypedDict):
    '''
    {
        "dims": List[int],
        "dtype": Union[str, TritonEnum]
    }
    '''
    dims: List[int]
    dtype: Union[str, TritonEnum]


class TensorConfig(TypedDict):
    '''
    {
        "input": List[TensorShapeConfig],
        "output": List[TensorShapeConfig]
    }
    '''
    input: List[TensorShapeConfig]
    output: List[TensorShapeConfig]


class PythonModuleConfig(TypedDict):
    '''
    {
        "path": str,
        "execute": str,
        "initialize": Optional[str],
        "finalize": Optional[str]
    }
    '''
    path: str
    execute: str
    initialize: Optional[str]
    finalize: Optional[str]


class VersionConfig(TypedDict):
    '''
    {
        "version": int,
        "path": str,
        "module": str,
        "execute": str,
        "initialize": str,
        "finalize": str
    }
    '''
    version: int
    path: Optional[str]
    module: Optional[PythonModuleConfig]


class InstanceGroupConfig(TypedDict):
    '''
    {
        "kind": str,
        "count": int,
        "gpus": List[int]
    }
    '''
    kind: str
    count: Optional[int]
    gpus: Optional[List[int]]


class EnsembleStepConfig(TypedDict):
    '''
    {
        "model_name": str,
        "model_version": int,
        "input_map": Dict[str, str],
        "output_map": Dict[str, str]
    }
    '''
    model: str
    version: Union[int, str]


class ModelConfig(TypedDict):
    '''
    {
        "engine": str,
        "max_batch_size": int,
        "versions": List[VersionConfig],
        "dynamic_batching": bool,
        "max_queue_delay_microseconds": int,
        "instance_group": InstanceGroupConfig,
        "requirements": List[str],
        "tensor": TensorConfig,
        "steps": List[EnsembleStepConfig]
    }
    '''
    engine: str
    max_batch_size: int
    versions: List[VersionConfig]
    dynamic_batching: Optional[bool]
    dtype: Optional[str]
    max_queue_delay_microseconds: Optional[int]
    instance_group: Optional[List[InstanceGroupConfig]]
    requirements: Optional[List[str]]
    tensor: Optional[TensorConfig]
    steps: Optional[List[EnsembleStepConfig]]


class TritonConfig(TypedDict):
    '''
    {
        "model_repository": str,
        "models": {
            [model_name]: ModelConfig
        }
    }
    '''
    model_repository: str
    models: Dict[str, ModelConfig]


class FormatedTensors(TypedDict):
    '''
    {
        "name": str,
        "data_type": str,
        "dims": Union[List[int], TritonEnum]
    }
    '''
    name: str
    data_type: str
    dims: Union[List[int], TritonEnum]


class FormatedInputOutputTensors(TypedDict):
    '''
    {
        "input": List[FormatedTensors],
        "output": List[FormatedTensors]
    }
    '''
    input: List[FormatedTensors]
    output: List[FormatedTensors]


class FormatedTritonConfig(TypedDict):
    '''
    {
        "name": str,
        "backend": str,
        "max_batch_size": int,
        "input": List[FormatedTensors],
        "output": List[FormatedTensors],
        "dynamic_batching": Dict,
        "instance_group": Dict
    }
    '''
    name: str
    backend: str
    max_batch_size: int
    input: List[FormatedTensors]
    output: List[FormatedTensors]
    dynamic_batching: Dict
    instance_group: Dict


class EnsembleSchedulingInputOutputMap(TypedDict):
    '''
    {
        "input": str,
        "output": str
    }
    '''
    key: str
    value: str


class EnsembleSchedulingStep(TypedDict):
    '''
    {
        "model_name": str,
        "model_version": int,
        "input_map": EnsembleSchedulingInputOutputMap,
        "output_map": EnsembleSchedulingInputOutputMap
    }
    '''
    model_name: str
    model_version: int
    input_map: EnsembleSchedulingInputOutputMap
    output_map: EnsembleSchedulingInputOutputMap


class EnsembleSchedulingConfig(TypedDict):
    '''
    {
        "step": List[EnsembleSchedulingStep]
    }
    '''
    step: List[EnsembleSchedulingStep]
