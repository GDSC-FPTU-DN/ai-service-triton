"""
Simulate the Triton Python Backend API for developing purpose.
---
Actual `triton_python_backend_utils` module by NVIDIA:
https://github.com/triton-inference-server/python_backend/blob/main/src/resources/triton_python_backend_utils.py

---
⚠️ In development, this module import `pb_utils.TritonModel` which is not
available in Triton server. **REMEMBER** to remove this import statement before
deploying to Triton server.
"""

from abc import abstractmethod
from numpy.typing import DTypeLike, ArrayLike, NDArray


# Classes ---------------------------------------------------------------------


class ModelConfig:
    """An object of ModelConfig class is used to describe
    the model configuration for autocomplete.
    Parameters
    ----------
    model_config : ModelConfig Object
        Object containing the model configuration. Only the max_batch_size, inputs
        and outputs properties can be modified for auto-complete model configuration.
    """

    @abstractmethod
    def __init__(self, model_config): ...

    @abstractmethod
    def as_dict(self) -> dict:
        """Provide the read-only access to the model configuration
        Returns
        -------
        dict
            dictionary type of the model configuration contained in
            the ModelConfig object
        """

    @abstractmethod
    def set_max_batch_size(self, max_batch_size: int):
        """Set the max batch size for the model.
        Parameters
        ----------
        max_batch_size : int
            The max_batch_size to be set.
        Raises
        ------
        ValueError
            If configuration has specified max_batch_size non-zero value which
            is larger than the max_batch_size to be set for the model.
        """

    @abstractmethod
    def set_dynamic_batching(self):
        """Set dynamic_batching as the scheduler for the model if no scheduler
        is set. If dynamic_batching is set in the model configuration, then no
        action is taken and return success.
        Raises
        ------
        ValueError
            If the 'sequence_batching' or 'ensemble_scheduling' scheduler is
            set for this model configuration.
        """

    @abstractmethod
    def add_input(self, input: dict):
        """Add the input for the model.
        Parameters
        ----------
        input : dict
            The input to be added.
        Raises
        ------
        ValueError
            If input contains property other than 'name', 'data_type',
            'dims', 'optional' or any of the non-optional properties
            are not set, or if an input with the same name already exists
            in the configuration but has different data_type or dims property
        """

    @abstractmethod
    def add_output(self, output: dict):
        """Add the output for the model.
        Parameters
        ----------
        output : dict
            The output to be added.
        Raises
        ------
        ValueError
            If output contains property other than 'name', 'data_type'
            and 'dims' or any of the properties are not set, or if an
            output with the same name already exists in the configuration
            but has different data_type or dims property
        """

    @abstractmethod
    def set_model_transaction_policy(self, transaction_policy_dict: dict):
        """
        Set model transaction policy for the model.
        Parameters
        ----------
        transaction_policy_dict : dict
            The dict, containing all properties to be set as a part
            of `model_transaction_policy` field.
        Raises
        ------
        ValueError
            If transaction_policy_dict contains property other
            than 'decoupled', or if `model_transaction_policy` already exists
            in the configuration, but has different `decoupled` property.
        """


class Tensor:
    """Tensor class represents a tensor in Triton. It contains the name of
    the tensor and the data in the tensor. The data is represented as a
    numpy array. The numpy array can be of any shape and any data type.
    """

    @abstractmethod
    def __init__(self, name: str, data: ArrayLike) -> None: ...

    @abstractmethod
    def as_numpy(self) -> NDArray:
        """Get the data in the tensor as a numpy array.
        """

    @abstractmethod
    def name(self) -> str:
        """Get the name of the tensor.
        """


class TritonError:
    """TritonError class represents an error in Triton. It contains the error
    message and the error code. The error code is represented as an integer.
    """
    UNKNOWN = ...
    INTERNAL = ...
    NOT_FOUND = ...
    INVALID_ARG = ...
    UNAVAILABLE = ...
    UNSUPPORTED = ...
    ALREADY_EXISTS = ...
    CANCELLED = ...

    @abstractmethod
    def __init__(self, error_message: str, error_code: int = ...) -> None: ...

    @abstractmethod
    def message(self) -> str:
        """Get the error message.
        """


class TritonModelException(Exception):
    """TritonModelException class represents an exception in Triton. It
    contains the error message and the error code. The error code is
    represented as an integer.
    """

    @abstractmethod
    def __init__(self, error_message: str, error_code: int = ...) -> None: ...


class InferenceRequest:
    """InferenceRequest class represents an inference request in Triton. It
    contains the input tensors for the request. The input tensors are
    represented as a list of Tensor objects.
    """

    @abstractmethod
    def inputs(self) -> list[Tensor]:
        """Get the input tensors for the request.
        """

    @abstractmethod
    def set_release_flags(self, release_flags: list[bool]) -> None:
        """Set the release flags for the input tensors. If the release flag
        for an input tensor is set to True, Triton will release the input
        tensor after the request is executed. If the release flag for an
        input tensor is set to False, Triton will not release the input
        tensor after the request is executed. The release flags are
        represented as a list of boolean values. The length of the release
        flags list must be the same as the length of the input tensors list.
        """

    @abstractmethod
    def exec(self) -> 'InferenceResponse':
        """Execute the request and get the response.
        """


class InferenceResponse:
    """InferenceResponse class represents an inference response in Triton. It
    contains the output tensors for the response. The output tensors are
    represented as a list of Tensor objects.
    """

    @abstractmethod
    def __init__(
        self, output_tensors: list[Tensor], error: TritonError = ...) -> None: ...

    @abstractmethod
    def output_tensors(self) -> list[Tensor]:
        """Get the output tensors for the response.
        """

    @abstractmethod
    def error(self) -> TritonError:
        """Get the error for the response.
        """


class TritonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.

    ⚠️ Remove this abstract class when deploying the model to Triton Server.
    """

    @abstractmethod
    def auto_complete_config(auto_complete_config: 'ModelConfig'):
        """`auto_complete_config` is called only once when loading the model
        assuming the server was not started with
        `--disable-auto-complete-config`. Implementing this function is
        optional. No implementation of `auto_complete_config` will do nothing.
        This function can be used to set `max_batch_size`, `input` and `output`
        properties of the model using `set_max_batch_size`, `add_input`, and
        `add_output`. These properties will allow Triton to load the model with
        minimal model configuration in absence of a configuration file. This
        function returns the `pb_utils.ModelConfig` object with these
        properties. You can use the `as_dict` function to gain read-only access
        to the `pb_utils.ModelConfig` object. The `pb_utils.ModelConfig` object
        being returned from here will be used as the final configuration for
        the model.

        Note: The Python interpreter used to invoke this function will be
        destroyed upon returning from this function and as a result none of the
        objects created here will be available in the `initialize`, `execute`,
        or `finalize` functions.

        Parameters
        ----------
        auto_complete_model_config : pb_utils.ModelConfig
          An object containing the existing model configuration. You can build
          upon the configuration given by this object when setting the
          properties for this model.

        Returns
        -------
        pb_utils.ModelConfig
          An object containing the auto-completed model configuration
        """

    @abstractmethod
    def initialize(self, args: dict) -> None:
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

    @abstractmethod
    def execute(self, requests: list[InferenceRequest]
                ) -> list[InferenceResponse]:
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

    @abstractmethod
    def finalize(self) -> None:
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """


# Functions -------------------------------------------------------------------


def serialize_byte_tensor(input_tensor: Tensor) -> bytes:
    """
    Serializes a bytes tensor into a flat numpy array of length prepended
    bytes. The numpy array should use dtype of np.object_. For np.bytes_,
    numpy will remove trailing zeros at the end of byte sequence and because
    of this it should be avoided.
    Parameters
    ----------
    input_tensor : np.array
        The bytes tensor to serialize.
    Returns
    -------
    serialized_bytes_tensor : np.array
        The 1-D numpy array of type uint8 containing the serialized bytes in 'C' order.
    Raises
    ------
    InferenceServerException
        If unable to serialize the given tensor.
    """


def deserialize_bytes_tensor(encoded_tensor: bytes) -> Tensor:
    """
    Deserializes an encoded bytes tensor into an
    numpy array of dtype of python objects
    Parameters
    ----------
    encoded_tensor : bytes
        The encoded bytes tensor where each element
        has its length in first 4 bytes followed by
        the content
    Returns
    -------
    string_tensor : np.array
        The 1-D numpy array of type object containing the
        deserialized bytes in 'C' order.
    """


def get_input_tensor_by_name(
        inference_request: InferenceRequest, name: str) -> Tensor:
    """Find an input Tensor in the inference_request that has the given
    name
    Parameters
    ----------
    inference_request : InferenceRequest
        InferenceRequest object
    name : str
        name of the input Tensor object
    Returns
    -------
    Tensor
        The input Tensor with the specified name, or None if no
        input Tensor with this name exists
    """


def get_output_tensor_by_name(inference_response: InferenceResponse,
                              name: str) -> Tensor:
    """Find an output Tensor in the inference_response that has the given
    name
    Parameters
    ----------
    inference_response : InferenceResponse
        InferenceResponse object
    name : str
        name of the output Tensor object
    Returns
    -------
    Tensor
        The output Tensor with the specified name, or None if no
        output Tensor with this name exists
    """


def get_input_config_by_name(model_config: dict, name: str) -> dict:
    """Get input properties corresponding to the input
    with given `name`
    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration
    name : str
        name of the input object
    Returns
    -------
    dict
        A dictionary containing all the properties for a given input
        name, or None if no input with this name exists
    """


def get_output_config_by_name(model_config: dict, name: str) -> dict:
    """Get output properties corresponding to the output
    with given `name`
    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration
    name : str
        name of the output object
    Returns
    -------
    dict
        A dictionary containing all the properties for a given output
        name, or None if no output with this name exists
    """


def using_decoupled_model_transaction_policy(model_config: dict) -> bool:
    """Whether or not the model is configured with decoupled
    transaction policy.
    Parameters
    ----------
    model_config : dict
        dictionary object containing the model configuration

    Returns
    -------
    bool
        True if the model is configured with decoupled transaction
        policy.
    """


def triton_to_numpy_type(data_type: str) -> DTypeLike:
    """Converts Triton data type to numpy data type.
    """


def numpy_to_triton_type(data_type: DTypeLike) -> str:
    """Converts numpy data type to Triton data type.
    """


def triton_string_to_numpy(triton_type_string: str) -> DTypeLike:
    """Converts Triton data type string to numpy data type.
    """
