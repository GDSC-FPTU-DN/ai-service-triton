'''
Triton Server Build Support.
----
Author: Ming-doan
Created: 2024-02-20
----
This module provides support for building Triton Server model repository
and its configuration files.
'''

import argparse
import shutil
from _abstract import TritonConfig
from _file_config import FileConfig
from _build_pbtxt import BuildProtoBufTxt
from _utils import get_absolute_path
from _constants import ERROR_PREFIX, WARNING_PREFIX


# Define argument parser
parser = argparse.ArgumentParser(
    description='Triton Server Build Model Module.')


def main():
    '''
    Main function for Triton Server Build Model Module.
    '''
    # Add arguments -----------------------------------------------------------
    # Model repository path. Eg: /path/to/model/repository
    parser.add_argument('--model-repository', type=str,
                        help='Path to the model repository')

    # Model model path. Eg: /path/to/model/model.onnx
    parser.add_argument('--model-path', type=str,
                        help='Path to the model model')

    # Model name. Name for triton services. Eg: model
    parser.add_argument('--model-name', type=str,
                        help='Name of the model to be built')

    # Model version. Eg: 1
    parser.add_argument('--model-version', type=int,
                        help='Version of the model to be built')

    # Maximum batch size of Triton server. Eg: 0 (default: 0, no batching)
    parser.add_argument('--max-batch-size', type=int,
                        help='Maximum batch size of Triton server')

    # Enable dynamic batching of Triton server. Eg: False (default: False)
    parser.add_argument('--dynamic-batching', type=bool,
                        help='Enable dynamic batching of Triton server')

    # Model configuration file path. Eg: /path/to/model/config.yaml
    parser.add_argument('-f', type=str,
                        help='Path to the model configuration file. If provided, other arguments will be ignored.')

    # Rebuild model repository. Eg: False (default: False)
    parser.add_argument('--rebuild', type=bool, default=False,
                        help='Rebuild model repository. If provided, model repository will be rebuilt.')

    # Parse arguments --------------------------------------------------------
    args = parser.parse_args()

    # Load configuration file if provided ------------------------------------
    if args.f:
        # Show warning if other arguments are provided.
        if args.model_repository or args.model_name or args.model_path or args.model_version or args.max_batch_size or args.dynamic_batching:
            print(WARNING_PREFIX +
                  "Configuration file provided. Other arguments will be ignored.")

        # Load configuration file
        config = FileConfig(args.f).get_config()

    # Use provided arguments -------------------------------------------------
    else:
        # Default configuration
        args.model_version = args.model_version or 1
        args.max_batch_size = args.max_batch_size or 0
        args.dynamic_batching = args.dynamic_batching or False

        # Check for missing arguments
        if not args.model_repository:
            print(
                ERROR_PREFIX + "Model repository path is required. Use --model-repository to specify.")
            return
        if not args.model_path:
            print(ERROR_PREFIX + "Model path is required. Use --model-path to specify.")
            return
        if not args.model_name:
            print(ERROR_PREFIX + "Model name is required. Use --model-name to specify.")
            return

        # Initialize configuration
        config: TritonConfig = {
            "model_repository": args.model_repository,
            "models": {}
        }

        # Add models configuration
        config["models"][args.model_name] = {
            "engine": "onnx",
            "dtype": "float32",
            "max_batch_size": args.max_batch_size,
            "dynamic_batching": args.dynamic_batching,
            "versions": [
                {
                    "version": args.model_version,
                    "path": args.model_path
                }
            ]
        }

    # Rebuild model repository if provided ------------------------------------
    if args.rebuild:
        shutil.rmtree(get_absolute_path(config["model_repository"]))

    # Write to model repository
    BuildProtoBufTxt(config).build()
    # try:
    # except Exception as e:
    #     print(ERROR_PREFIX + str(e))


# Run main function if module is run directly
if __name__ == '__main__':
    main()
