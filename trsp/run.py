import os
import argparse
import subprocess
from _constants import BUILD_DIR, ERROR_PREFIX


parser = argparse.ArgumentParser(
    description='Triton Server Running Module.')


def run():
    build_directory = os.path.join(os.getcwd(), BUILD_DIR)
    # Add arguments -----------------------------------------------------------
    # Port to be used by docker container. Eg: 7860
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to be used by docker container')

    # Parse arguments --------------------------------------------------------
    args = parser.parse_args()

    # Looking for Dockerfile in build directory
    if not os.path.exists(os.path.join(build_directory, 'Dockerfile')):
        print(ERROR_PREFIX + 'Dockerfile not found in build directory')
        return

    # Build docker image
    try:
        subprocess.run(
            ['docker', 'build', '-t', 'triton-server', build_directory])
    except Exception as e:
        print(ERROR_PREFIX + 'Failed to build docker image. ' + str(e))
        return

    # Run docker container
    try:
        subprocess.run(['docker', 'run', '-d', '-p',
                       f'{args.port}:{args.port}', 'triton-server'])
    except Exception as e:
        print(ERROR_PREFIX + 'Failed to run docker container. ' + str(e))
        return


# Run function
if __name__ == '__main__':
    run()
