import argparse
import unittest

# Import test modules
from rembg_test import RembgModuleTest


parser = argparse.ArgumentParser(description='Triton Server Deployment Test')


def main():
    # Add arguments
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host name of Triton server')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port number of Triton server')

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments to test modules
    suite = unittest.TestSuite()
    suite.addTest(RembgModuleTest('test_remove_background',
                  host=args.host, port=args.port))

    # Run tests
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    main()
