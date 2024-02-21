import unittest
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import tritonclient.http as httpclient


class RembgModuleTest(unittest.TestCase):
    def __init__(self, method_name: str, **kwargs):
        super(RembgModuleTest, self).__init__(method_name)
        self.url = kwargs['host'] + ":" + str(kwargs['port'])
        self.__triton_client = httpclient.InferenceServerClient(
            url=self.url, verbose=True)

    def test_remove_background(self):
        # Get image from url
        image_url = "https://img.freepik.com/photos-gratuite/bouchent-portrait-beau-jeune-homme-confiant-t-shirt-blanc-nature-exterieure-floue_176420-6301.jpg?size=626&ext=jpg&ga=GA1.1.1788068356.1706486400&semt=ais"
        test_image = np.array(Image.open(
            BytesIO(requests.get(image_url).content)))
        # Inputs data
        inputs = [
            httpclient.InferInput('input.1', test_image.shape, "FP32"),
        ]
        # Outputs data
        outputs = [
            httpclient.InferRequestedOutput('1959', "FP32"),
        ]
        # Request to triton server
        query_response = self.__triton_client.infer(
            model_name="rembg",
            inputs=inputs,
            outputs=outputs
        )
        # Get response image
        response_image = query_response.as_numpy('output_image')
        # Implement test
        self.assertEqual(response_image.shape, (test_image.shape[0], test_image.shape[1], 4),
                         "Not implemented yet")
