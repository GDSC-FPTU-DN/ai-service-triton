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
            url=self.url)

    def test_remove_background(self):
        # Get image from url
        image_url = "https://img.freepik.com/photos-gratuite/bouchent-portrait-beau-jeune-homme-confiant-t-shirt-blanc-nature-exterieure-floue_176420-6301.jpg?size=626&ext=jpg&ga=GA1.1.1788068356.1706486400&semt=ais"
        pil_image = Image.open(BytesIO(requests.get(image_url).content))

        # Resize image
        resize_pil_image = pil_image.resize((320, 320), Image.LANCZOS)

        # Convert to numpy
        np_image = np.array(resize_pil_image)

        # Expand dims for batch_size axis
        np_image = np.expand_dims(np_image, 0).astype(np.uint8)

        # Inputs data
        inputs = [
            httpclient.InferInput(
                'rembg_input_1', np_image.shape, "UINT8"),
        ]
        inputs[0].set_data_from_numpy(np_image)

        # Request to triton server
        query_response = self.__triton_client.infer(
            model_name="rembg",
            inputs=inputs
        )
        # Get response image
        response_mask = query_response.as_numpy('rembg_output_1')
        response_img = Image.fromarray(response_mask).resize(pil_image.size)

        # Cutout image
        empty_img = Image.new("RGBA", (pil_image.size), 0)
        cutout_img = Image.composite(pil_image, empty_img, response_img)

        # Save result image
        cutout_img.save("rembg_test_result.png")

        # Implement test
        self.assertEqual(
            response_mask.shape,
            (320, 320),
            "Response image shape is not correct. Expected: (320, 320), Got: " + str(
                response_mask.shape)
        )
