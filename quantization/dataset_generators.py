import os
import cv2
import numpy as np
from loguru import logger
from utils import get_annotations
from tqdm import tqdm

class CostomCOCODaset():
    def __init__(
        self, 
        image_folder, 
        annotations, 
        input_size,
        max_size=500, 
        transpose_to_chw=True,
        input_channels=3,
        img_type = "rgb",
        ):

        self._idx = 0 

        assert self._idx < max_size <= len(annotations), \
            f"Choose max_size between {self._idx} and {len(annotations)}"
        self.annotations = annotations 

        img_type = img_type.lower()        
        assert img_type in ["rgb", "bayer"], \
            f"The images you passed is not supported, please choose from ('rgb', 'bayer')"
        self.img_type = img_type

        if img_type == "bayer" and input_channels == 3:
            logger.info("If you are resizing the image via DEMOSAICING, make sure \
                that input_size is 2 as large as desired input for the model !!!")
 
        self.max_idx = max_size 
        self.data_dir = image_folder
        self.input_size = input_size
        self.transpose_to_chw = transpose_to_chw
        self.input_channels = input_channels

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):

        if self._idx >= self.max_idx:
            raise StopIteration()

        filename = self.annotations[self._idx]["file_name"]

        if self.img_type == "bayer":
            filename = filename.replace("jpg", "png")
        img_file = os.path.join(self.data_dir, filename)

        # read image as BGR
        image = cv2.imread(
            img_file, 
            cv2.IMREAD_UNCHANGED
        )

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        if self.img_type == "bayer" and self.input_channels == 3: 
            image = self.preproc_bayer(
                img = image,
                input_size = self.input_size,
                input_channels = self.input_channels,
            )
        else:
            image = self.preproc(
                image, 
                self.input_size, 
                input_channels = self.input_channels
            )

        image = self.slicing(image)

        if not self.transpose_to_chw:
            image = image.transpose(1, 2, 0)

        self._idx += 1
        return image[None] 
    
    def __len__(self):
        return self.max_idx

    @staticmethod
    def preproc_bayer(img, input_size, input_channels, swap=(2, 0, 1)):
        #BGR image
        h, w, c = img.shape 
        # get correct size 
        if h > 2 * input_size[0]: 
            img = img[:2 * input_size[0], :, :]
        if w > 2 * input_size[1]: 
            img = img[:, :2 * input_size[1], :] 

        # reszie by demosaicing
        img = img.astype(np.uint16)
        output = np.zeros((img.shape[0] // 2, img.shape[1] // 2, input_channels), dtype=np.int16) 
        output[:, :, 0] =  img[1::2,  ::2, 0]
        output[:, :, 1] = (img[ ::2,  ::2, 0] + img[1::2, 1::2, 0]) / 2 
        output[:, :, 2] =  img[ ::2, 1::2, 0]
        output = output.astype(np.uint8)
        output = output.transpose(swap)
        return output 

    @staticmethod
    def preproc(img, input_size, swap=(2, 0, 1), input_channels=3):
        if len(img.shape) == 3:
            padded_img = np.ones(
                (input_size[0], input_size[1], input_channels),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        if len(resized_img.shape) == 2:
            resized_img = resized_img[..., None]

        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img 

    @staticmethod  
    def slicing(image):
        patch_top_left = image[..., ::2, ::2]
        patch_top_right = image[..., ::2, 1::2]
        patch_bot_left = image[..., 1::2, ::2]
        patch_bot_right = image[..., 1::2, 1::2]
        image = np.concatenate(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            axis=0,
        )
        return image

class GvsocInputGeneratorCOCO(CostomCOCODaset):
    def __init__(
        self, 
        image_folder, 
        annotations, 
        gvsoc_inputs_folder,
        input_size,
        model_type,
        input_channels=3,
        ):

        assert model_type.lower() in ["bayer", "rgb"], \
            "model_type must be 'bayer' or 'rgb'"
        self.model_type = model_type.lower()

        if model_type.lower() == "rgb" and input_channels != 3:
            logger.warning("Input_channels must be 3 for rgb model. Changing to 3")
            input_channels = 3

        logger.info("Image type is {}".format(self.model_type))
        logger.info("Input channels is {}".format(input_channels))

        self.gvsoc_inputs_folder = gvsoc_inputs_folder
        if not os.path.exists(self.gvsoc_inputs_folder):
            os.makedirs(self.gvsoc_inputs_folder)

        annotations = get_annotations(annotations)
        super().__init__(
            image_folder, 
            annotations, 
            input_size,
            max_size = len(annotations), 
            input_channels = input_channels,
        )

    def __next__(self):

        if self._idx >= self.max_idx:
            raise StopIteration()

        filename = self.annotations[self._idx]["file_name"]

        if self.model_type == "bayer":
            filename = filename.replace("jpg", "png")

        img_file = os.path.join(self.data_dir, filename)
        image = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        if self.model_type == "bayer" and self.input_channels == 3:
            image = self.preproc_bayer(
                image, 
                self.input_size, 
                input_channels=self.input_channels
            )
        else: 
            image = self.preproc(
                image, 
                self.input_size, 
                input_channels=self.input_channels
            )
        image = image.transpose(1, 2, 0)

        self._idx += 1
        return image, filename.split(".")[0]
    
    def generate_and_save(self):

        for image, filename in tqdm(self):
            save_path = os.path.join(
                self.gvsoc_inputs_folder,
                filename + (".ppm" if self.input_channels == 3 else ".pgm")
            ) 
            cv2.imwrite(save_path, image)
