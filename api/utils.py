from ast import literal_eval
import numpy as np

def parseImgStr(image_str) -> np.ndarray:
    data_list = literal_eval(image_str)
    image_array = np.array(data_list, dtype=np.float32) / 255.0
        
    if image_array.shape == (28, 29):
        image_array = image_array[:, :28]
    elif image_array.shape != (28, 28):
        if image_array.size == 784:
            image_array = image_array.reshape(28, 28)
        else:
            raise ValueError(f"Invaild shape: {image_array.shape}, expected (28, 28).")
        
    return image_array