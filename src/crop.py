import os

import numpy as np
from PIL import Image

def crop_images_in_folder(folder_path, new_folder):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.png')):
                file_path = os.path.join(root, file)
                image = Image.open(file_path)

                image_values = np.array(image, dtype=np.int32)
                # bg_color = np.array([255, 255, 255])
                bg_color = image_values[0, 0]
                tolerance = 5

                # Find non-transparent bounding box
                grid = np.meshgrid(np.arange(image.size[0]), np.arange(image.size[1]))

                non_zero = np.sum(np.abs(image_values - bg_color), axis=2) > tolerance
                if not np.any(non_zero):
                    continue
                non_zero_x = grid[0][non_zero]
                non_zero_y = grid[1][non_zero]

                min_x = np.min(non_zero_x)
                max_x = np.max(non_zero_x + 1)
                min_y = np.min(non_zero_y)
                max_y = np.max(non_zero_y + 1)

                bbox = (min_x, min_y, max_x, max_y)

                # Crop the image to its content
                cropped_image = image.crop(bbox)

                scale = 0.5
                cropped_image = cropped_image.resize((int(cropped_image.size[0] * scale), int(cropped_image.size[1] * scale)), Image.Resampling.LANCZOS)

                # Save the cropped image back to the same file
                new_root = root.replace(folder_path, new_folder)

                os.makedirs(new_root, exist_ok=True)
                cropped_image.save(os.path.join(new_root, file))

                print(f"{new_root}{file} cropped successfully.")


# crop_images_in_folder("../images/comp", "../images/cropped")
crop_images_in_folder("images/", "images_cropped/")
