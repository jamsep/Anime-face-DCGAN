import os
import requests
import asyncio
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from pygelbooru import Gelbooru
from os.path import splitext
import numpy as np

import cv2
from typing import Tuple

load_dotenv()

API_KEY = os.environ.get("GELBOORU_API_KEY")
USER_ID = os.environ.get("GELBOORU_USER_ID")

# Create an instance of the Gelbooru class
gelbooru = Gelbooru(API_KEY, USER_ID)

async def get_posts():
    # Search for images with the specified tags
    posts = await gelbooru.search_posts(tags=['looking_at_viewer', 'solo', 'close-up'], exclude_tags=['pussy', 'ass', 'absurdres', 'eye_focus', 'animated', 'nipples', 'stepped_on', 'pantyshot', 'japanese_(nationality)', 'photo_(medium)'], limit=20000)
    return posts

# Create a directory to store the images
if not os.path.exists('gelbooru_images'):
    os.makedirs('gelbooru_images')


# credit to: https://gist.github.com/IdeaKing/11cf5e146d23c5bb219ba3508cca89ec
def resize_with_pad(image: np.array, 
                    new_shape: Tuple[int, int], 
                    padding_color: Tuple[int] = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

# save images to directory 
for post in asyncio.run(get_posts()):
    response = requests.get(post.file_url)
    print(response)
    print(post.filename)
    img = Image.open(BytesIO(response.content))
    img = resize_with_pad(np.asarray(img), [256, 256]) # turns img to an np.array
    img = Image.fromarray(img) # converts back np.array to img
    img = img.save(f"gelbooru_images/{post.filename}")

