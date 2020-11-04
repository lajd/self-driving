from typing import List
import matplotlib.pyplot as plt
import numpy as np

def chunk_list(list_elements, chunk_size):
    """Chunks a list into chunk_size chunks, with last chunk the remaining
    elements.

    :param list_elements:
    :param chunk_size:
    :return:
    """
    chunks = []
    while True:
        if len(list_elements) > chunk_size:
            chunk = list_elements[0:chunk_size]
            list_elements = list_elements[chunk_size:]
            chunks.append(chunk)
        else:
            chunks.append(list_elements)
            break
    return chunks


def show_images_in_columns(images: List[np.ndarray], titles: List[str]=None, cmaps: List[str]=None, figsize=(25, 15)):
    num_cols = len(images)
    fig, axes = plt.subplots(ncols=num_cols, figsize=figsize)

    if not cmaps:
        cmaps = [None] * num_cols

    if not titles:
        titles = [None] * num_cols
    assert len(images) == len(titles) == len(cmaps)
    for i, image in enumerate(images):
        axes[i].imshow(image, cmap=cmaps[i])
        axes[i].set_title(titles[i])
        axes[i].axis('off')
