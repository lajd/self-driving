from typing import List, Union, Optional, Callable
from moviepy.editor import VideoFileClip
import ffmpeg
import skvideo.io

from tools.image_utils import annotate_lanes_on_image_with_canny_and_hough


def video_to_numpy(video_file_path: str):
    video_array = skvideo.io.vread(video_file_path)
    return video_array


def get_video_attributes(input_mp4_filepath: str, stream_index: Optional[int] = None):

    def get_info(info, stream_index_):
        return {
            'width': int(info['streams'][stream_index_]['width']),
            'height': int(info['streams'][stream_index_]['height']),
            'duration': float(info['streams'][stream_index_]['duration']),
            'num_frames': int(info['streams'][stream_index_]['nb_frames']),
            'fps': eval(info['streams'][stream_index_]['avg_frame_rate'])
        }

    stream_info = ffmpeg.probe(input_mp4_filepath)

    if stream_index:
        return get_info(stream_info, stream_index)
    else:
        e = ValueError("No valid streams")
        for stream_index, _ in enumerate(stream_info):
            try:
                return get_info(stream_info, stream_index)
            except KeyError as e:
                continue
        raise e


def annotate_lanes_on_video_with_canny_and_hough(video_file_path: str, image_annotation_function: Callable = annotate_lanes_on_image_with_canny_and_hough):
    clip1 = VideoFileClip(video_file_path)
    annotated_video = clip1.fl_image(image_annotation_function)
    return annotated_video
