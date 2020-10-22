import skvideo.io


def video_to_numpy(video_file_path: str):
    video_array = skvideo.io.vread(video_file_path)
    return video_array
