from datetime import timedelta


def convert_framenumber2timestamp(frame_number, fps):
    return timedelta(seconds=frame_number / fps)