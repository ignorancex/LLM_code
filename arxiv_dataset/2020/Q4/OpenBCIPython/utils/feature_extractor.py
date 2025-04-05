
def _get_frame(audio, index, frame):
    if index < 0:
        return None
    return audio.raw[(index * frame):(index + 1) * frame]

def _get_frame_array(audio, index, frame):
    if index < 0:
        return None
    return audio.data[(index * frame):(index + 1) * frame]