import numpy as np
import torch


def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    """
    Finds the note timings based on the onsets and frames information

    Parameters
    ----------
    onsets: torch.FloatTensor, shape = [frames, bins]
    frames: torch.FloatTensor, shape = [frames, bins]
    velocity: torch.FloatTensor, shape = [frames, bins]
    onset_threshold: float
    frame_threshold: float

    Returns
    -------
    pitches: np.ndarray of bin_indices
    intervals: np.ndarray of rows containing (onset_index, offset_index)
    velocities: np.ndarray of velocity values
    """
    onsets = (onsets > onset_threshold).cpu()
    frames = (frames > frame_threshold).cpu()
    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    pitches = []
    intervals = []
    velocities = []

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        offset = frame
        velocity_samples = []

        while onsets[offset, pitch].item() or frames[offset, pitch].item():
            if onsets[offset, pitch].item():
                velocity_samples.append(velocity[offset, pitch].item())
            offset += 1
            if offset == onsets.shape[0]:
                break

        if offset > onset:
            pitches.append(pitch)
            intervals.append([onset, offset])
            velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)

def extract_notes_time(onset_times, offset_times, frames, velocity, frame_threshold=0.25):
    frames = (frames > frame_threshold).cpu()
    onset_diff = torch.cat([frames[:1, :], frames[1:, :] - frames[:-1, :]], dim=0) == 1
    offset_diff = torch.cat([frames[:1, :], -(frames[1:, :] - frames[:-1, :])], dim=0) == 1

    frames_onset = []
    pitches_onset = []
    for nonzero_onset in onset_diff.nonzero():
        frames_onset.append(nonzero_onset[0].item())
        pitches_onset.append(nonzero_onset[1].item())

    frames_offset = {}
    for nonzero_offset in offset_diff.nonzero():
        frame_offset = nonzero_offset[0].item()
        pitch = nonzero_offset[1].item()
        pitch_offset_frame_list = frames_offset.get(pitch, [])
        pitch_offset_frame_list.append(frame_offset)
        frames_offset[pitch] = pitch_offset_frame_list

    pitches = []
    intervals = []
    velocities = []
    for pitch, frame_onset in zip(pitches_onset, frames_onset):
        pitch_offset_frame_list = frames_offset.get(pitch, [])
        if len(pitch_offset_frame_list) == 0:
            frame_offset = 625-1
        else:
            frame_offset = pitch_offset_frame_list.pop(0)
            frames_offset[pitch] = pitch_offset_frame_list
        if (offset_times[frame_offset][pitch] > onset_times[frame_onset][pitch]):
            pitches.append(pitch)
            intervals.append([onset_times[frame_onset][pitch], offset_times[frame_offset][pitch]])
            velocities.append(velocity[frame_onset, pitch])

    return np.array(pitches), np.array(intervals), np.array(velocities)


def notes_to_frames(pitches, intervals, shape):
    """
    Takes lists specifying notes sequences and return

    Parameters
    ----------
    pitches: list of pitch bin indices
    intervals: list of [onset, offset] ranges of bin indices
    shape: the shape of the original piano roll, [n_frames, n_bins]

    Returns
    -------
    time: np.ndarray containing the frame indices
    freqs: list of np.ndarray, each containing the frequency bin indices
    """
    roll = np.zeros(tuple(shape))
    for pitch, (onset_time, offset_time) in zip(pitches, intervals):
        onset = min(625-1, int(onset_time * 31.25))
        offset = min(625-1, int(offset_time * 31.25))
        roll[onset:offset, pitch] = 1

    time = np.arange(roll.shape[0])
    freqs = [roll[t, :].nonzero()[0] for t in time]
    return time, freqs
