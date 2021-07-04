import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from pyquaternion import Quaternion
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import math


def get_cropped_view(img, bbox):
    """Crop image"""
    return img[bbox[1]:bbox[3], bbox[0]: bbox[2]]


def get_frame_difference(frame1, frame2):
    """Compute normalized RMSE"""
    return np.linalg.norm(frame1 - frame2) / max(
        np.mean(frame1.flatten()), 10)


def extract_video_signal(video_file, bbox=(284, 143, 1085, 667),):
    """Extract video frame differences and corresponding time stamps"""

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise ValueError('Could not open video file')
    fps = cap.get(cv2.CAP_PROP_FPS)

    video_diff_signal = []
    video_ts_array = [-1/fps/2]
    prev_frame = cap.read()[1]
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_frame = get_cropped_view(prev_frame, bbox)

    while cap.isOpened():

        ret, frame = cap.read()
        if frame is None:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = get_cropped_view(frame, bbox)
        frame_diff = get_frame_difference(prev_frame, frame)
        video_diff_signal.append(frame_diff)
        prev_frame = frame.copy()
        video_ts_array.append(video_ts_array[-1] + 1/fps)

    del video_ts_array[0]
    video_ts_array = np.array(video_ts_array)
    video_diff_signal = np.array(video_diff_signal)
    return video_diff_signal, video_ts_array


def extract_motion_signal(motion_file):
    """Extract motion signals: Rotation angles and translation distances"""

    prev_ts = None
    prev_position = None
    prev_orientation = None
    motion_ts_array = []
    rotation_signal = []
    translation_signal = []
    with open(motion_file, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:

            ts = int(line[1]) / 1000  # timestamp in seconds
            orientation = [line[8], *line[5:8]]
            orientation = [float(c) for c in orientation]
            orientation = Quaternion(orientation).unit
            position = np.array([float(c) for c in line[2:5]])

            if prev_ts is not None:

                motion_ts_array.append((ts + prev_ts) / 2)

                rotation = (prev_orientation.conjugate * orientation).angle
                rotation_signal.append(np.array(rotation))

                translation = np.linalg.norm(prev_position - position)
                translation_signal.append(translation)

            prev_position = position
            prev_orientation = orientation
            prev_ts = ts

    motion_ts_array = np.array(motion_ts_array)
    rotation_signal = np.array(rotation_signal)
    translation_signal = np.array(translation_signal)

    return motion_ts_array, rotation_signal, translation_signal


def synchronize(
        video_diff_signal, video_ts_array, motion_signal, motion_ts_array,
        first_frame_ts_range):

    # Prepare all shifts to be tested
    time_shifts = np.arange(
        first_frame_ts_range[0], first_frame_ts_range[1] + 0.0001, 1/6)

    r_array_masked = []
    for time_shift in time_shifts:

        # Transform the motion signal timestamps with the current shift
        motion_ts_array_corrected = motion_ts_array - time_shift

        # Get the start and end times of the sensor signal
        max_signal_start_time = motion_ts_array_corrected[0]
        min_signal_end_time = motion_ts_array_corrected[-1]

        # Check that the sensor signal overlaps with the frame signal
        if max_signal_start_time > video_ts_array[-1] or \
                min_signal_end_time < video_ts_array[0]:
            r_array_masked.append(math.nan)
            continue

        # Crop the overlapping section of the frame signal
        video_diff_signal_l_idx = np.argwhere(
            video_ts_array >= max_signal_start_time)[0][0]
        video_diff_signal_r_idx = np.argwhere(
            video_ts_array <= min_signal_end_time)[-1][0]
        this_video_diff_signal = \
            video_diff_signal[
            video_diff_signal_l_idx: video_diff_signal_r_idx + 1]
        this_video_ts_array = \
            video_ts_array[video_diff_signal_l_idx: video_diff_signal_r_idx + 1]

        # Discard the zero-values in the frame signal
        video_diff_signal_mask = this_video_diff_signal > 0

        # Ensure a minimum duration of the remaining non-zero,
        # overlapping frame signal
        if np.sum(video_diff_signal_mask) / 30 < 5:
            r_array_masked.append(math.nan)
            continue

        # Interpolate the sensor signal with the transformed timestamps
        # at the frame timestamps
        motion_signal_corrected = np.interp(
            this_video_ts_array, motion_ts_array_corrected,
            motion_signal)

        # Compute the Spearman correlation coefficient
        R_masked = scipy.stats.spearmanr(
            this_video_diff_signal[video_diff_signal_mask],
            motion_signal_corrected[video_diff_signal_mask])
        r_array_masked.append(R_masked.correlation)

    # Choose the shift with the largest correlation coefficient
    r_array_masked = np.array(r_array_masked)
    r_max = r_array_masked.max()
    best_shift = time_shifts[np.argmax(r_array_masked)]
    return best_shift, r_max, time_shifts, r_array_masked


def main(video_file, motion_file, save_steps=True, load_steps=True,
         to_plot=False, first_frame_ts_range=(-10, 30)):

    # Try to load the pre-computed video difference signal
    video_diff_signal_file = video_file.parent /\
        (video_file.stem + "_video_signal.npy")
    video_ts_array_file = video_file.parent /\
        (video_file.stem + "_video_ts_array.npy")
    if load_steps and video_diff_signal_file.exists()\
            and video_ts_array_file.exists():
        video_diff_signal = np.load(video_diff_signal_file)
        video_ts_array = np.load(video_ts_array_file)

    else:
        # Compute the video difference signal
        video_diff_signal, video_ts_array = extract_video_signal(video_file)
        if save_steps:
            np.save(video_diff_signal_file, video_diff_signal)
            np.save(video_ts_array_file, video_ts_array)

    # Apply median filter to remove zero-difference artifacts and noise
    video_diff_signal = scipy.signal.medfilt(video_diff_signal)

    # Extract the motion signals
    motion_ts_array, rotation_signal, translation_signal =\
        extract_motion_signal(motion_file)

    # Filter, scale and add the rotation and translation signals
    rotation_signal = scipy.signal.medfilt(rotation_signal)
    translation_signal = scipy.signal.medfilt(translation_signal)
    sync_motion_signal = rotation_signal / rotation_signal.std() +\
        translation_signal / translation_signal.std()

    # Compute the best shift (timestamp of the first video frame in the scanner
    # clock reference
    best_shift, r_max, time_shifts, r_array_masked = synchronize(
        video_diff_signal, video_ts_array, sync_motion_signal, motion_ts_array,
        first_frame_ts_range)

    print(f"First video frame is at ScanTrainer time stamp {best_shift}.")
    print(f"The R value is {r_max}.")

    if to_plot:
        plt.figure()
        plt.plot(time_shifts, r_array_masked)
        plt.show()

    return best_shift, r_max


if __name__ == "__main__":
    video_file = Path('WIN_20201110_20_42_25_Pro.mp4').resolve()
    motion_file = Path('AuditTrail_201110_204428_0l2lbt.csv').resolve()
    main(video_file, motion_file, save_steps=False, load_steps=False, to_plot=True)
