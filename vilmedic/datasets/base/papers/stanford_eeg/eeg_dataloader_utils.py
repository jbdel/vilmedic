import numpy as np
import eeghdf
import os
from vilmedic.constants import DATA_DIR

STANFORD_INCLUDED_CHANNELS = [
    "EEG Fp1",
    "EEG Fp2",
    "EEG F3",
    "EEG F4",
    "EEG C3",
    "EEG C4",
    "EEG P3",
    "EEG P4",
    "EEG O1",
    "EEG O2",
    "EEG F7",
    "EEG F8",
    "EEG T3",
    "EEG T4",
    "EEG T5",
    "EEG T6",
    "EEG Fz",
    "EEG Cz",
    "EEG Pz",
]

FREQUENCY = 200

SEIZURE_STRINGS = ["sz", "seizure", "absence", "spasm"]
FILTER_SZ_STRINGS = ["@sz", "@seizure"]


def compute_stanford_file_tuples(stanford_dataset_dir, lpch_dataset_dir, splits):
    """
    Given the splits, processes file tuples from filemarkers
    file tuple: (eeg filename, location of sz or -1 if no sz, split)
    """

    file_tuples = []
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    for split in splits:
        for hospital in ["lpch", "stanford"]:
            data_dir = (
                stanford_dataset_dir if hospital == "stanford" else lpch_dataset_dir
            )
            for sz_type in ["non_sz", "sz"]:
                filemarker_dir = (
                    f"{DATA_DIR}/EEG/file_markers/"
                    + f"file_markers_{hospital}/{sz_type}_{split}.txt"
                )
                filemarker_contents = open(filemarker_dir, "r").readlines()
                for fm in filemarker_contents:
                    fm_tuple = fm.strip("\n").split(",")
                    filepath = os.path.join(data_dir, fm_tuple[0])
                    fm_tuple = (filepath, float(fm_tuple[1]), split)
                    file_tuples.append(fm_tuple)

    return file_tuples


def get_stanford_sz_times(eegf):
    df = eegf.edf_annotations_df
    seizure_df = df[df.text.str.contains("|".join(SEIZURE_STRINGS), case=False)]
    return seizure_df["starts_sec"].tolist()


def is_increasing(channel_indices):
    """
    Check if a list of indices is sorted in ascending order.
    If not, we will have to convert it to a numpy array before slicing,
    which is a rather expensive operation
    Returns: bool
    """
    last = channel_indices[0]
    for i in range(1, len(channel_indices)):
        if channel_indices[i] < last:
            return False
        last = channel_indices[i]
    return True


def get_ordered_channels(
    file_name,
    labels_object,
    channel_names=STANFORD_INCLUDED_CHANNELS,
    verbose=False,
):
    """
    Reads channel names and returns consistent ordering
    Args:
        file_name (str): name of edf file
        labels_object: extracted from edf signal using f.getSignalLabels()
        channel_names (List(str)): list of channel names
        verbose (bool): whether to be verbose
    Returns:
        list of channel indices in ordered form
    """

    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except IndexError:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def stanford_eeg_loader(filepath, sz_start_idx, split, clip_len):
    """
    given filepath and sz_start, extracts EEG clip of length 60 sec

    """

    # load EEG signal
    eegf = eeghdf.Eeghdf(filepath)
    ordered_channels = get_ordered_channels(
        filepath, eegf.electrode_labels, channel_names=STANFORD_INCLUDED_CHANNELS
    )
    phys_signals = eegf.phys_signals

    # get seizure time

    if sz_start_idx == -1 or split != "train":
        sz_start = sz_start_idx
    else:
        sz_times = get_stanford_sz_times(eegf)
        sz_start = sz_times[int(sz_start_idx)]

    # extract clip
    if sz_start == -1:
        max_start = max(phys_signals.shape[1] - FREQUENCY * clip_len, 0)
        sz_start = int(max_start / 2)
        sz_start /= FREQUENCY

    start_time = int(FREQUENCY * max(0, sz_start))
    end_time = start_time + int(FREQUENCY * clip_len)

    if not is_increasing(ordered_channels):
        eeg_slice = phys_signals[:, start_time:end_time]
        eeg_slice = eeg_slice[ordered_channels, :]
    else:
        eeg_slice = (
            phys_signals.s2u[ordered_channels]
            * phys_signals.data[ordered_channels, start_time:end_time].T
        ).T

    diff = FREQUENCY * clip_len - eeg_slice.shape[1]
    # padding zeros
    if diff > 0:
        zeros = np.zeros((eeg_slice.shape[0], diff))
        eeg_slice = np.concatenate((eeg_slice, zeros), axis=1)
    eeg_slice = eeg_slice.T

    return eeg_slice
