import torch
from scipy.signal import resample
import tqdm
from torch.utils.data import Dataset, DataLoader
from vilmedic.datasets.base.TextDataset import TextDataset
import glob

from vilmedic.datasets.base.utils import load_file

import numpy as np
import os
import pyedflib

FREQUENCY = 200
TUH_INCLUDED_CHANNELS = [
    "EEG FP1",
    "EEG FP2",
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
    "EEG FZ",
    "EEG CZ",
    "EEG PZ",
]


def get_edf_signals(edf):
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals


def get_ordered_channels(
        file_name,
        labels_object,
        channel_names,
):
    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]
    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except ValueError:
            # Channel not in edf, ignore signal
            print("file {} miss one/several channels, skipping.".format(file_name))
            return None
    return ordered_channels


def load_session(eeg_path, clip_len, stride, max_num_clips, time_step_size, do_seizure=False):
    # Get file of session
    session_files = glob.glob(os.path.join(eeg_path, "*.npy"))
    session_array = [np.load(npy) for npy in session_files]

    # Convert entire EEG session into clips of length clip_len
    session_clips = []
    for signal_array in session_array:
        physical_clip_len = int(FREQUENCY * clip_len)
        num_clips = (signal_array.shape[-1] - clip_len * FREQUENCY) // (
                stride * FREQUENCY
        ) + 1
        for clip_idx in range(num_clips):
            start_window = clip_idx * FREQUENCY * stride
            end_window = np.minimum(signal_array.shape[-1], start_window + physical_clip_len)
            curr_slc = signal_array[:, start_window:end_window]
            physical_time_step_size = int(FREQUENCY * time_step_size)

            start_time_step = 0
            time_steps = []
            while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
                end_time_step = start_time_step + physical_time_step_size
                curr_time_step = curr_slc[:, start_time_step:end_time_step]
                time_steps.append(curr_time_step)
                start_time_step = end_time_step

            eeg_clip = np.stack(time_steps, axis=0)
            session_clips.append(eeg_clip)

    session_clips = np.stack(session_clips)

    # Padding or truncating to max_num_clips
    session_num_clips = session_clips.shape[0]
    if session_num_clips == max_num_clips:
        pass
    elif session_num_clips < max_num_clips:
        session_clips = np.pad(session_clips,
                               pad_width=[(0, max_num_clips - session_num_clips), (0, 0), (0, 0), (0, 0)],
                               mode='constant',
                               constant_values=0)
    else:
        session_clips = session_clips[:max_num_clips]

    assert session_clips.shape[0] == max_num_clips
    return torch.from_numpy(np.stack(session_clips, axis=0)).type(torch.FloatTensor)


def process_files(eeg_paths):
    # For each path (session)
    for i, eeg_folder in enumerate(tqdm.tqdm(eeg_paths)):
        session_files = glob.glob(os.path.join(eeg_folder, "*.edf"))
        # for each edf in the session
        for edf in session_files:

            npy_filename = edf.replace(".edf", ".npy")

            # If not pre-processed, do it
            if not os.path.exists(npy_filename):
                f = pyedflib.EdfReader(edf)
                signals = get_edf_signals(f)
                ordered_channels = get_ordered_channels(
                    edf, f.getSignalLabels(), TUH_INCLUDED_CHANNELS
                )

                # One or several channels are missing
                if ordered_channels is None:
                    continue

                signal_array = np.array(signals[ordered_channels, :])
                sample_freq = f.getSampleFrequency(0)
                f.close()

                # Sampling
                if sample_freq != FREQUENCY:
                    num = int(FREQUENCY * int(signal_array.shape[1] / sample_freq))
                    signal_array = resample(signal_array, num=num, axis=1)

                # Save
                np.save(npy_filename, signal_array)


def read_file(root, image_path, split, file):
    lines = load_file(os.path.join(root, split + '.' + file))
    return [os.path.join(image_path, line) for line in lines]


class TUHDataset(Dataset):
    """
    Pytorch dataset that loads both EEG clips and accompanying text report
    """

    def __init__(
            self,
            seq,
            eeg,
            split,
            ckpt_dir,
            **kwargs
    ):
        # Seq
        self.seq = TextDataset(**seq, split=split, ckpt_dir=ckpt_dir)
        # For decoding, if needed
        self.tokenizer = self.seq.tokenizer
        self.tokenizer_max_len = self.seq.tokenizer_max_len

        # EEG
        self.root = eeg.root
        self.file = eeg.file
        self.eeg_path = eeg.eeg_path
        self.clip_len = eeg.clip_len
        self.split = split
        self.ckpt = ckpt_dir

        if self.file is not None:
            self.eeg_paths = read_file(self.root, self.eeg_path, split, self.file)

        process_files(self.eeg_paths)

    def __len__(self):
        assert len(self.eeg_paths) == len(self.seq)
        return len(self.eeg_paths)

    def __getitem__(self, index):

        return {'image': load_session(self.eeg_paths[index],
                                      clip_len=self.clip_len,
                                      stride=self.clip_len,
                                      max_num_clips=30,
                                      time_step_size=1),
                **self.seq.__getitem__(index)}

    def get_collate_fn(self):
        def collate_fn(batch):
            new_batch = []
            new_masks = []
            for sample in batch:
                sample_images = sample['image']
                sample_mask = (sample_images.sum(dim=(1, 2, 3)) != 0)
                new_batch.append(sample_images)
                new_masks.append(sample_mask)

            collated = {'images': torch.stack(new_batch),
                        'images_mask': torch.stack(new_masks)}

            collated = {**self.seq.get_collate_fn()(batch), **collated}
            return collated

        return collate_fn
