from eeg_dataloader_utils import (
    TUH_INCLUDED_CHANNELS,
    FREQUENCY,
    resample_data,
    get_edf_signals,
    get_ordered_channels,
)
from tqdm import tqdm
import argparse
import numpy as np
import os
import pyedflib
import h5py


def resample_all(raw_edf_dir, save_dir):
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    failed_files = []
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]

        save_fn = os.path.join(save_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
        if os.path.exists(save_fn):
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)

            orderedChannels = get_ordered_channels(
                edf_fn, False, f.getSignalLabels(), TUH_INCLUDED_CHANNELS
            )
            signals = get_edf_signals(f)
            signal_array = np.array(signals[orderedChannels, :])
            sample_freq = f.getSampleFrequency(0)
            if sample_freq != FREQUENCY:
                signal_array = resample_data(
                    signal_array,
                    to_freq=FREQUENCY,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resampled_signal", data=signal_array)
            hf.create_dataset("resample_freq", data=FREQUENCY)

        except BaseException:
            failed_files.append(edf_fn)

    print("DONE. {} files failed.".format(len(failed_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resample.")
    parser.add_argument(
        "--raw_edf_dir",
        type=str,
        default=None,
        help="Full path to raw edf files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Full path to dir to save resampled signals.",
    )
    args = parser.parse_args()

    resample_all(args.raw_edf_dir, args.save_dir)
