import os
import librosa
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import scipy
from spk_embed.dataset import WavDataset
from collections import defaultdict
import h5py
from torch.nn.utils.rnn import pad_sequence

FFT_SIZE = 512
SGRAM_DIM = FFT_SIZE // 2 + 1


class BCVCCDataset(Dataset):
    def __init__(self, original_metadata, data_dir, spk_embed_dir, idtable_path=None, split="train",
                 padding_mode="zero_padding", use_mean_listener=False, load_spk_model_data=False):
        self.data_dir = data_dir
        self.spk_embed_dir = spk_embed_dir
        self.split = split
        self.padding_mode = padding_mode
        self.use_mean_listener = use_mean_listener
        self.load_spk_model_data = load_spk_model_data

        # cache features
        self.features = {}

        # add mean listener to metadata
        if use_mean_listener:
            mean_listener_metadata = self.gen_mean_listener_metadata(original_metadata)
            metadata = original_metadata + mean_listener_metadata
        else:
            metadata = original_metadata

        # get judge id table and number of judges
        if idtable_path is not None:
            if os.path.isfile(idtable_path):
                self.idtable = torch.load(idtable_path)
            elif self.split == "train":
                self.gen_idtable(metadata, idtable_path)
            self.num_judges = len(self.idtable)

        self.metadata = []
        self.wavscp = []
        if self.split == "train":
            for wav_name, judge_name, avg_score, score in metadata:
                self.metadata.append([wav_name, avg_score, score, self.idtable[judge_name]])
                self.wavscp.append([wav_name, self._path_from_wav_name(wav_name)])
        else:
            for sys_name, wav_name, avg_score in metadata:  # (sys_name, wav_name, avg_score)
                self.metadata.append([sys_name, wav_name, avg_score])
                self.wavscp.append([wav_name, self._path_from_wav_name(wav_name)])

            # build system list
            self.systems = list(set([item[0] for item in metadata]))

        if self.load_spk_model_data:
            self.spk_model_dataset = WavDataset(self.wavscp, fs=16000)

    def _path_from_wav_name(self, wav_name: str):
        return os.path.join(self.data_dir, "wav", wav_name)

    def __getitem__(self, idx):
        if self.split == "train":
            wav_name, avg_score, score, judge_id = self.metadata[idx]
        else:
            sys_name, wav_name, avg_score = self.metadata[idx]

        # cache features
        if wav_name in self.features:
            mag_sgram = self.features[wav_name]
        else:
            h5_path = os.path.join(self.data_dir, "bin", wav_name + ".h5")
            if os.path.isfile(h5_path):
                data_file = h5py.File(h5_path, 'r')
                mag_sgram = np.array(data_file['mag_sgram'][:])
                timestep = mag_sgram.shape[0]
                mag_sgram = np.reshape(mag_sgram, (timestep, SGRAM_DIM))
            else:
                wav, _ = librosa.load(self._path_from_wav_name(wav_name), sr=16000)
                mag_sgram = np.abs(
                    librosa.stft(wav, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)).astype(
                    np.float32).T
            self.features[wav_name] = mag_sgram

        # speaker embedding
        if self.load_spk_model_data:
            # TODO: randomly select the audio length between 2s to 4s
            spk_embed, _ = self.spk_model_dataset[idx, 3]
        else:
            spk_embed = np.load(os.path.join(self.spk_embed_dir, wav_name.replace('.wav', '.npy')))
            spk_embed = torch.from_numpy(spk_embed)

        if self.split == "train":
            return mag_sgram, spk_embed, avg_score, score, judge_id
        else:
            return mag_sgram, spk_embed, avg_score, sys_name, wav_name

    def __len__(self):
        return len(self.metadata)

    def gen_mean_listener_metadata(self, original_metadata):
        assert self.split == "train"
        mean_listener_metadata = []
        wav_names = set()
        for wav_name, _, avg_score, _ in original_metadata:
            if wav_name not in wav_names:
                mean_listener_metadata.append([wav_name, "mean_listener", avg_score, avg_score])
                wav_names.add(wav_name)
        return mean_listener_metadata

    def gen_idtable(self, metadata, idtable_path):
        self.idtable = {}
        count = 0
        for _, judge_name, _, _ in metadata:
            # mean listener always takes the last id
            if judge_name not in self.idtable and not judge_name == "mean_listener":
                self.idtable[judge_name] = count
                count += 1
        if self.use_mean_listener:
            self.idtable["mean_listener"] = count
            count += 1
        torch.save(self.idtable, idtable_path)

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[0].shape[0])
        bs = len(sorted_batch)  # batch_size
        avg_scores = torch.FloatTensor([sorted_batch[i][2] for i in range(bs)])
        mag_sgrams = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        mag_sgrams_lengths = torch.from_numpy(np.array([mag_sgram.size(0) for mag_sgram in mag_sgrams]))

        spk_embeds = torch.stack([sorted_batch[i][1] for i in range(bs)], dim=0)

        if self.padding_mode == "zero_padding":
            mag_sgrams_padded = pad_sequence(mag_sgrams, batch_first=True)
        elif self.padding_mode == "repetitive":
            max_len = mag_sgrams_lengths[0]
            mag_sgrams_padded = []
            for mag_sgram in mag_sgrams:
                this_len = mag_sgram.shape[0]
                dup_times = max_len // this_len
                remain = max_len - this_len * dup_times
                to_dup = [mag_sgram for t in range(dup_times)]
                to_dup.append(mag_sgram[:remain, :])
                mag_sgrams_padded.append(torch.Tensor(np.concatenate(to_dup, axis=0)))
            mag_sgrams_padded = torch.stack(mag_sgrams_padded, dim=0)
        else:
            raise NotImplementedError

        if not self.split == "train":
            sys_names = [sorted_batch[i][3] for i in range(bs)]
            wav_names = [sorted_batch[i][4] for i in range(bs)]
            return mag_sgrams_padded, spk_embeds, avg_scores, sys_names, wav_names
        else:
            scores = torch.FloatTensor([sorted_batch[i][3] for i in range(bs)])
            judge_ids = torch.LongTensor([sorted_batch[i][4] for i in range(bs)])
            return mag_sgrams_padded, mag_sgrams_lengths, spk_embeds, avg_scores, scores, judge_ids


def get_dataset(
        dataset_name, data_dir, spk_embed_dir, split, idtable_path=None, padding_mode="zero_padding",
        use_mean_listener=False, load_spk_model_data=False
):
    if dataset_name in ["BVCC", "OOD"]:
        names = {"train": "TRAINSET", "valid": "DEVSET", "test": "TESTSET"}

        metadata = defaultdict(dict)
        metadata_with_avg = list()

        # read metadata
        with open(os.path.join(data_dir, "sets", names[split]), "r") as f:
            lines = f.read().splitlines()

            # line has format <system, wav_name, score, _, judge_name>
            for line in lines:
                parts = line.split(",")
                sys_name = parts[0]
                wav_name = parts[1]
                score = int(parts[2])
                judge_name = parts[4]
                metadata[sys_name + "|" + wav_name][judge_name] = score

        # calculate average score
        for _id, v in metadata.items():
            sys_name, wav_name = _id.split("|")
            avg_score = np.mean(np.array(list(v.values())))
            if split == "train":
                for judge_name, score in v.items():
                    metadata_with_avg.append([wav_name, judge_name, avg_score, score])
            else:
                # in testing mode, additionally return system name and only average score
                metadata_with_avg.append([sys_name, wav_name, avg_score])

        return BCVCCDataset(
            metadata_with_avg, data_dir, spk_embed_dir, idtable_path, split, padding_mode, use_mean_listener,
            load_spk_model_data,
        )
    else:
        raise NotImplementedError


def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
