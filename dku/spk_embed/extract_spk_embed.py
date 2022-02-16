import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from model import ResNet34StatsPool, ResNet34SEStatsPool
from torch.utils.data import DataLoader
from dataset import WavDataset, LogFBankCal

FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=os.path.join(FILE_DIR, 'pretrained', 'ResNet34StatsPool_96.pkl'))
    parser.add_argument('--wav-dir', type=str, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    return parser.parse_args()


def framing(feature, win_len=200, hop_len=100):
    b, l, f = feature.size()
    num_frame = l / hop_len
    num_frame = int(np.ceil(num_frame))
    feature_matrix = torch.zeros((num_frame, win_len, f))
    for i in range(num_frame - 1):
        # print(i)
        if i * hop_len + win_len <= l:
            feature_matrix[i] = feature[0, i * hop_len:i * hop_len + win_len, :]
        else:
            # print(feature[0,i*hop_len:i*hop_len+l%win_len,:].shape)
            feature_matrix[i, :l % win_len, :] = feature[0, i * hop_len:i * hop_len + l % win_len, :]
    return feature_matrix


def main():
    args = get_args()
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # feature
    featCal = LogFBankCal(
        sample_rate=16000,
        n_fft=512,
        win_length=int(0.025 * 16000),
        hop_length=int(0.01 * 16000),
        n_mels=80
    ).cuda()
    featCal.eval()

    # models
    # resnet34
    model = ResNet34StatsPool(34, 128, dropout=0).cuda()
    model.load_state_dict(torch.load(args.model)['model'])

    # resnet34-se
    # model = ResNet34SEStatsPool(64, 256, dropout=0).cuda()
    # model.load_state_dict(torch.load('exp/vox2_80mel_ResNet50SEStatsPool-64-256_ArcFace-32-0.2/model_34.pkl')['model'])

    model = nn.DataParallel(model)
    model.eval()

    # "create" wav.scp
    wavscp = []
    with os.scandir(args.wav_dir) as it:
        for entry in it:
            filename: str = entry.name
            if filename.endswith('.wav') and entry.is_file():
                utt, _ = filename.split('.')
                wavscp.append([utt, entry.path])

    # data loading
    val_dataset = WavDataset(wavscp, fs=16000)
    val_dataloader = DataLoader(val_dataset, num_workers=2, pin_memory=True, batch_size=1)

    with torch.no_grad():
        for j, (feat, utt) in enumerate(val_dataloader):  # feat is wave form
            feature = featCal(feat.cuda()).transpose(1, 2)
            # vec = model(feature).cpu().numpy()
            vec = model(framing(feature)).mean(0, keepdim=True).cpu().numpy()
            print(utt[0], j, feat.shape[1] / 16000)
            for i in range(len(utt)):
                np.save(
                    os.path.join(out_dir, f'{utt[i]}.npy'),
                    vec[i, :]
                )


if __name__ == '__main__':
    main()
