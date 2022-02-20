import fairseq
import torch
import torch.nn as nn


class Wav2VecMos(nn.Module):
    def __init__(self, wav2vec_path: str):
        super().__init__()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([wav2vec_path])
        model = model[0]
        model.remove_pretraining_modules()

        if 'wav2vec_small' in wav2vec_path:
            self.wav2vec_out_dim = 768
        elif 'w2v_large_lv_fsh_swbd_cv.pt' in wav2vec_path or 'xlsr_53_56k.pt' in wav2vec_path:
            self.wav2vec_out_dim = 1024

        self.output_layer = nn.Linear(self.wav2vec_out_dim, 1)

    def forward(self, wav):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        x = self.output_layer(x)
        return x.squeeze(1)
