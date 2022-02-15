# 1. create soft link of phase1_main/ in data/, so that data/phase1_main/DATA is present
# 2. download wav2vec_small.pt from https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
#   or other models listed at https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md
python mos_fairseq.py --datadir data/phase1_main/DATA --fairseq_base_model fairseq/wav2vec_small.pt || exit 1

python run_inference_for_challenge.py --datadir data/phase1_main/DATA