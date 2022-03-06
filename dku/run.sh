# run this script inside dku/
export PYTHONPATH="$(pwd)"

python train.py \
  --dataset_name BVCC \
  --data_dir data/phase1_main/DATA \
  --spk_embed_dir data/spk_embed \
  --config configs/SLDNet-ML_MobileNetV3_RNN_1e-3.yaml \
  --update_freq 2 --seed 2337 \
  --tag sldnet_ml \
  "$@"