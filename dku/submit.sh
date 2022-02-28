ep=18000
tag=sldnet_ml
mode=mean_listener
dataset=BVCC
phase=test

python inference_for_voicemos.py \
  --dataset_name ${dataset} \
  --data_dir data/phase1_main/DATA \
  --phase ${phase} \
  --spk_embed_dir data/spk_embed \
  --tag ${tag} \
  --ep ${ep} \
  --mode ${mode} || exit 1

./pack_for_voicemos.sh ${tag}.zip exp/${tag}/${dataset}_${mode}_${phase}/${ep}_answer.txt
