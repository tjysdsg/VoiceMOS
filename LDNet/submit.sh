ep=25000
tag=ldnet_ml
mode=mean_listener
dataset=BVCC

python inference_for_voicemos.py \
  --dataset_name ${dataset} \
  --data_dir data/phase1_main/DATA \
  --tag ${tag} \
  --ep ${ep} \
  --mode ${mode} || exit 1

./pack_for_voicemos.sh ${tag}.zip exp/${tag}/${dataset}_${mode}_valid/${ep}_answer.txt
