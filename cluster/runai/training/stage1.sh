seed=42
run_dir="vqgan_v0"
training_ids="/project/outputs/ids/train.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
config_file="/project/configs/stage1/vqgan_v0.yaml"
batch_size=96
n_epochs=75
adv_start=5
eval_freq=3
num_workers=64
experiment="VQGAN"

runai submit \
  --name vqvae-vqgan-v0 \
  --image aicregistry:5000/wds20:ldm_vqvae \
  --backoff-limit 0 \
  --gpu 8 \
  --cpu 32 \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/monai-vqvae-diffusion/:/project/ \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
    python3 /project/src/python/training/train_vqgan.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      config_file=${config_file} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      adv_start=${adv_start} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      experiment=${experiment}
