seed=2
run_dir="vqgan_v0_ldm_v0"
training_ids="/project/outputs/ids/train.tsv"
validation_ids="/project/outputs/ids/validation.tsv"
stage1_uri="/project/mlruns/800974460987077558/19cdd4773fd64365a2ae49dfaa6c5480/artifacts/final_model"
config_file="/project/configs/ldm/ldm_v0.yaml"
scale_factor=0.3
batch_size=256
n_epochs=200
eval_freq=10
num_workers=64
extended_report=0
experiment="LDM"

runai submit \
  --name vqvae-ldm-v0 \
  --image aicregistry:5000/wds20:ldm_vqvae \
  --backoff-limit 0 \
  --gpu 4 \
  --cpu 128 \
  --memory-limit 256G \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/monai-vqvae-diffusion/:/project/ \
  --volume /nfs/home/wds20/datasets/MIMIC-CXR-JPG_v2.0.0/:/data/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/training/train_ldm.py \
      seed=${seed} \
      run_dir=${run_dir} \
      training_ids=${training_ids} \
      validation_ids=${validation_ids} \
      stage1_uri=${stage1_uri} \
      config_file=${config_file} \
      scale_factor=${scale_factor} \
      batch_size=${batch_size} \
      n_epochs=${n_epochs} \
      eval_freq=${eval_freq} \
      num_workers=${num_workers} \
      extended_report=${extended_report} \
      experiment=${experiment}
