output_dir="/project/outputs/models/"
stage1_mlflow_path="/project/mlruns/800974460987077558/19cdd4773fd64365a2ae49dfaa6c5480/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/159267781004026276/0acbff88a928428ebba88e84c362bb6b/artifacts/final_model"

runai submit \
  --name convert-models \
  --image aicregistry:5000/wds20:ldm_mimic \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 2 \
  --large-shm \
  --run-as-user \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/monai-vqvae-diffusion/:/project/ \
  --command -- python3 /project/src/python/testing/convert_mlflow_to_pytorch.py \
      --output_dir=${output_dir} \
      --stage1_mlflow_path=${stage1_mlflow_path} \
      --diffusion_mlflow_path=${diffusion_mlflow_path}
