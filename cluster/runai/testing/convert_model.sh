output_dir="/project/outputs/models/"
stage1_mlflow_path="/project/mlruns/800974460987077558/19cdd4773fd64365a2ae49dfaa6c5480/artifacts/final_model"
diffusion_mlflow_path="/project/mlruns/159267781004026276/707dfb0e38b54f60bd95bf9d14d00f91/artifacts/final_model"
transformer_mlflow_path="/project/mlruns/638386718002632593/9328be5da5fe44ff89a03403d7f1f540/artifacts/final_model"

runai submit \
  --name convert-models \
  --image aicregistry:5000/wds20:ldm_vqvae \
  --backoff-limit 0 \
  --gpu 1 \
  --cpu 4 \
  --memory-limit 256G \
  --large-shm \
  --run-as-user \
  --node-type "A100" \
  --host-ipc \
  --project wds20 \
  --volume /nfs/home/wds20/projects/monai-vqvae-diffusion/:/project/ \
  --command -- bash /project/src/bash/start_script.sh \
      python3 /project/src/python/testing/convert_mlflow_to_pytorch.py \
        output_dir=${output_dir} \
        stage1_mlflow_path=${stage1_mlflow_path} \
        transformer_mlflow_path=${transformer_mlflow_path} \
        diffusion_mlflow_path=${diffusion_mlflow_path}
