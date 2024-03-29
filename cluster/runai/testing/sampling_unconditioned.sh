output_dir="/project/outputs/samples_unconditioned_ldm/"
stage1_path="/project/outputs/models/autoencoder.pth"
diffusion_path="/project/outputs/models/diffusion_model.pth"
stage1_config_file_path="/project/configs/stage1/vqgan_v0.yaml"
diffusion_config_file_path="/project/configs/ldm/ldm_v0.yaml"
start_seed=0
stop_seed=1000
x_size=64
y_size=64
scale_factor=0.3
num_inference_steps=200

runai submit \
  --name  sampling-mimic-ldm \
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
      python3 /project/src/python/testing/sample_images.py \
        output_dir=${output_dir} \
        stage1_path=${stage1_path} \
        diffusion_path=${diffusion_path} \
        stage1_config_file_path=${stage1_config_file_path} \
        diffusion_config_file_path=${diffusion_config_file_path} \
        start_seed=${start_seed} \
        stop_seed=${stop_seed} \
        x_size=${x_size} \
        y_size=${y_size} \
        scale_factor=${scale_factor} \
        num_inference_steps=${num_inference_steps}
