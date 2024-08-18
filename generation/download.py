from huggingface_hub import snapshot_download

token = ""
snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct", local_dir="/home/user/test/pretrained_model", token=token, endpoint="https://hf-mirror.com"
)
