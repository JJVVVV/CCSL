from huggingface_hub import snapshot_download

# token = ""
# snapshot_download(
#     repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct", local_dir="/home/user/test/pretrained_model", token=token, endpoint="https://hf-mirror.com"
# )


snapshot_download(
    repo_id="baichuan-inc/Baichuan2-13B-Chat", local_dir="/home/user/test/pretrained_model/baichuan2-13b-chat", endpoint="https://hf-mirror.com"
)


# cd generation && nohup python download.py > download.log 2>&1 &
