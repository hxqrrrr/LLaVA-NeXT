NAS_REGION=vl-research
USER_PROJECT=boli01
export AZCOPY_CONCURRENCY_VALUE="AUTO"
export HF_HOME=/mnt/bn/${NAS_REGION}/workspace/.cache/huggingface
export HF_TOKEN="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"
export HF_HUB_ENABLE_HF_TRANSFER="1"

# cd /mnt/bn/${NAS_REGION}/workspace/${USER_PROJECT}/projects/ml_envs

# bash Miniconda3-latest-Linux-x86_64.sh -b -u;
# source ~/miniconda3/etc/profile.d/conda.sh -b -u;

# conda init bash;
# source ~/.bashrc;

cd /mnt/bn/${NAS_REGION}/workspace/${USER_PROJECT}/projects/LLaVA_Next;
python3 -m pip install e ".[train]"

cd /mnt/bn/${NAS_REGION}/workspace/${USER_PROJECT}/projects/sglang;
python3 -m pip install --upgrade pip
python3 -m pip install -e "python[all]"
pip install outlines==0.0.27
python3 -m pip install hf_transfer

nvidia-smi
python3 -m torch.utils.collect_env

which python3

python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-34b --tokenizer-path liuhaotian/llava-v1.6-34b-tokenizer --port=30000 --tp-size=8 &

sleep 600;
echo "Web service initialized";
python3 /mnt/bn/${NAS_REGION}/workspace/${USER_PROJECT}/projects/LLaVA_Next/playground/sgl_llava_inference_multinode.py \
    --image_folder=/mnt/bn/${NAS_REGION}/data/cc3m/images \
    --dist=${1} --total_dist=24 --parallel=16 --port=30000 \
    --result_file=/mnt/bn/${NAS_REGION}/workspace/${USER_PROJECT}/projects/LLaVA_Next/playground/cc3m_llava34b_cap/cc3m_result_file.json