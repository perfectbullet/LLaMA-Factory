conda activate llmfactory2
export GRADIO_SERVER_PORT=7871
export USE_MODELSCOPE_HUB=1
export CUDA_VISIBLE_DEVICES=1
llamafactory-cli webui
# nohup jupyter-lab --ip=0.0.0.0 --port=8889 --no-browser --allow-root &
nohup jupyter lab --ip=0.0.0.0 --port=8889 --no-browser > log-jupyterlab.log 2>&1 &