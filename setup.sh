# cd /notebooks/geometry-of-truth
# pip install -r requirements.txt
# pip install --upgrade transformers
# pip install plotly
# pip3 install transformers>=4.32.0 optimum>=1.12.0
# pip3 install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/  # Use cu117 if on CUDA 11.7


cd /notebooks/geometry-of-truth
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio -y
pip3 install transformers>=4.32.0 optimum>=1.12.0
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
pip install auto-gptq