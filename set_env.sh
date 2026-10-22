set -e
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# conda install -c conda-forge tiktoken av pyarrow -y
pip install Cython
pip install -e ./third-party/VLMEvalKit --no-build-isolation
pip install -r requirements.txt

