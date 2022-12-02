python3 -m venv torchat
. torchat/bin/activate
pip3 install nltk
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
python3 nltk_install.py