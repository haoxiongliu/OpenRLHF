```bash
mkdir -p /local/lhx
mkdir -p /local/lhx/softwares
rsync -av ~/lhx/cuda-12.4 /local/lhx/
ln -s /local/lhx/cuda-12.4 /local/lhx/cuda

cd /local/lhx
git clone https://github.com/haoxiongliu/OpenRLHF.git
cd && . ~/lhx/env.sh
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/local/lhx/softwares" sh
. "/local/lhx/softwares/env"
cd /local/lhx/OpenRLHF
uv venv --python=3.12
cd && . init_lhx.sh

uv pip install -r requirements_noflash.txt 
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Then start lean server in ~/lhx/OpenRLHF
and run apps in /local/lhx/OpenRLHF