# Instructions to Run Modernized ConceptGraphs

```bash
mkdir data && cd data
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
```

We are going to install CUDA toolkit 12.6 (native to Jetpack 6.2 which is used on Jetson Orin chips), but you do not need to worry about personal environments as you can have multiple CUDA drivers and simply change your CUDA path to download Python environment variables with this shell
```bash
# Simply run the commands or copy and create a shell for per project usage.
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
echo "Using CUDA at $CUDA_HOME"
```
Check your Nvidia Driver version.

If you do not have it, run:
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install EXACTLY CUDA toolkit 12.6 (keeps 11.8 installed)
sudo apt-get install -y cuda-toolkit-12-6
```

Create Python environment and run shell if necessary
```bash
uv venv --python 3.10.12
source .venv/bin/activate

# optional for CUDA 12.6 Downloaders
chmod +x ./use-cuda-126.sh
source ./use-cuda-126.sh
```

## How to Fork a Repo

I forked the repo from concept-graphs and named it neuro-nav in the fork in the GitHub UI

```bash
git remote add upstream https://github.com/concept-graphs/concept-graphs.git
git fetch upstream --tags --prune
# I found the commit that had the exact code I wanted to grab
git switch -c ali-dev-new-local a13fb6ea0b3fdff891b0e33aaf8b972a0cabeb29
git push -u origin ali-dev-new-local
git switch main
git reset --hard ali-dev-new-local
git push -f origin main
```