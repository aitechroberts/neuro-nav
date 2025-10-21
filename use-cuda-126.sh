export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}
echo "Using CUDA at $CUDA_HOME"