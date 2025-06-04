export PYTHONPATH=.:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nnodes=1 --nproc_per_node=8 scripts/train_vla.py "$@"
