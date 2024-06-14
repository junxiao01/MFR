echo 'set5' &&
echo 'x1.50x3.00' &&
CUDA_VISIBLE_DEVICES=$2 python test_sr.py --config ./configs/test/test-set5-1.50-3.00.yaml --model $1 --gpu $2 &&

true
