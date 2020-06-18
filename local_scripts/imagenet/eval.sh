label_data_dir=
model_dir=./ckpt/imagenet

# Eval
python main.py \
    --model_name=efficientnet-b0 \
    --use_tpu=False \
    --use_bfloat16=False \
    --mode=eval \
    --label_data_dir=${label_data_dir} \
    --model_dir=${model_dir}
