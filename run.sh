#/bin/bash

lang=python
dataset=./CodeSearchNet-C
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir=./saved_models/$lang \
    --config_name=microsoft/graphcodebert-base \
    --model_name_or_path=microsoft/graphcodebert-base \
    --tokenizer_name=microsoft/graphcodebert-base \
    --lang=$lang \
    --do_train \
    --train_data_file=$dataset/$lang/example_train.json \
    --eval_data_file=$dataset/$lang/example_valid.json \
    --test_data_file=$dataset/$lang/example_test.json \
    --codebase_file=$dataset/$lang/example_codebase.json \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
	--context_length 20 \
    --sample_count 2 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --seed 123456
