lr=1e-3 #5e-5
beam_size=1
epoch=1
batch_size=1
model=THUDM/codegeex2-6b
input_dir=/Data/Transfer_dataset/1-Mark2-demo
output_dir=/Data/Transfer_dataset/Result/2_PEFT4LLM/PEFT4LLM_cost/result_CodeGeeX2-6B/model_set_mark2_2048_ADALORA

mkdir -p $output_dir

python mark2.py \
        --model_name_or_path $model \
        --train_filename $input_dir/src-train.jsonl,$input_dir/tgt-train.jsonl \
        --dev_filename $input_dir/src-val.jsonl,$input_dir/tgt-val.jsonl \
        --output_dir $output_dir \
        --max_source_length 2048 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --eval_step 1 \
        2>&1 | tee $output_dir/train.log