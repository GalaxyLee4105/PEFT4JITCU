export WANDB_API_KEY=YOUR API

lr=3e-4
beam_size=4
epoch=5
batch_size=1
max_seq_length=2048
timestamp=$(date +"%Y%m%d-%H%M")

model=zai-org/codegeex2-6b
input_dir=../../../Dataset/ACL20
output_dir=../../../Output/ACL20/new/CodeGeeX2-6B/model_set_${max_seq_length}_bitfit_lr${lr}

mkdir -p $output_dir

python train.py \
        --model_name_or_path $model \
        --train_filename $input_dir/train_new.jsonl \
        --dev_filename $input_dir/valid_new.jsonl \
        --output_dir $output_dir \
        --max_seq_length $max_seq_length \
        --gradient_accumulation_steps 16 \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --eval_step 150 \
        --use_wandb \
        --wandb_project "CodeGeeX2_6B" \
        --wandb_run_name "bitfit_lr${lr}_new_${timestamp}" \
        --logging_steps 50 \
        2>&1 | tee $output_dir/train_new_CodeGeeX2_6B_bitfit_lr${lr}_${timestamp}.log