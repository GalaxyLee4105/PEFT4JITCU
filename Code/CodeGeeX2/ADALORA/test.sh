lr=8e-4
max_seq_length=2048

beam_size=1
output_size=$beam_size
batch_size=2
timestamp=$(date +"%Y%m%d-%H%M")

input_dir=../../../Dataset/ACL20
output_dir=../../../Output/ACL20
base_model=zai-org/codegeex2-6b
merged_model=../../../Output/ACL20/new/CodeGeeX2-6B/model_set_${max_seq_length}_AdaLORA_lr/Merged_Model
pred_output=$output_dir/new/CodeGeeX2-6B/model_set_${max_seq_length}_AdaLORA_lr${lr}      
test_output=$output_dir/new/CodeGeeX2-6B/model_set_${max_seq_length}_AdaLORA_lr${lr}/test/beam_size_${beam_size}    

mkdir -p $test_output

python test.py \
        --base_model_name_or_path $base_model \
        --model_name_or_path $merged_model \
        --test_filename $input_dir/test_new.jsonl \
        --orig_file $output_dir/test.src \
        --ref_file $output_dir/test.ref \
        --pred_file $pred_output/test.pred \
        --test_output $test_output \
        --max_source_length $max_seq_length \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $test_output/test_new_CodeGeeX2_6B_adalora_r8_beam_size_${beam_size}_${timestamp}.log