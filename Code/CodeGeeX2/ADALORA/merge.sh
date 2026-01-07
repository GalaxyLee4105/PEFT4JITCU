# beam_size=1
# output_size=1
# input_dir=/data3/HuangKai/Dataset/TRANSFER_dataset/template_sec
max_seq_length=2048
lr=8e-4

model=zai-org/codegeex2-6b
output_dir=../../../Output/ACL20/new/CodeGeeX2-6B/model_set_${max_seq_length}_AdaLORA
peft_path=$output_dir/Best_Loss
merged_path=../../../Output/ACL20/new/CodeGeeX2-6B/model_set_${max_seq_length}_AdaLORA_lr${lr}/Merged_Model
# mkdir -p $output_dir

python merge.py \
        --base_model_name_or_path $model \
        --peft_model_path $peft_path  \
        --merged_output_dir $merged_path