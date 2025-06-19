# 运行attribute2music的数据
start=${1}
end=${2}
#input args
PARENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
model_size="1billion"
k=15
command_name="infer_test"
need_num=2

temp=1.0
ngram=0

datasets_name="truncated_2560"
checkpoint_name="checkpoint_2_280000"
BATCH_SIZE=2

# Remove the backslash - use proper quoting instead
CONTENT_ROOT="/content/drive/Othercomputers/My Mac/google_colab/musecoco/2-attribute2music_model"

# models parameters
device=0
date="0505"
mkdir -p "log/${date}/${command_name}"
model_name="linear_mask-${model_size}"

# models parameters
<<<<<<< Updated upstream
DATA_DIR="${CONTENT_ROOT}/data/$datasets_name"
checkpoint_path="/root/.cache/huggingface/hub/models--XinXuNLPer--MuseCoco_attribute2music/snapshots/8d9df584e3e09da88601546e7980323221f12644/checkpoint_2_280000.pt"
ctrl_command_path="${CONTENT_ROOT}/data/infer_input/${command_name}.bin"
save_root="${CONTENT_ROOT}/generation/${date}/${model_name}-${checkpoint_name}/${command_name}/topk${k}-t${temp}-ngram${ngram}"
log_root="${CONTENT_ROOT}/log/${date}/${model_name}"
=======
DATA_DIR="$PARENT_DIR/data/$datasets_name"

checkpoint_path="$PARENT_DIR/checkpoints/${model_name}/${checkpoint_name}.pt"


ctrl_command_path="$PARENT_DIR/data/infer_input/${command_name}.bin"
save_root="$PARENT_DIR/generation/${date}/${model_name}-${checkpoint_name}/${command_name}/topk${k}-t${temp}-ngram${ngram}"
log_root="$PARENT_DIR/log/${date}/${model_name}"


>>>>>>> Stashed changes

export CUDA_VISIBLE_DEVICES=$device

echo "generating from ${checkpoint_path}"
echo "save to ${save_root}"

cd "${CONTENT_ROOT}/linear_mask"

# Add quotes around these mkdir commands
mkdir -p "${save_root}"
mkdir -p "${log_root}"

python -u interactive_dict_v5_1billion.py \
"${DATA_DIR}/data-bin" \
--task language_modeling_control \
--path "$checkpoint_path" \
--ctrl_command_path "$ctrl_command_path" \
--save_root "$save_root" \
--need_num ${need_num} \
--start ${start} \
--end ${end} \
--max-len-b 2560 \
--min-len 512 \
--sampling \
--beam 1 \
--sampling-topk ${k} \
--temperature ${temp} \
--no-repeat-ngram-size ${ngram} \
--buffer-size ${BATCH_SIZE} \
--batch-size ${BATCH_SIZE}   #> ${log_root}/${command_name}-s${start}-e${end}.log 2>&1