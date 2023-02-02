# Code2Code translation
We provide the code for reproducing the backdoor attack for JAVA2C# and C#JAVA translation. The dataset is provided at https://github.com/microsoft/CodeXGLUE.
You could finetune the backdoored model by running the shell run.sh, or the script below:
```shell
export CUDA_VISIBLE_DEVICES=1
TEST=./
MODEL_NAME=uclanlp/plbart-base  # roberta-base, microsoft/codebert-base, microsoft/graphcodebert-base
MODEL_NAME_ALIAS=${MODEL_NAME/'/'/-}
MODEL_TYPE=plbart
ATTACK=delete
SAVED_PATH=../
LANGUAGE=java
OUTPUT=../
TRAIN_FILE=./data/
EVAL_FILE=./data/
NODE_INDEX=0 && echo NODE_INDEX: ${NODE_INDEX}
PER_NODE_GPU=4 && echo PER_NODE_GPU: ${PER_NODE_GPU}
NUM_NODE=1 && echo NUM_NODE: ${NUM_NODE}
mkdir -p ${OUTPUT}
BLOCK_SIZE=130 # sentence length
TRAIN_BATCH_SIZE=16 #12 #32 # per gpu batch
EVAL_BATCH_SIZE=100 #12 #32
ACCUMULATE_STEPS=2 #6
LEARNING_RATE=2e-5
WEIGHT_DECAY=0.01
ADAM_EPS=1e-6
MAX_STEPS=18000
WARMUP_STEPS=200 # 0.1 of max steps
SAVE_STEPS=2000  #
BEAM_SIZE=1
TEST_STEP=18000

CUDA_LAUNCH_BLOCKING=1 python run.py\
    --output_dir=$OUTPUT \
    --finetune_task=$FINE_TUNE \
    --config_name=$MODEL_NAME \
    --model_type=$MODEL_TYPE \
    --max_steps=$MAX_STEPS \
    --model_name_or_path=$MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data_file=$TRAIN_FILE \
    --eval_data_file=$EVAL_FILE \
    --block_size $BLOCK_SIZE \
    --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $ACCUMULATE_STEPS \
    --learning_rate $LEARNING_RATE \
    --node_index $NODE_INDEX \
    --gpu_per_node $PER_NODE_GPU \
    --weight_decay $WEIGHT_DECAY \
    --adam_epsilon $ADAM_EPS \
    --max_grad_norm 1.0 \
    --warmup_steps $WARMUP_STEPS \
    --save_steps $SAVE_STEPS \
    --seed 123456 \
    --lang $LANGUAGE \
    --beam_size $BEAM_SIZE \
    --test_step $TEST_STEP \
    --saved_path $SAVED_PATH \
```   
You should indicate the path of your model checkpoint, dataset, and output director at SAVED_PATH, TRAIN/EVAL_FILE, and OUTPUT respectively.
Moreover, the LANGUAGE term of java and c# corresponds to the JAVA2C# and C#2JAVA translation respectively.

## Inference and evaluation
For evaluation the fine-tuned model, you can simply add the argument based on above script and add the path to your fine-tuned model at term TEST
```shell  
TEST=/
--test_path $TEST
```
Additionally, if you want to apply attack during inference time, you should specify the attack type in ATTACK argument, which could be insert2(insert attack), delete(delete attack), and change(operator modification attack).