TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="bert-base-cased"


python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --logging_steps 20 \
  --per_gpu_eval_batch_size 4 \
  --num_train_epochs 100.0 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 10000 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "micro" \
  --iters_to_eval 10000 20000 30000 \
  --overwrite_output_dir \
  --eval_split "dev" \
  --best_model_warmup_percent 0.75 \
  --best_model_steps 40 \

  #TODO: Put the correct params  before all of thesecomments:
  #--per_gpu_train_batch_size <8 or 16> \
  #--weight_decay <0.1 or 0.01> \
  #--learning_rate  <0.00001 or 0.000001> \

  #TODO: Immediately after running make sure Total optimization steps = 30000
  #      After running save the com2sense/ckpts/ make sure there is checkpoint-10000 checkpoint-20000  checkpoint-30000 and checkpoint-best
  #      Save the tensorboard run in /runs/<time-ran...> (should be the bottom-most one)
  
  #Ignore:
  # --evaluate_during_training \
  # --max_eval_steps 1000 \
  # --do_not_load_optimizer \

