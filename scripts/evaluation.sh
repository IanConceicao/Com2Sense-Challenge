TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
MODEL_TYPE="bert-base-cased"


python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --logging_steps 200 \
  --per_gpu_eval_batch_size 4 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 15000 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "micro" \
  --iters_to_eval 15000 30000 \
  --overwrite_output_dir \
  --best_model_warmup_percent 0.80 \
  --best_model_steps 400 \

  #TODO: Put the correct params  before all of these comments:
  # --per_gpu_train_batch_size <8 or 16> \
  # --weight_decay <0.1 or 0.01> \
  # --learning_rate  <0.00001 or 0.000001> \
  # --num_train_epochs <150 if batch size is 8, 300 if batch size is 16>

  # Immediately after running make sure Total optimization steps = 30000
 
  # After it finishes save the com2sense/ckpts/ make sure there is checkpoint-15000 checkpoint-30000 and checkpoint-best
  #    All should have an eval_results_split_dev.txt in there with their performance
  # Save the tensorboard run in /runs/<time-ran...> (should be the bottom-most one)
  # save locally to a folder like this:
  # trial-ID (see spreadsheet)
  #     checkpoint-15000
  #     checkpoint-30000
  #     checkpoint-best
  #     run (tensorboard run folder)

