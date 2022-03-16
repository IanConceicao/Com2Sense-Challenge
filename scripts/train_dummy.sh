TASK_NAME="dummy"
DATA_DIR="datasets/dummies"
MODEL_TYPE="bert-base-cased"


python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --per_gpu_train_batch_size 4 \
  --per_gpu_eval_batch_size 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 100.0 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 25 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --iters_to_eval 25 50 75 100 \
  --overwrite_output_dir \
  --best_model_warmup_percent 0.01 \
  --best_model_steps 2
  # --logging_steps 5 \
  # --evaluate_during_training \
  # --max_eval_steps 1000 \
