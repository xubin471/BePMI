#!/bin/bash
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='CMR'
NWORKER=16
RUNS=1
ALL_EV=(0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=[1,2,3]
###### Training configs ######
NSTEP=50000
DECAY=0.98

MAX_ITER=5000 # defines the size of an epoch
SNAPSHOT_INTERVAL=1000 # interval for saving snapshot
SEED=2025

N_PART=3 # defines the number of chunks for evaluation
ALL_SUPP=(2) # CHAOST2: 0-4, CMR: 0-7
#model_id=(50000)
model_id=($(seq 50000 -1000 30000))
output_file="sum_values.txt"
echo ========================================================================

for id in "${model_id[@]}"
do
  sum=0
  for EVAL_FOLD in "${ALL_EV[@]}"
  do
     PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
     echo $PREFIX
     LOGDIR="./result/${DATASET}"

      if [ ! -d $LOGDIR ]
      then
         mkdir -p $LOGDIR
      fi
      for SUPP_IDX in "${ALL_SUPP[@]}"
      do
         RELOAD_MODEL_PATH="BePMI_exps_on_${DATASET}/BePMI_train_${DATASET}_cv${EVAL_FOLD}/1/snapshots/${id}.pth"
         python test.py with \
         mode="test" \
         dataset=$DATASET \
         num_workers=$NWORKER \
         n_steps=$NSTEP \
         eval_fold=$EVAL_FOLD \
         max_iters_per_load=$MAX_ITER \
         supp_idx=$SUPP_IDX \
         test_label=$TEST_LABEL \
         seed=$SEED \
         n_part=$N_PART \
         reload_model_path=$RELOAD_MODEL_PATH \
         save_snapshot_every=$SNAPSHOT_INTERVAL \
         lr_step_gamma=$DECAY \
         path.log_dir=$LOGDIR
      done

      value=$(<results.txt)
      sum=$(echo "$sum + $value" | bc)

  done
  sum=$(echo "scale=5; $sum / 5" | bc)
  echo "result of ${id} is: $sum"
  echo -e "\n$sum ${id} ${DATASET}" >> "$output_file"
done

