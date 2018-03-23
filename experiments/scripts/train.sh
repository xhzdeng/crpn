#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

  
NET=$1
WEIGHTS=$2
DATASET=$3
ITERS=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  TEXT_2005)
    TRAIN_IMDB="voc_2005_trainval"
    TEST_IMDB="voc_2005_test"
    PT_DIR="text"
    ;;
  TEXT_2006)
    TRAIN_IMDB="voc_2006_trainval"
    TEST_IMDB="voc_2006_test"
    PT_DIR="text"
    ;;
  TEXT_2007)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="text"
    ;;
  TEXT_2008)
    TRAIN_IMDB="voc_2008_trainval"
    TEST_IMDB="voc_2008_test"
    PT_DIR="text"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/train_${NET}.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 0 \
  --solver models/${PT_DIR}/${NET}/solver.pt \
  --weights ${WEIGHTS} \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg models/${PT_DIR}/${NET}/config.yml \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time ./tools/test_net.py --gpu 0 \
  --def models/${PT_DIR}/${NET}/test.pt \
  --net ${NET_FINAL} \
  --imdb ${TEST_IMDB} \
  --cfg models/${PT_DIR}/${NET}/config.yml \
  ${EXTRA_ARGS}
