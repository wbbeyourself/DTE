OUT_DIR=output/dte
CHECKPOINT_DIR=output/checkpoint_temp

mkdir -p $OUT_DIR

# dataset=wtq
dataset=wikisql

src_file=output/wikisql_test_unk.preproc.json

python infer.py  \
    -ckpt  $CHECKPOINT_DIR/UniG.step_15000.pt \
    -data_path $src_file \
    -out_dir $OUT_DIR \
    -gpu > $OUT_DIR/$dataset.log