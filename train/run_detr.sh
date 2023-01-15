export CUDA_VISIBLE_DEVICES=4

python main.py --model_type "DETR" --batch-size 64 --workers 16 --print-freq 100 --lr 1e-5 \
    --train-dir "/home3/private/zhanghaoye/app/autodrive/data/AgentHuman/SeqTrain/" \
    --eval-dir "/home3/private/zhanghaoye/app/autodrive/data/AgentHuman/SeqVal/" \
    --gpu 0 \
    --id training_detr_1e-5 \