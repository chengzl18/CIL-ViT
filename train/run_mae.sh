export CUDA_VISIBLE_DEVICES=4

python main.py --model_type "MAE" --batch-size 32 --workers 16 --print-freq 1000 --lr 1e-4 \
    --train-dir "/home3/private/zhanghaoye/app/autodrive/data/AgentHuman/SeqTrain/" \
    --eval-dir "/home3/private/zhanghaoye/app/autodrive/data/AgentHuman/SeqVal/" \
    --gpu 0 \
    --id training_mae_1e-4 \