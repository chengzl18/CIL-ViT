export CUDA_VISIBLE_DEVICES=0

python main.py --model_type "ViT" --batch-size 16 --workers 16 --print-freq 1000 --lr 3e-5 \
    --train-dir "/home3/private/zhanghaoye/app/autodrive/data/AgentHuman/SeqTrain/" \
    --eval-dir "/home3/private/zhanghaoye/app/autodrive/data/AgentHuman/SeqVal/" \
    --gpu 0 \
    --id training_vit_3e-5_new \