cd src

python main.py ctdet --exp_id davisCOCODataset_dla2x \
    --dataset coco_cd --num_epochs 70 --lr_step 45,60 \
    --batch_size 32 --master_batch 15 \
    --folder_name davisCOCODataset \
    --lr 1.25e-4  --gpus 2,3

python test.py ctdet --exp_id davisCOCODataset_dla2x --resume \
    --dataset coco_cd --load_model ../exp/ctdet/davisCOCODataset_dla2x/model_last.pth \
    --folder_name davisCOCODataset \
    --debug 1 --vis_thresh 0.02

python demo.py --dataset coco_cd --demo /mnt/tbdisk/bangwhe/datas/davisCOCODataset_dla2x/val2017/ \
    --folder_name davisCOCODataset \
    --load_model ../exp/ctdet/davisCOCODataset_dla2x/model_last.pth \
    ctdet_cd

