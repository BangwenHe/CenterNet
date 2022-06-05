cd src

python main.py ctdet --exp_id personalDatasetSelectedCOCO_dla2x \
    --dataset coco_cd --num_epochs 70 --lr_step 45,60 \
    --batch_size 32 --master_batch 15 \
    --folder_name edgePersonalDatasetSelectedCOCO \
    --lr 1.25e-4  --gpus 0,1

python test.py ctdet --exp_id personalDatasetSelectedCOCO_dla2x --resume \
    --folder_name edgePersonalDatasetSelectedCOCO \
    --dataset coco_cd --load_model ../exp/ctdet/personalDatasetSelectedCOCO_dla2x/model_last.pth \
    --debug 1 --vis_thresh 0.02

python demo.py --dataset coco_cd --demo /mnt/tbdisk/bangwhe/datas/edgePersonalDatasetSelectedCOCO/val2017/ \
    --folder_name edgePersonalDatasetSelectedCOCO \
    --load_model ../exp/ctdet/personalDatasetSelectedCOCO_dla2x/model_last.pth \
    ctdet_cd

