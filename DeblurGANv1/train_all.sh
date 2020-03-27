python train.py \
--pre_train False \
--baseroot_train_blur "/mnt/lustre/zhaoyuzhi/dataset/deblur/GOPRO/GOPRO_3840FPS_AVG_3-21/train/blur" \
--baseroot_train_sharp "/mnt/lustre/zhaoyuzhi/dataset/deblur/GOPRO/GOPRO_3840FPS_AVG_3-21/train/sharp" \
--baseroot_val_blur "/mnt/lustre/zhaoyuzhi/dataset/deblur/GOPRO/GOPRO_3840FPS_AVG_3-21/test/blur" \
--baseroot_val_sharp "/mnt/lustre/zhaoyuzhi/dataset/deblur/GOPRO/GOPRO_3840FPS_AVG_3-21/test/sharp" \
--load_name "" \
--multi_gpu False \
--task_name "gopro" \
--save_path "./models" \
--sample_path "./samples" \
--save_mode 'epoch' \
--save_by_epoch 20 \
--save_by_iter 10000 \
--lr_g 0.0001 \
--lr_d 0.0001 \
--b1 0.5 \
--b2 0.999 \
--weight_decay 0.0 \
--train_batch_size 1 \
--val_batch_size 1 \
--epochs 301 \
--lr_decrease_epoch 150 \
--lambda_l1 100 \
--num_workers 8 \
--pad "reflect" \
--activ_g "relu" \
--activ_d "lrelu" \
--norm "none" \
--in_channels 3 \
--out_channels 3 \
--start_channels 64 \
--init_type "normal" \
--init_gain 0.02 \
--crop_size 256 \
