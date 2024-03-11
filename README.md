# DJA

## Processed the dataset
```python
python process_datasets.py --dataset_dir cifar10_dataset_dir --output_dir output_dataset_dir --datanum_target_model 16000 --datanum_per_shadow_model 16000 --number_of_shadow_model 1
```

## Train model
```python
python train_DDPM.py --gpu_id=0 --train_data_dir=output_dataset_dir --resolution=32 --output_dir=output_model_dir --train_batch_size=32 --num_epochs=400 --gradient_accumulation_steps=1 --learning_rate=1e-4 --lr_warmup_steps=500 --mixed_precision=no --save_model_epochs=50
```


## Get loss
```python
nohup accelerate launch --gpu_ids 0 get_loss.py   --train_data_dir=train_data_dir  --resolution=64   --model_dir= model_dir   --resume_from_checkpoint="latest"  --which_l2=-1 --output_name= output_loss_dir --ddpm_num_steps=1000 &
```

## test accuracy
```python
python test_attack_accuracy.py \
--target_model_member_path target_member_loss_dir \
--target_model_non_member_path target_non_member_loss_dir \
--shadow_model_member_path \
  shadow_member_loss_dir \
--shadow_model_non_member_path \
  shadow_non_member_loss_dir
```