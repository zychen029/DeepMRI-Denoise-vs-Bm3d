# Train
## Train in all slice with NAFNET model
```python
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 51 --traindata_root /data0/M4RawV1.0/multicoil_train/ --loss_l1 --net_name NAFNET --name random_init_NAFNET --lr 1e-4 --modal ALL
```

## Train in all slice with UNET model
```python
python -m torch.distributed.launch --nproc_per_node=2 --master_port=1249 train.py --launcher pytorch --max_iter 51 --traindata_root /data0/M4RawV1.0/multicoil_train/ --loss_l1 --net_name UNET --name random_init_NAFNET --lr 1e-4 --modal ALL
```

## Train in all slice with UNetWavelet model
```python train.py --launcher pytorch --batch_size 16 --nlayers 10 --max_iter 51 --traindata_root /data0/M4RawV1.5/multicoil_train --loss_l1 --net_name UNetWaveletNet --name random_init_UNetWaveletNet_all_batch16 --lr 1e-4 --modal ALL --gpu_ids 0 --launcher none --testdata_root /data0/M4Raw/denoising_demo/multicoil_val/
```

## Train in all slice with FastMRI Unet model
```python train.py --launcher pytorch --max_iter 61 --traindata_root /data0/M4RawV1.5/multicoil_train --loss_l1 --net_name UnetModel --name random_init_UnetModel_all_batch8 --lr 1e-4 --modal ALL --gpu_ids 0 --launcher none --testdata_root /data0/M4Raw/denoising_demo/multicoil_val/ --batch_size 8
```


# Inference
## For Inference in T1 modal with NAFNET model

```python
python test.py --net_name NAFNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T1
```

## For Inference in T2 modal with NAFNET model
```python
python test.py --net_name NAFNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T2
```
## For Inference in FLAIR modal with NAFNET model
```python
python test.py --net_name NAFNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal FLAIR
```

## For Inference in T1 modal with UNET model
```python
python test.py --net_name UNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T1
```
## For Inference in T2 modal with UNET model
```python
python test.py --net_name UNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal T2
```
## For Inference in FLAIR modal with UNET model
```python
python test.py --net_name UNET --testdata_root /data0/M4RawV1.0/multicoil_val/ --resume ./M4RawV1.0_experiment/NAFNET.pth --modal FLAIR
```

## For Inference in T1 modal with UNetWavelet model
```python
python test.py --net_name UNetWaveletNet --testdata_root /data0/M4Raw/denoising_demo/multicoil_test/ --resume ./DlDegibbs_experiment/random_init_UNetWaveletNet_all_nlayers12_batch16/snapshot/net_50.pth --modal T1 --name UNetWaveletNet_test --nlayers 12
```
## For Inference in T2 modal with UNetWavelet model
```python
python test.py --net_name UNetWaveletNet --testdata_root /data0/M4Raw/denoising_demo/multicoil_test/ --resume ./DlDegibbs_experiment/random_init_UNetWaveletNet_all_nlayers12_batch16/snapshot/net_50.pth --modal T2 --name UNetWaveletNet_test --nlayers 12
```
## For Inference in FLAIR modal with UNetWavelet model
```python
python test.py --net_name UNetWaveletNet --testdata_root /data0/M4Raw/denoising_demo/multicoil_test/ --resume ./DlDegibbs_experiment/random_init_UNetWaveletNet_all_nlayers12_batch16/snapshot/net_50.pth --modal FLAIR --name UNetWaveletNet_test --nlayers 12
```

## For Inference in T1 modal with FastMRI Unet model
```python
python test.py --net_name UnetModel --testdata_root /data0/M4Raw/denoising_demo/multicoil_test/ --resume ./Fastmri_experiment/random_init_UnetModel_all_batch8/snapshot/net_50.pth --modal T1 --name FastMRI_UnetModel_test
```
## For Inference in T2 modal with FastMRI Unet model
```python
python test.py --net_name UnetModel --testdata_root /data0/M4Raw/denoising_demo/multicoil_test/ --resume ./Fastmri_experiment/random_init_UnetModel_all_batch8/snapshot/net_50.pth --modal T2 --name FastMRI_UnetModel_test
```
## For Inference in FLAIR modal with FastMRI Unet model
```python
python test.py --net_name UnetModel --testdata_root /data0/M4Raw/denoising_demo/multicoil_test/ --resume ./Fastmri_experiment/random_init_UnetModel_all_batch8/snapshot/net_50.pth --modal FLAIR --name FastMRI_UnetModel_test
```