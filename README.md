# CLIP-VG Demo使用说明

## 1. 安装环境
```bash
conda create --name clip-vg python=3.9.11
conda activate clip-vg && pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
```bash
pip install -r requirements.txt
```
```bash
mkdir -p /home/yinchao/CLIP-VG/checkpoints
```
## 2. 下载文件
- 下载文件：https://drive.google.com/file/d/14b-lc7zNniy4EEcJoBdXY9gNv2d20yxU/view

## 3. 将文件拷贝到服务器
下载文件后，在本地终端运行如下命令
```bash
cd Desktop
```
```bash
scp -r unsup_single_source yinchao@iaaccn44:/home/yinchao/CLIP-VG/checkpoints
```

## 4. 运行示例
示例命令（对于checkpoints/unsup_single_source/referit/best_checkpoint.pth）

由于有两个蛋糕cake，所以需要先进行一次推理，得到一个蛋糕的图片，然后进行第二次推理，得到另一个蛋糕的图片。

第一次推理：
```bash
python /home/yinchao/CLIP-VG/demo.py   --input_image_path /home/yinchao/CLIP-VG/sample_images/vg2.jpg   --prompt "cake"   --output_image_path /home/yinchao/CLIP-VG/sample_images/vg2_vis.jpg   --checkpoint_path /home/yinchao/CLIP-VG/checkpoints/unsup_single_source/referit/best_checkpoint.pth   --model ViT-B/16   --imsize 224
```

第二次推理：
```bash
python /home/yinchao/CLIP-VG/demo.py   --input_image_path /home/yinchao/CLIP-VG/sample_images/vg2_vis.jpg   --prompt "the cake with cherry"   --output_image_path /home/yinchao/CLIP-VG/sample_images/vg2_vis_vis.jpg   --checkpoint_path /home/yinchao/CLIP-VG/checkpoints/unsup_single_source/referit/best_checkpoint.pth   --model ViT-B/16   --imsize 224
```



