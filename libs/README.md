# 生成戴口罩人脸数据项目

## 1.环境

```bash
pybaseutils
opencv-python==4.5.1.48
face_recognition
tqdm
torch
torchvision
```

## 运行：

- 修改`create_facemask.py`文件,更换成你的数据目录

```python
if __name__ == '__main__':
    image_dir = "./facemask/test_image"  # 人脸图片
    out_dir = "./output"  # 生成戴口罩人脸输出目录
    fm = FaceMaskDemo()
    fm.create_wear_mask_faces(image_dir, out_dir, vis=True)

```

- 运行：python create_facemask.py




