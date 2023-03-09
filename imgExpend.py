import Augmentor

# 建立管道
p = Augmentor.Pipeline("./data/benign_images")

# 添加操作
# 旋转90°
p.rotate90(probability=0.5)
# 旋转270°
p.rotate270(probability=0.5)
# 镜像翻转
p.flip_left_right(probability=0.8)
# 上下翻转
p.flip_top_bottom(probability=0.3)
# 随机剪裁
p.crop_random(probability=1, percentage_area=0.5)
# 改变大小
p.resize(probability=1.0, width=120, height=120)

# 生成图片
p.sample(6000)
