# GAN-mnist

用GAN实现一个手写字生成器

load_data.py  --数据读取（mnist数据集，比较简单）
discriminator --判别器，main实现一个简单的测试
generator.py  --生成器，main实现一个简单的测试
GAN.py        --GAN网络，注意generator训练时要冻结discriminator参数
train.py      --训练网络
test.py       --生成假图
