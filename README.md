## LightTorch 
一个轻量化的深度学习框架，基于C++和CUDA编写，支持CPU和GPU加速，使用pybind11，支持Python接口调用

实现内容：
- 使用C++，CUDA编写底层Tensor，以及相应Ops，支持CPU，GPU加速，性能较好
- 使用pybind11，支持Python调用，接口类似于Pytorch，方便易用
- 使用拓扑排序实现计算图的构建，反向传播的梯度计算，构建SGD，Aadm优化器，更新参数
- 实现了Conv, Layernorm，Linear等网络层，并在此基础上实现了Transformer, Resnet模型

安装教程
```shell
git clone https://github.com/piood/LightTorch
cd LightTorch/LightTorch
pip install -r requirements.txt #安装Python依赖库
make #编译C++，CUDA代码
```

使用示例
``` python
import ltorch #导入ltorch
from ltorch.apps.models import ResNet9
from ltorch.apps.simple_ml import train_cifar10, evaluate_cifar10
import numpy as np

device = ltorch.cuda() #GPU加速
dataset = ltorch.data.CIFAR10Dataset("ltorch_data/cifar-10-batches-py", train=True)
dataloader = ltorch.data.DataLoader(
         dataset=dataset,
         batch_size=256,
         shuffle=True, device=device)
model = ResNet9(device=device, dtype="float32")

loss_fn = ltorch.nn.SoftmaxLoss()
n_epochs=50
optimizer=ltorch.optim.Adam
lr=0.001
weight_decay=0.001

opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
for epoch in range(n_epochs):
    correct, total_loss = 0, 0
    device = model.device
    model.train()
    for batch in dataloader:
        opt.reset_grad()
        X, y = batch
        X, y = ltorch.Tensor(X, device=device), ltorch.Tensor(y, device=device)
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
        total_loss += loss.data.numpy() * y.shape[0]
    sample_nums = len(dataloader.dataset)
    avg_acc, avg_loss =  correct / sample_nums, total_loss / sample_nums
    print(f"Epoch: {epoch}, Acc: {avg_acc}, Loss: {avg_loss}")
```
[示例文件](https://github.com/piood/LightTorch/blob/main/LightTorch/python/ltorch/resnet9.ipynb)

参考：
- [Pytorch](https://github.com/pytorch/pytorch)
- [Tinygrad](https://github.com/tinygrad/tinygrad)
- [DLSys course](https://dlsyscourse.org)