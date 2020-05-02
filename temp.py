import numpy as np
import torch
temp = np.array([[1,2,3,4],
                 [2,3,4,5],
                 [4,5,6,7]])
print("temp:", temp)
temp_to_tensor = torch.Tensor(temp)
print("temp_to_tensor:", temp_to_tensor)
print("temp_to_tensor.data:", temp_to_tensor.data.max(1, keepdim=True)[1])# 只返回最大值的索引
print(temp_to_tensor.data.max(1, keepdim=True)[1].squeeze().shape)
temp = temp_to_tensor.data.max(1, keepdim=True)[1].squeeze()
print("temp:", temp)
temp_to_np = temp.numpy()
print("temp_to_np:", temp_to_np.shape)