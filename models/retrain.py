import os
import io
import base64
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import json
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义模型架构
class Net(nn.Module):
    def __init__(self, layer_sizes):
        super(Net, self).__init__()
        self.Wz = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            m = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            nn.init.xavier_normal_(m.weight, gain=1)
            nn.init.constant_(m.bias, 0.)
            self.Wz.append(m)

    def forward(self, x):
        H = torch.tanh(self.Wz[0](x))
        for linear in self.Wz[1:-1]:
            H = torch.tanh(linear(H))
        return self.Wz[-1](H)


class DeepONet(nn.Module):
    def __init__(self, b_dim, t_dim, layer_sizes_b, layer_sizes_t):
        super(DeepONet, self).__init__()
        self.branch = Net(layer_sizes_b)
        self.trunk = Net(layer_sizes_t)
        self.b = nn.Parameter(torch.zeros(4))

    def forward(self, x, l):
        x = self.branch(x)
        l = self.trunk(l)
        x_split = torch.chunk(x, chunks=4, dim=1)
        l_split = torch.chunk(l, chunks=4, dim=1)
        outputs = []
        for i, (x_part, l_part) in enumerate(zip(x_split, l_split)):
            res = torch.einsum("bi,bi->b", x_part, l_part) + self.b[i]
            outputs.append(res.unsqueeze(-1))
        return torch.cat(outputs, dim=1)


def retrain(data_file, epoch, gamma, batch_size):
    if not data_file or not os.path.exists(data_file):
        yield 'ERROR|||未提供有效的数据文件'
        return

    try:
        data_sol = np.load(data_file)
    except Exception as e:
        yield f'ERROR|||加载数据失败：{str(e)}'
        return

    # 获取文件名
    basename = os.path.basename(data_file)
    # 去掉文件扩展名
    filename_without_extension = os.path.splitext(basename)[0]
    # 找到 "Ma" 出现的位置
    ma_index = filename_without_extension.find('Ma')
    if ma_index != -1:
        # 提取从开头到 "Ma" 之前的部分
        input_str = filename_without_extension[:ma_index].strip('_')
    else:
        # 如果没有找到 "Ma"，则提取整个文件名（去掉扩展名）
        input_str = filename_without_extension

    # 解析输入变量
    selected_inputs = input_str.split('_') if input_str else []

    print(f"input_indices: {selected_inputs}")

    data_sol[:, :, 2] = data_sol[:, :, 2] / 16
    data_sol[:, :, 3] = data_sol[:, :, 3] / 4
    data_sol[:, :, 4] = data_sol[:, :, 4] / 1
    data_sol[:, :, 5] = data_sol[:, :, 5] / 50
    data_sol[:, :, 0:2] = data_sol[:, :, 0:2] * 10

    Ma_full = np.linspace(4., 7., 61)
    Ma_train = np.hstack((Ma_full[0:19], Ma_full[20:61]))
    Ma_train_sol = np.vstack((data_sol[0:19, :, :], data_sol[20:61, :, :]))
    Ma_test_sol = data_sol[19:20, :, :]

    N_ma = Ma_train_sol.shape[0]
    N_loc = Ma_train_sol.shape[1]
    N_rand = 5000
    N_rand1 = 1000
    N_rand2 = N_rand - N_rand1

    input_indices = [0]
    if 'rho' in selected_inputs: input_indices.append(2)
    if 'u' in selected_inputs: input_indices.append(3)
    if 'v' in selected_inputs: input_indices.append(4)
    if 'p' in selected_inputs: input_indices.append(5)

    u_train = np.zeros((N_ma * N_rand, len(input_indices)))
    loc_train = np.zeros((N_ma * N_rand, 2))
    y_train = np.zeros((N_ma * N_rand, 4))

    for i in range(N_ma):
        a = Ma_train_sol[i]
        b1 = a[a[:, 1] > 0.5 * (a[:, 0] - 0.4)]
        b2 = a[a[:, 1] < 0.5 * (a[:, 0] - 0.4)]
        N_c1 = np.random.choice(b1.shape[0], N_rand1, replace=False)
        N_c2 = np.random.choice(b2.shape[0], N_rand2, replace=False)
        for j in range(N_rand):
            k = i * N_rand + j
            u_train[k, 0] = Ma_train[i]
            if j < N_rand1:
                t = N_c1[j]
                u_train[k, 1:] = b1[t, input_indices[1:]]
                loc_train[k, :] = b1[t, :2]
                y_train[k, :] = b1[t, 2:6]
            else:
                t = N_c2[j - N_rand1]
                u_train[k, 1:] = b2[t, input_indices[1:]]
                loc_train[k, :] = b2[t, :2]
                y_train[k, :] = b2[t, 2:6]

    u_train = torch.tensor(u_train, dtype=torch.float32).to(device)
    loc_train = torch.tensor(loc_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

    b_size = [len(input_indices), 16, 64, 128, 128, 64, 32]
    t_size = [2, 32, 128, 256, 256, 128, 64, 32]
    learning_rate = 1e-3
    step_size = 10
    train_loader = DataLoader(TensorDataset(u_train, loc_train, y_train), batch_size=batch_size, shuffle=True)

    model = DeepONet(len(input_indices), 2, b_size, t_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_func = nn.MSELoss()

    yield "开始训练模型..."
    for epochs in range(epoch):
        model.train()
        total_loss = 0.0
        for u_batch, loc_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(u_batch, loc_batch)
            loss = loss_func(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        yield f"Epoch [{epochs + 1}/{epoch}], Loss: {total_loss / len(train_loader):.6f}"

    yield "训练完成，正在测试模型..."

    # 测试模型
    Ma_test = Ma_full[19:20].reshape(1, 1)
    N_ma = Ma_test_sol.shape[0]
    N_loc = Ma_test_sol.shape[1]
    u_test = np.zeros((N_ma * N_loc, len(input_indices)))  # 同样，第二维由模型输入决定：马赫数+输入量的个数（例如：输入ρ, u, v, p，则第二维是1+4=5）
    loc_test = np.zeros((N_ma * N_loc, 2))
    y_test = np.zeros((N_ma * N_loc, 4))

    for i in range(N_ma):
        for j in range(N_loc):
            k = i * N_loc + j
            u_test[k, 0] = Ma_test[i]
            u_test[k, 1:] = Ma_test_sol[i, j, input_indices[1:]]  # 同样，数据的特征有六个，依次是x, y, ρ, u, v, p，根据模型输入对数组b1进行调整
            loc_test[k, :] = Ma_test_sol[i, j, :2]
            y_test[k, :] = Ma_test_sol[i, j, 2:]

    out_test = model(
        torch.Tensor(u_test).to(device),
        torch.Tensor(loc_test).to(device)
    ).detach().cpu().numpy()

    # 计算误差
    errors = []
    for i in range(4):
        l2_rel = np.linalg.norm(out_test[:, i] - y_test[:, i]) / np.linalg.norm(y_test[:, i])
        mse_test = np.mean((out_test[:, i] - y_test[:, i]) ** 2)
        errors.append((l2_rel, mse_test))

    # cmp
    xpoints = Ma_test_sol[0, :, 0]
    ypoints = Ma_test_sol[0, :, 1]

    for i in range(4):
        fig = plt.figure(figsize=(18, 5))
        plt.suptitle(f"Feature {i + 1}")

        # 绘制预测值
        plt.subplot(1, 3, 1)
        plt.scatter(xpoints, ypoints, c=out_test[0:xpoints.shape[0], i], s=0.5, cmap='OrRd', label='pred density')
        plt.colorbar()
        plt.legend()
        plt.title("Predicted Density")

        # 绘制真实值
        plt.subplot(1, 3, 2)
        plt.scatter(xpoints, ypoints, c=y_test[0:xpoints.shape[0], i], s=0.5, cmap='OrRd', label='true density')
        plt.colorbar()
        plt.legend()
        plt.title("True Density")

        # 绘制误差
        plt.subplot(1, 3, 3)
        error = np.abs(y_test[0:xpoints.shape[0], i] - out_test[0:xpoints.shape[0], i])
        plt.scatter(xpoints, ypoints, c=error, s=0.5, cmap='OrRd', label='error', vmin=0, vmax=0.05)
        plt.colorbar()
        plt.legend()
        plt.title("Error")

        # 调整子图间距
        plt.subplots_adjust(wspace=0.3, hspace=0.3)

        # Save the figure to the current directory
        plt.savefig(f"./static/result/feature_{i + 1}.png")
        # Show the figure

    model_bytes = io.BytesIO()
    torch.save(model.state_dict(), model_bytes)
    model_bytes.seek(0)
    model_base64 = base64.b64encode(model_bytes.getvalue()).decode('utf-8')

    final_result = {
        'status': '模型重新训练并测试成功',
        'errors': errors,
        'model_base64': model_base64,
    }

    yield "DONE|||" + json.dumps(final_result, ensure_ascii=False)