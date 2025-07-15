import os
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {}

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

def load_models():
    global model_paths  # 声明为全局变量
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    model_paths = {
        "0.pth": [""],  # 修改为没有任何输入变量
        "1.pth": ["rho"],
        "2.pth": ["u"],
        "3.pth": ["v"],
        "4.pth": ["p"],
        "5.pth": ["rho", "u"],
        "6.pth": ["rho", "v"],
        "7.pth": ["rho", "p"],
        "8.pth": ["u", "v"],
        "9.pth": ["u", "p"],
        "10.pth": ["v", "p"],
        "11.pth": ["rho", "u", "v"],
        "12.pth": ["rho", "u", "p"],
        "13.pth": ["rho", "v", "p"],
        "14.pth": ["u", "v", "p"],
        "15.pth": ["rho", "u", "v", "p"]
    }

    for path, inputs in model_paths.items():
        model_path = os.path.join(UPLOAD_FOLDER, path)
        if os.path.exists(model_path):
            if inputs == [""]:
                b_size = [1, 16, 64, 128, 128, 64, 32]  # 仅使用 Ma，输入维度为 1
            else:
                b_size = [len(inputs) + 1, 16, 64, 128, 128, 64, 32]  # 其他模型

            t_size = [2, 32, 128, 256, 256, 128, 64, 32]
            model = DeepONet(len(inputs) + 1, 2, b_size, t_size).to(device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device), strict=False)
            model.eval()

            if inputs == [""]:
                models["Ma_only"] = model
            else:
                models[frozenset(inputs)] = model
        else:
            print(f"警告：模型文件 {path} 未找到，跳过加载。")

# 预测逻辑
def predict(data):
    global model_paths  # 确保可以访问全局的 model_paths

    Ma = data.get('Ma', None)
    x = data.get('x', None)
    y = data.get('y', None)
    rho = data.get('rho', None)
    u = data.get('u', None)
    v = data.get('v', None)
    p = data.get('p', None)

    if Ma is None or x is None or y is None:
        missing_fields = []
        if Ma is None:
            missing_fields.append('马赫数 (Ma)')
        if x is None:
            missing_fields.append('X 坐标 (x)')
        if y is None:
            missing_fields.append('Y 坐标 (y)')
        return {'error': f'未输入：{", ".join(missing_fields)}'}

    try:
        Ma = float(Ma)
        x = float(x)
        y = float(y)
    except ValueError:
        return {'error': '输入的马赫数、X 坐标或 Y 坐标格式不正确'}

    if rho is None and u is None and v is None and p is None:
        model_key = "Ma_only"
    else:
        input_vars = set()
        if rho is not None:
            input_vars.add("rho")
        if u is not None:
            input_vars.add("u")
        if v is not None:
            input_vars.add("v")
        if p is not None:
            input_vars.add("p")
        model_key = frozenset(input_vars)

    # 反向查找模型文件名
    model_path = None
    for path, inputs in model_paths.items():
        if inputs == [""] and model_key == "Ma_only":
            model_path = path
            break
        elif frozenset(inputs) == model_key:
            model_path = path
            break

    if model_key not in models:
        return {'error': f'未找到对应的模型：{model_key}，路径: {model_path}'}

    # 匹配对应的文件夹
    folder_map = {f"{i}.pth": f"./static/result/{i}" for i in range(16)}  # 0-15.pth -> /result/0 - /result/15
    folder_path = folder_map.get(model_path, None)

    # 获取该文件夹下的所有图片
    image_paths = []
    if folder_path and os.path.exists(folder_path):  # 先检查文件夹是否存在
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                image_paths.append(os.path.join(folder_path, file))
    else:
        print(f"警告：文件夹 {folder_path} 不存在，或未找到图片")

    inputs = [Ma]
    if rho is not None:
        inputs.append(float(rho))
    if u is not None:
        inputs.append(float(u))
    if v is not None:
        inputs.append(float(v))
    if p is not None:
        inputs.append(float(p))

    branch_input = torch.tensor([inputs], dtype=torch.float32).to(device)
    trunk_input = torch.tensor([[x, y]], dtype=torch.float32).to(device)

    model = models[model_key]
    with torch.no_grad():
        output = model(branch_input, trunk_input).detach().cpu().numpy()[0]
    return {
        'model_path': model_path,  # 返回模型路径
        'image_paths': image_paths,  # 仅返回图片路径
        'density': float(output[0]),
        'u': float(output[1]),
        'v': float(output[2]),
        'pressure': float(output[3])
    }
