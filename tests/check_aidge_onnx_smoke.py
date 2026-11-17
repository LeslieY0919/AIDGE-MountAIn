import torch
import torch.nn as nn
import aidge_core
import aidge_onnx

from mountain import paths

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.net(x)

model = TinyNet().eval()
dummy_input = torch.randn(1, 3, 64, 64)
torch.onnx.export(model, dummy_input, str(paths.artifact("tiny_test.onnx")), input_names=["input"], output_names=["output"], opset_version=13)

aidge_model = aidge_onnx.load_onnx(str(paths.artifact("tiny_test.onnx")))
print("Graphe charge dans Aidge :", len(aidge_model.get_nodes()), "noeuds")
