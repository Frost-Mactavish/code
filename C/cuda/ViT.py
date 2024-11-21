import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

device = torch.device("cuda")
# device = torch.device("cpu")


model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)

# Warm-up
for _ in range(10):
    dummy_input = torch.randn(8, 3, 224, 224).to(device)
    model(dummy_input)

for _ in range(10):
    input = torch.rand(96, 3, 224, 224).to(device)
    model(input)

'''
对PyTorch实现中,VisionTransformer类的类方法forward进行改动
通过PyTorch内置Profiler类监视 x = self.encoder(x) 运行时间
torchvision > models > vision_transformer.py > VisionTransformer

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # 嵌入torch.profiler类,监视encoder块的运行时间
        from torch.profiler import profile, record_function, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("Encoder Runtime"):
                x = self.encoder(x)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
'''