import torch
from torch_geometric.utils import softmax
from chemprop.nn import Aggregation


class Set2Set(Aggregation):
    def __init__(
        self,
        in_channels: int,
        processing_steps: int = 6,
        n_layers: int = 3,
        dim: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(dim, **kwargs)
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.lstm = torch.nn.LSTM(
            self.out_channels, self.in_channels, num_layers=n_layers, **kwargs
        )
        self.reset_parameters()
        
        self.hparams["in_channels"] = in_channels
        self.hparams["processing_steps"] = processing_steps
        self.hparams["n_layers"] = n_layers

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, H: torch.Tensor, batch: torch.Tensor):
        dim_size = batch.max().int() + 1
        index_torch = batch.unsqueeze(1).repeat(1, H.shape[1])
        h = (
            H.new_zeros((self.lstm.num_layers, dim_size, H.size(-1)), dtype=H.dtype),
            H.new_zeros((self.lstm.num_layers, dim_size, H.size(-1)), dtype=H.dtype),
        )
        q_star = H.new_zeros(dim_size, self.out_channels, dtype=H.dtype)

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(dim_size, self.in_channels)
            e = (H * q[batch]).sum(dim=-1, keepdim=True, dtype=H.dtype)
            a = softmax(e, batch, None, dim=self.dim).to(H.dtype)
            r = torch.zeros(
                dim_size, H.shape[1], dtype=H.dtype, device=H.device
            ).scatter_reduce_(
                self.dim, index_torch, a * H, reduce="sum", include_self=False
            )
            q_star = torch.cat([q, r], dim=-1)

        return q_star

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"
