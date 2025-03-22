import kan


class MultiKAN(kan.MultKAN):
    def __init__(self, in_dim, out_dim, **kwargs):
        width = [in_dim, out_dim]
        super(MultiKAN, self).__init__(width=width, **kwargs)

    def forward(self, x, singularity_avoiding=False, y_th=10):
        return super().forward(x, singularity_avoiding, y_th)

    def regularization_loss(self):
        pass
