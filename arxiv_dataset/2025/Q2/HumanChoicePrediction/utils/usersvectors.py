import torch


class UsersVectors:
    def __init__(self, user_dim, n_layers):
        self.user_dim = user_dim
        self.n_layers=n_layers
        self.users = {}
        self.init_user = torch.randn(self.n_layers, self.user_dim, dtype=torch.double, requires_grad=True)

    def get_init_vector(self):
        return self.init_user

    def delete_all_users(self):
        self.users = {}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.users[key] if key in self.users.keys() else self.get_init_vector()
        else:
            return torch.stack([self.users[k] if k in self.users.keys() else self.get_init_vector() for k in key])

    def __setitem__(self, key, new_value):
        if isinstance(key, int):
            self.users[key] = new_value
        elif len(key) == 1:
            self.users[key.item()] = new_value
        else:
            assert len(key) == len(new_value)
            for k, value in zip(key, new_value):
                self.users[k] = value

    def __len__(self):
        return len(self.users)
