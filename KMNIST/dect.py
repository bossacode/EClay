import torch
import torch.nn as nn
# import geotorch


def compute_ecc(nh, index, lin, out):
    ecc = torch.nn.functional.sigmoid(200 * torch.sub(lin, nh))
    return torch.index_add(out, 1, index, ecc).movedim(0, 1)


def compute_ect_points(data, v, lin, out):
    nh = data.x @ v
    return compute_ecc(nh, data.batch, lin, out)


def compute_ect_edges(data, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    return compute_ecc(nh, data.batch, lin, out) - compute_ecc(eh, data.batch[data.edge_index[0]], lin, out)


def compute_ect_faces(data, v, lin, out):
    nh = data.x @ v
    eh, _ = nh[data.edge_index].max(dim=0)
    fh, _ = nh[data.face].max(dim=0)
    return (
        compute_ecc(nh, data.batch, lin, out)
        - compute_ecc(eh, data.batch[data.edge_index[0]], lin, out)
        + compute_ecc(fh, data.batch[data.face[0]], lin, out)
        )


class EctLayer(nn.Module):
    def __init__(self, bump_steps, num_features, num_thetas, R, ect_type, device, fixed=False):
        super().__init__()
        self.bump_steps = bump_steps
        self.num_thetas = num_thetas
        self.device = device
        self.fixed = fixed
        self.lin = (
            torch.linspace(-R, R, bump_steps)
            .view(-1, 1, 1)
            .to(device)
        )
        if self.fixed:
            self.v = torch.vstack(
                [
                    torch.sin(torch.linspace(0, 2 * torch.pi, num_thetas)),
                    torch.cos(torch.linspace(0, 2 * torch.pi, num_thetas)),
                ]
            ).to(device)
        else:
            self.v = (torch.rand(size=(num_features, num_thetas)) - 0.5).T.to(device)
            self.v /= self.v.pow(2).sum(axis=1).sqrt().unsqueeze(1)
            self.v = nn.Parameter(self.v.T)

        if ect_type == "points":
            self.compute_ect = compute_ect_points
        elif ect_type == "edges":
            self.compute_ect = compute_ect_edges
        elif ect_type == "faces":
            self.compute_ect = compute_ect_faces

    # def postinit(self):
    #     if not self.fixed:
    #         geotorch.constraints.sphere(self, "v")

    def forward(self, data):
        out = torch.zeros(
            size=(
                self.bump_steps,
                data.batch.max().item() + 1,
                self.num_thetas,
            ),
            device=self.device,
        )
        return self.compute_ect(data, self.v, self.lin, out)