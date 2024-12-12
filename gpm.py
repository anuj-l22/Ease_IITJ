import torch
import torch.nn as nn

class GPM:
    def __init__(self, device, max_components=20, epsilon=1e-5, ns=32, threshold=0.95):
        self.device = device
        self.max_components = max_components
        self.epsilon = epsilon
        self.ns = ns
        self.threshold = threshold  # fraction of variance to retain
        self.layer_params = {}
        self.param_initialized = False
        self.activations = {}
        self.hook_handles = []

    def get_layer_name(self, param_name):
        parts = param_name.split('.')
        return ".".join(parts[:-2])

    def initialize_params(self, model):
        self.layer_params = {}
        for name, p in model.named_parameters():
            if p.requires_grad and "adapter" in name.lower():
                layer_name = self.get_layer_name(name)
                if layer_name not in self.layer_params:
                    self.layer_params[layer_name] = {"param_info": [], "Q": None}
                size = p.numel()
                self.layer_params[layer_name]["param_info"].append((name, p.shape, size))
        self.param_initialized = True

    def flatten_layer_grads(self, model, layer_name):
        param_dict = dict(model.named_parameters())
        grads = []
        for (name, shape, size) in self.layer_params[layer_name]["param_info"]:
            p = param_dict[name]
            if p.grad is None:
                grads.append(torch.zeros(size, device=self.device))
            else:
                grads.append(p.grad.view(-1))
        if len(grads) == 0:
            return None
        return torch.cat(grads)

    def unflatten_layer_grads(self, model, layer_name, flat_grads):
        idx = 0
        param_dict = dict(model.named_parameters())
        for (name, shape, size) in self.layer_params[layer_name]["param_info"]:
            p = param_dict[name]
            g_slice = flat_grads[idx: idx+size].view(shape)
            p.grad.data.copy_(g_slice)
            idx += size

    def register_representation_hooks(self, model):
        self.remove_hooks()
        self.activations = {ln: [] for ln in self.layer_params.keys()}

        def hook_fn(layer_name):
            def fn(m, inp, out):
                if out.dim() > 2:
                    out_flat = out.view(out.size(0), -1)
                else:
                    out_flat = out
                self.activations[layer_name].append(out_flat.detach())
            return fn

        self.hook_handles = []
        for layer_name in self.layer_params.keys():
            module = model
            for part in layer_name.split('.'):
                module = getattr(module, part)
            handle = module.register_forward_hook(hook_fn(layer_name))
            self.hook_handles.append(handle)

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def collect_representations(self, model, loader, known_classes):
        model.eval()
        collected = 0
        with torch.no_grad():
            for _, (idx, inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - known_classes >= 0,
                    aux_targets - known_classes,
                    -1,
                )

                valid_mask = aux_targets >= 0
                if valid_mask.sum() == 0:
                    continue

                inputs = inputs[valid_mask]
                _ = model(inputs, test=False)

                collected += inputs.size(0)
                if collected >= self.ns:
                    break

        for layer_name in self.activations.keys():
            if len(self.activations[layer_name]) > 0:
                self.activations[layer_name] = torch.cat(self.activations[layer_name], dim=0)
            else:
                self.activations[layer_name] = None

    def update_memory(self, model, loader, criterion, known_classes):
        if not self.param_initialized:
            self.initialize_params(model)

        self.register_representation_hooks(model)
        self.collect_representations(model, loader, known_classes)
        self.remove_hooks()

        for layer_name in self.layer_params.keys():
            R = self.activations[layer_name]
            if R is None or R.size(0) == 0:
                continue

            R_t = R.t()  # [D, N]
            Q = self.layer_params[layer_name]["Q"]
            if Q is not None:
                proj = Q @ (Q.t() @ R_t)  # [D, N]
                R_t_hat = R_t - proj
            else:
                R_t_hat = R_t

            if R_t_hat.size(1) == 0:
                continue

            # Perform SVD on R_t_hat
            U, S, Vt = torch.linalg.svd(R_t_hat, full_matrices=False)
            # S: [min(D, N)] singular values

            # Compute total energy
            total_energy = (S**2).sum().item()
            energy_threshold = self.threshold * total_energy

            # Find minimal k that satisfies threshold
            cumulative = 0.0
            k = 0
            for i in range(S.size(0)):
                cumulative += S[i].item()**2
                if cumulative >= energy_threshold:
                    k = i + 1
                    break

            # Also cap k at max_components
            k = min(k, self.max_components)

            if k == 0:
                # If no direction is chosen, skip (should not usually happen if threshold < 1)
                continue

            U_sub = U[:, :k]  # top-k vectors

            if Q is None:
                Q_new = U_sub
            else:
                M = torch.cat([Q, U_sub], dim=1)
                Q_new, _ = torch.linalg.qr(M)
                if Q_new.size(1) > self.max_components:
                    Q_new = Q_new[:, :self.max_components]

            self.layer_params[layer_name]["Q"] = Q_new

        self.activations.clear()

    def project_gradients(self, model):
        for layer_name, layer_data in self.layer_params.items():
            Q = layer_data["Q"]
            if Q is None:
                continue
            flat_g = self.flatten_layer_grads(model, layer_name)
            if flat_g is None:
                continue
            g = flat_g.unsqueeze(1)
            proj = Q @ (Q.t() @ g)
            g_new = g - proj
            self.unflatten_layer_grads(model, layer_name, g_new.squeeze(1))
