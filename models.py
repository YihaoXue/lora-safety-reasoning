from peft import get_peft_model
import torch
from tqdm import tqdm

class OrthogonalRegularizationMixin:
    def inject_orthogonal_regularization(self, beta, k, target_modules=None, orthogonal_mode="column", weighted=True, normalize=False):

        assert orthogonal_mode in ["column", "row", "both"]

        self._orth_beta = beta
        self._orth_mode = orthogonal_mode
        self._orth_targets = target_modules or ["gate_proj", "up_proj", "down_proj"]

        modules = [
            (name, module)
            for name, module in self.named_modules()
            if any(t in name for t in self._orth_targets)
        ]

        # Store initial weights for regularization
        for name, module in tqdm(modules, desc=f"Registering orthogonal buffers; mode: {self._orth_mode}"):
            if hasattr(module, 'weight'):
                device = module.weight.device
                # low rank approximation
                with torch.no_grad():
                    weight = module.weight.detach().clone().to(device).to(torch.float32)
                    k = min(k, min(weight.shape))  # ensure k is valid
                    U, S, V = torch.svd_lowrank(weight, q=k)

                    if orthogonal_mode == "column": 
                        if weighted:
                            span = (torch.diag(S) @ U.T).to(torch.bfloat16) 
                            if normalize:
                                span = span / torch.norm(span, p='fro')
                        else:
                            span = (U.T).contiguous().to(torch.bfloat16)
                        module.register_buffer("init_span", span)
                    elif orthogonal_mode == "row":
                        if weighted: 
                            span = (V @ torch.diag(S)).to(torch.bfloat16)
                            if normalize:
                                span = span / torch.norm(span, p='fro') 
                        else:
                            span = V.contiguous().to(torch.bfloat16)
                        module.register_buffer("init_span", span)
                    else: # both
                        if weighted: 
                            span_c = (torch.diag(S) @ U.T).to(torch.bfloat16) 
                            span_r = (V @ torch.diag(S)).to(torch.bfloat16) 
                            if normalize:
                                span_c = span_c / torch.norm(span_c, p='fro')
                                span_r = span_r / torch.norm(span_r, p='fro')
                        else:
                            span_c = (U.T).contiguous().to(torch.bfloat16)
                            span_r = V.contiguous().to(torch.bfloat16)
                        module.register_buffer("init_span_c", span_c)
                        module.register_buffer("init_span_r", span_r)

        # Patch forward
        original_forward = self.forward

        def patched_forward(self, **kwargs):
            outputs = original_forward(**kwargs)
            loss = outputs.loss
            reg_loss = 0.0

            #device = kwargs["input_ids"].device #if hasattr(outputs, "logits") else self.base_model.model.device
            device = loss.device
            for name, module in self.named_modules():
                if any(t in name for t in self._orth_targets):
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        delta_w = module.lora_B[self.active_adapter].weight.to(device) @ module.lora_A[self.active_adapter].weight.to(device)
                        
                        if self._orth_mode == "both":
                            init_span_c = getattr(module, "init_span_c", None)
                            if init_span_c is None:
                                raise RuntimeError(f"Missing init_span_c for module {name}")
                            init_span_c = init_span_c.to(device)

                            init_span_r = getattr(module, "init_span_r", None)
                            if init_span_r is None:
                                raise RuntimeError(f"Missing init_span_r for module {name}")
                            init_span_r = init_span_r.to(device)
                        else:
                            init_span = getattr(module, "init_span", None)
                            if init_span is None:
                                raise RuntimeError(f"Missing init_span for module {name}")
                            init_span = init_span.to(device)

                        #print(init_span.shape)

                        if self._orth_mode == "column":
                            penalty = torch.norm(init_span @ delta_w, p='fro') ** 2
                        elif self._orth_mode == "row":
                            penalty = torch.norm(delta_w @ init_span, p='fro') ** 2
                        else: # both
                            penalty = torch.norm(init_span_c @ delta_w @ init_span_r, p='fro') ** 2
                        
                        if normalize:
                            penalty = penalty / ( torch.norm(delta_w, p='fro') ** 2 + 1e-8 )

                        reg_loss += penalty

            loss = (loss + self._orth_beta * reg_loss).to(device)
            return type(outputs)(loss=loss, **{k: v for k, v in outputs.items() if k != 'loss'})

        self.forward = patched_forward.__get__(self, self.__class__)


def get_orthogonal_peft_model(base_model, peft_config, beta, k, target_modules=None, orthogonal_mode="column", weighted=True, normalize=False):
    model = get_peft_model(base_model, peft_config)
    # Dynamically add mixin
    model.__class__ = type("OrthogonalPeftModel", (model.__class__, OrthogonalRegularizationMixin), {})
    model.inject_orthogonal_regularization(beta=beta, k=k, normalize=normalize, weighted=weighted, target_modules=target_modules, orthogonal_mode=orthogonal_mode)
    return model
