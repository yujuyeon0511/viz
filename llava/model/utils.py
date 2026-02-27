from transformers import AutoConfig
import torch


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, T_I_V_A.txt, D) tensor.
    v: (N, T_I_V_A.txt, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return ((u - v) ** 2).sum(dim=-1) ** 0.5


def cal_acc(self, output, label):
    chosen_tokens = torch.max(output.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
    target = label[:, 2:]
    gen_acc = (chosen_tokens.reshape(-1) == target.reshape(-1)).to(torch.long)  # [B*S]
    valid_mask = (label != -100).reshape(-1)
    valid_tokens = gen_acc & valid_mask  # [B*S]
    gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
    return gen_acc
def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
