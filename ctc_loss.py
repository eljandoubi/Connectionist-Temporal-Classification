import torch

NEG_INF = torch.finfo(torch.float32).min


def ctc_loss(
    log_probs: torch.FloatTensor,
    targets: torch.LongTensor,
    input_lengths: torch.LongTensor,
    target_lengths,
    blank: int = 0,
    reduction="none",
):
    seq_len, batch_size, num_classes = log_probs.shape
    device = log_probs.device
    B = torch.arange(batch_size)

    _t_a_r_g_e_t_s_ = torch.cat(
        [targets, torch.zeros(batch_size, 1, device=device, dtype=torch.long)], dim=-1
    )

    _t_a_r_g_e_t_s_ = torch.stack(
        [torch.full_like(_t_a_r_g_e_t_s_, blank), _t_a_r_g_e_t_s_], dim=-1
    ).flatten(start_dim=-2)

    diff_labels = torch.cat(
        [
            torch.tensor([[False, False]], device=device).expand(batch_size, -1),
            _t_a_r_g_e_t_s_[:, 2:] != _t_a_r_g_e_t_s_[:, :-2],
        ],
        dim=-1,
    )

    log_probs_ = log_probs.gather(dim=-1, index=_t_a_r_g_e_t_s_.expand(seq_len, -1, -1))


if __name__ == "__main__":
    print(NEG_INF)
    T, B, C = 128, 256, 32
    t = 50
    blank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 1
    atol = 1e-3

    logits = torch.randn(T, B, C, device=device, requires_grad=True)
    targets = torch.randint(blank + 1, C, (B, t), dtype=torch.long, device=device)
    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    target_lengths = torch.full((B,), t, dtype=torch.long, device=device)
    log_probs = logits.log_softmax(dim=-1)
