from typing import Literal
import torch

NEG_INF = torch.finfo(torch.float32).min


def ctc_loss(
    log_probs: torch.FloatTensor,
    targets: torch.LongTensor,
    input_lengths: torch.LongTensor,
    target_lengths: torch.LongTensor,
    blank: int = 0,
    reduction: Literal["none", "sum", "mean"] = "none",
):
    assert reduction in ["none", "sum", "mean"], (
        "Select reductions from 'none', 'sum' and 'mean' !"
    )
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

    log_alpha = torch.full(
        (seq_len, batch_size, 2 + _t_a_r_g_e_t_s_.shape[-1]), NEG_INF, device=device
    )

    log_alpha[0, :, 2] = log_probs[0, :, blank]
    log_alpha[0, :, 3] = log_probs[0, B, _t_a_r_g_e_t_s_[:, 1]]

    for t in range(1, T):
        ### Compute all possible ways to reach some position s in log_alpha[t, :, s] ###
        ### - Stay at s [log_alpha[t-1, :, s]]
        ### - move from s-1 [log_alpha[t-1, :, s-1]]
        ### - move from s-2 [log_alpha[t-1, :, s-2]] (if allowed by diff labels)
        ### sum these probabilities since any of these transitions are valid!
        ### We compute this in parallel though for all s, instead of one at a time

        ### Probabilities for the Current Timestep ###
        log_probs_t = log_probs_[t]

        ### Probability of staying from the previous timestep ###
        log_alpha_t_prev_stay = log_alpha[t - 1, :, 2:]

        ### Probability of transitioning form the previous timestep ###
        log_alpha_t_prev_next = log_alpha[t - 1, :, 1:-1]

        ### Mask identifying valid transitions (Cant transition from A -> A without a blank token in between) ###
        log_alpha_two_step_transition = torch.where(
            diff_labels, input=log_alpha[t - 1, :, :-2], other=NEG_INF
        )

        prob = torch.logsumexp(
            torch.stack(
                [
                    log_alpha_t_prev_next,
                    log_alpha_t_prev_stay,
                    log_alpha_two_step_transition,
                ]
            ),
            dim=0,
        )

        ### Add our probs of this transition to the previous t probs (log turns multiplication to sum)
        log_alpha[t, :, 2:] = log_probs_t + prob

    ### Only grab till the T of each samples length indicated in input_lengths (anything after would just be padding)
    final_log_alpha = log_alpha[input_lengths - 1, B]

    ### Now a key idea! If our labels are: [28, 1, 17, 21]
    ### And then we added in blank tokens in between to create: [0, 28, 0, 1, 0, 17, 0, 21, 0, 28] (_t_a_r_g_e_t_s)
    ### Then our final label can either be 21 or 0 (remeber, the 28 we concatenated at the end was just filler we never touch it)
    ending_on_label_idx = 2 + target_lengths * 2 - 1
    ending_on_blank_idx = 2 + target_lengths * 2
    indexes_to_grab = torch.stack([ending_on_label_idx, ending_on_blank_idx], dim=-1)
    label_or_blank_ending_log_alphas = final_log_alpha.gather(
        dim=-1, index=indexes_to_grab
    )

    loss = -torch.logsumexp(label_or_blank_ending_log_alphas, dim=-1)

    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    return loss


if __name__ == "__main__":
    print(NEG_INF)
    T, B, C = 128, 256, 32
    dt = 50
    blank = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 1
    atol = 1e-3

    logits = torch.randn(T, B, C, device=device, requires_grad=True)
    targets = torch.randint(blank + 1, C, (B, dt), dtype=torch.long, device=device)
    input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    target_lengths = torch.full((B,), dt, dtype=torch.long, device=device)
    log_probs = logits.log_softmax(dim=-1)
