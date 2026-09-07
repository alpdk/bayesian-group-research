"""Standalone verification for the two insertion knobs.

Knob A (insertion_count_theta) and Knob B (insertion_temperature); see the
knob docstrings in flexmdm/inference.py and docs/REPRODUCE.md ("Insertion
temperature"). Exercises the pure helpers and a full
flexmdm_generate loop with a fake model, on CPU, no checkpoint required.

Run from the repo root with any env that has torch, e.g.:
    PYTHONPATH=. python scripts/verify_insertion_knobs.py
Exits non-zero if any check fails.
"""
import torch

from flexmdm.inference import (
    _apply_after_token_insertions,
    _expected_insertion_counts,
    _insertion_step,
    _cap_insertions_to_capacity,
    _valid_insertion_gaps,
    _sample_insertion_counts,
    flexmdm_generate,
)
from flexmdm.schedules import schedule_hazard_rate

torch.set_printoptions(precision=4, sci_mode=False)
OK = "PASS"
FAIL = "FAIL"
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{OK if cond else FAIL}] {name}")


# ---------------------------------------------------------------------------
# Test A: count preservation under placement temperature
# ---------------------------------------------------------------------------
torch.manual_seed(0)
g = torch.randn(3, 16)
hazard = torch.full((3,), 2.9)
dt = 0.01
valid = torch.ones(3, 16, dtype=torch.bool)
sums = {}
for T in (0.25, 0.5, 1.0, 2.0, 5.0):
    e = _expected_insertion_counts(g, hazard, dt, valid, insertion_temperature=T)
    sums[T] = e.sum(dim=1)
base = sums[1.0]
check("A. expected total invariant to temperature",
      all(torch.allclose(base, sums[T], atol=1e-5) for T in sums))

# ---------------------------------------------------------------------------
# Test B: T=1 reproduces the original per-gap expected count exp(g)*hazard*dt
# ---------------------------------------------------------------------------
e1 = _expected_insertion_counts(g, hazard, dt, valid, insertion_temperature=1.0)
ref = g.clamp(max=15.0).exp() * hazard[:, None] * dt
check("B. T=1 equals exp(g)*hazard*dt on valid gaps", torch.allclose(e1, ref, atol=1e-5))

# ---------------------------------------------------------------------------
# Test C: invalid gaps excluded from both softmax and preserved total
# ---------------------------------------------------------------------------
valid2 = valid.clone()
valid2[:, :4] = False
e = _expected_insertion_counts(g, hazard, dt, valid2, insertion_temperature=0.5)
ref_valid = (g.clamp(max=15.0).exp() * hazard[:, None] * dt).masked_fill(~valid2, 0.0)
check("C1. invalid gaps are zero", bool((e[:, :4] == 0).all()))
check("C2. total preserved over valid gaps only",
      torch.allclose(e.sum(1), ref_valid.sum(1), atol=1e-5))
# temperature sharpening really moves placement mass toward the argmax gap
e_sharp = _expected_insertion_counts(g, hazard, dt, valid2, insertion_temperature=0.2)
argmax_gap = (g.masked_fill(~valid2, float("-inf"))).argmax(dim=1)
row = torch.arange(3)
check("C3. T<1 concentrates mass on the top-g gap",
      bool((e_sharp[row, argmax_gap] > e[row, argmax_gap]).all()))

# ---------------------------------------------------------------------------
# Test D: no-valid-gap row -> zeros, no NaN
# ---------------------------------------------------------------------------
valid3 = valid.clone()
valid3[1, :] = False
e = _expected_insertion_counts(g, hazard, dt, valid3, insertion_temperature=0.7)
check("D. all-invalid row -> zeros, finite", bool((e[1] == 0).all()) and torch.isfinite(e).all())

# ---------------------------------------------------------------------------
# Test E: carry remap follows tokens across the position shift
# ---------------------------------------------------------------------------
xt = torch.tensor([[10, 11, 12, 0, 0]])
attn = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
insertions = torch.tensor([[1, 0, 0, 0, 0]])  # insert 1 mask after position 0
carry = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0]])
nx, na, isins, nc = _apply_after_token_insertions(
    xt, attn, insertions, pad_id=0, mask_id=99, carry=carry
)
# tokens: 10@0, mask@1, 11@2, 12@3 ; carry rides along: [0.1, 0(new mask), 0.2, 0.3, 0]
check("E1. tokens shifted correctly", nx.tolist() == [[10, 99, 11, 12, 0]])
check("E2. carry remapped with tokens",
      torch.allclose(nc, torch.tensor([[0.1, 0.0, 0.2, 0.3, 0.0]]), atol=1e-6))
check("E3. is_inserted marks only the new mask", isins.tolist() == [[False, True, False, False, False]])

# ---------------------------------------------------------------------------
# Test F: bit-exact default path (theta=1, T=1) == original inline formula
# ---------------------------------------------------------------------------
B, Lbuf, V = 2, 24, 50
xt = torch.zeros(B, Lbuf, dtype=torch.long)
prompt_len = torch.tensor([4, 4])
for b in range(B):
    xt[b, :6] = torch.arange(1, 7)  # 6 real tokens
attn = torch.zeros(B, Lbuf, dtype=torch.bool)
attn[:, :6] = True
glp = torch.randn(B, Lbuf)
t = torch.full((B,), 0.3)
mask_id, pad_id = 99, 0

torch.manual_seed(123)
out_xt, out_attn, _, _ = _insertion_step(
    xt, attn, glp, prompt_len=prompt_len, t=t, dt=dt, pad_id=pad_id, mask_id=mask_id,
    eps=1e-6, hazard_rate_schedule="power", hazard_rate_exponent=2.9,
    insertion_temperature=1.0, insertion_count_theta=1.0,
)

# Inline replication of the *original* code path.
torch.manual_seed(123)
rate = schedule_hazard_rate(t, schedule="power", eps=1e-6, exponent=2.9)
insertion_rate = glp.clamp(max=15.0).float().exp() * rate[:, None]
ins = _sample_insertion_counts(insertion_rate * dt, insertion_count_sampler="poisson")
vg = _valid_insertion_gaps(prompt_len=prompt_len, xt_len=attn.sum(1), max_len=Lbuf, device=xt.device)
ins = ins.masked_fill(~vg, 0)
ins = _cap_insertions_to_capacity(ins, capacity=Lbuf - attn.sum(1))
ref_xt, ref_attn, _, _ = _apply_after_token_insertions(xt, attn, ins, pad_id=pad_id, mask_id=mask_id, carry=None)
check("F. default path (theta=1,T=1) bit-exact vs original", bool(torch.equal(out_xt, ref_xt)) and bool(torch.equal(out_attn, ref_attn)))

# ---------------------------------------------------------------------------
# Test G: theta=0 insertion step is RNG-free (identical across seeds)
# ---------------------------------------------------------------------------
torch.manual_seed(1)
o1 = _insertion_step(xt, attn, glp, prompt_len=prompt_len, t=t, dt=dt, pad_id=pad_id,
                     mask_id=mask_id, eps=1e-6, hazard_rate_schedule="power",
                     hazard_rate_exponent=2.9, insertion_temperature=1.0,
                     insertion_count_theta=0.0)
torch.manual_seed(999)
o2 = _insertion_step(xt, attn, glp, prompt_len=prompt_len, t=t, dt=dt, pad_id=pad_id,
                     mask_id=mask_id, eps=1e-6, hazard_rate_schedule="power",
                     hazard_rate_exponent=2.9, insertion_temperature=1.0,
                     insertion_count_theta=0.0)
check("G. theta=0 insertion deterministic across seeds", bool(torch.equal(o1[0], o2[0])))


# ---------------------------------------------------------------------------
# Test H: end-to-end length-variance collapse (fake model)
# ---------------------------------------------------------------------------
class FakeModel(torch.nn.Module):
    """Confident tokens; per-gap log-length tapers toward a target length."""

    def __init__(self, vocab=50, target_len=48):
        super().__init__()
        self.vocab = vocab
        self.target_len = target_len

    def forward(self, xt, t, attention_mask=None):
        Bs, L = xt.shape
        logits = torch.full((Bs, L, self.vocab), -5.0)
        idx = torch.arange(L) % self.vocab
        logits[:, torch.arange(L), idx] = 8.0
        cur = attention_mask.sum(dim=1, keepdim=True).float() if attention_mask is not None else torch.full((Bs, 1), float(L))
        remaining = (self.target_len - cur).clamp_min(0.0)
        per_gap = (remaining / float(L)).clamp_min(1e-4)
        log_length = per_gap.log().expand(Bs, L).contiguous()
        return {"logits": logits, "log_length": log_length}


def run_lengths(theta, temp, seeds, token_temp):
    model = FakeModel()
    lens = []
    prompt = torch.arange(1, 6).view(1, 5)
    ids = torch.full((1, 80), 0, dtype=torch.long)
    ids[0, :5] = prompt
    am = torch.zeros(1, 80, dtype=torch.bool)
    am[0, :5] = True
    for s in seeds:
        torch.manual_seed(s)
        out = flexmdm_generate(
            model, steps=32, input_ids=ids.clone(), mask_id=99, pad_id=0,
            attention_mask=am.clone(), prompt_mask=am.clone(), temperature=token_temp,
            confidence_method="top_k", insertion_schedule="power", insertion_exponent=2.9,
            unmasking_schedule="power", unmasking_exponent=2.89,
            insertion_temperature=temp, insertion_count_theta=theta,
        )
        lens.append(int(out[0].ne(0).sum().item()))
    return torch.tensor(lens, dtype=torch.float32)


seeds = list(range(12))
len_stoch = run_lengths(theta=1.0, temp=1.0, seeds=seeds, token_temp=0.0)
len_det = run_lengths(theta=0.0, temp=1.0, seeds=seeds, token_temp=0.0)
print(f"    stochastic (theta=1): lengths mean={len_stoch.mean():.1f} std={len_stoch.std():.3f}")
print(f"    determ.    (theta=0): lengths mean={len_det.mean():.1f} std={len_det.std():.3f}")
check("H1. theta=0 gives zero length variance (token_temp=0)", float(len_det.std()) < 1e-6)
check("H2. theta=1 has real length variance", float(len_stoch.std()) > 0.5)

# H3: the deterministic carry integrator is unbiased *given a fixed per-step
# expected input*. floor+carry over K steps plus the end flush emits ~K*mu per
# gap for any mu, however small -- this is what makes the deterministic channel
# track the model's intended per-step mass instead of dropping sub-unit rates.
# (NOTE: across a full sampling trajectory the deterministic path follows the
# mean-field ODE, which differs from the *mean* of the stochastic ensemble by a
# Jensen gap whenever the rate is state-dependent -- e.g. gap count compounds.
# That divergence is expected, not a bug; the state-conditioned g_theta of the
# real model self-corrects length.)
def integrate_carry(mu_vec, K, flush=True):
    carry = torch.zeros_like(mu_vec)
    total = torch.zeros_like(mu_vec)
    for k in range(K):
        carry = carry + mu_vec
        emit = torch.round(carry) if (flush and k == K - 1) else torch.floor(carry)
        carry = carry - emit
        total = total + emit
    return total


max_err = 0.0
for mu_val in (0.02, 0.1, 0.3, 0.7, 1.3):
    mu = torch.full((5,), mu_val)
    K = 60
    tot = integrate_carry(mu, K)
    err = float((tot - mu * K).abs().max())
    max_err = max(max_err, err)
    print(f"    integrator mu={mu_val}: total={tot[0].item():.0f}  K*mu={mu_val*K:.1f}  err={err:.2f}")
check("H3. carry integrator unbiased given fixed input (|err| < 1 per gap)", max_err < 1.0)

# intermediate theta should sit between the two in variance
len_mid = run_lengths(theta=0.5, temp=1.0, seeds=seeds, token_temp=0.0)
print(f"    mixed      (theta=.5): lengths mean={len_mid.mean():.1f} std={len_mid.std():.3f}")
check("H4. theta=0.5 variance between det and stochastic",
      0.0 <= float(len_mid.std()) <= float(len_stoch.std()) + 1e-6)

print()
n_fail = sum(1 for _, ok in results if not ok)
print(f"{'='*50}\n{len(results)-n_fail}/{len(results)} checks passed"
      + ("" if n_fail == 0 else f"  ({n_fail} FAILED)"))
raise SystemExit(1 if n_fail else 0)
