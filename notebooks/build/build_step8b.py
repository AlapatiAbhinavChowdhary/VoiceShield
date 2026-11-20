"""Builds step8b_consistency_lora.ipynb: teacher-anchored consistency LoRA fine-tuning.

Same data, DSP effects, test sets and export as Step 8 (cells reused); only the intro, the training
cell and the output names change. The frozen original model listens to the CLEAN clip; the LoRA model must
give the same decision for the PROCESSED version of that clip (plus the usual label loss).

    python build_step8b.py ../step8b_consistency_lora.ipynb
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_step4 as b4  # noqa: E402
import build_step8 as b8  # noqa: E402

S8 = [src for kind, src in b8.CELLS if kind == "code"]      # S8[0] = Cell 1 ... S8[7] = Cell 8
CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md("""
# Step 8b: teacher-anchored consistency LoRA, making the detector robust to consumer audio processing

**What Step 8 showed:** the original detector is near-perfect on clean audio but breaks under everyday processing
(Bluetooth CVSD: EER 0% -> 10%, about 27% of fakes missed; Opus: about 33% of real voices flagged at the normal cut-off).
Plain LoRA fine-tuning moved the model away from what it knew before it learned the new conditions, and stopped early.

**Method (ours):** for every training clip we make a clean crop and a processed copy of the SAME crop
(background noise + one consumer-DSP effect).
- The **frozen original model** (LoRA switched off) scores the *clean* crop: that is the teacher's decision.
- The **LoRA model** scores the *processed* crop and is trained with
  `loss = cross-entropy(label) + lambda * tau^2 * KL(teacher_clean || student_processed)`,
  where the KL term is used only when the teacher is right about the clean clip (so it never teaches a mistake).
- So the model learns "give the clean-audio answer even after processing", anchored to the original, which also prevents forgetting.

Gentler learning rate (5e-5), longer patience. Same unseen test sets as Step 8; the original model is step 0.

**Inputs:** GPU T4 x2, Internet ON, secret `HF_TOKEN`; attach your Step 4 notebook (Add Input -> Your Work) to reuse `step3_data.zip`.
**Outputs:** `step8b_results.csv`, `step8b_history.csv`, `voiceshield_lora_adapter/`, `voiceshield_dsp_lora.onnx`, `step8b_threshold.json`.
""")

for i in range(6):                                          # Cells 1-6 unchanged from Step 8
    code(S8[i])

code('''
# CELL 7: teacher-anchored consistency LoRA (backbone frozen; LoRA adapters + 2-way head learn)
from peft import LoraConfig, get_peft_model
BS, ACCUM, EPOCHS = 8, 2, 3
LR_LORA, LR_HEAD, WARMUP = 5e-5, 1e-5, 100
EVAL_EVERY, PATIENCE = 100, 6
LAMBDA, TAU = 1.0, 2.0
P_PROCESS = 0.85                                 # share of training clips the student hears processed
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "intermediate_dense", "output_dense"]
BEST = f"{WORK}/step8b_best_trainable.pt"

for p in model.parameters():
    p.requires_grad = False
model.ssl = get_peft_model(model.ssl, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                                 target_modules=LORA_TARGETS))
for p in model.head.parameters():
    p.requires_grad = True
model.to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable parameters: {n_train / 1e6:.2f} M of {n_all / 1e6:.0f} M ({100 * n_train / n_all:.2f}%)")

def process(x):
    """Background noise (half the time) + ONE consumer-DSP effect (or a low-pass), as in Step 8."""
    x = np.asarray(x, np.float64)
    if random.random() < 0.5:
        x = add_noise(x, random.uniform(5, 30))
    if random.random() < 0.8:
        return DSP[random.choice(DSP_NAMES)](x)
    return lowpass(x, random.uniform(3500, 7000))

class PairSet(torch.utils.data.Dataset):
    """(clean crop, processed copy of the same crop, target) - target 0 = fake, 1 = real."""
    def __init__(self, df):
        self.paths, self.labels = df.path.tolist(), df.label.tolist()
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        clean = crop(trim(load_wav(self.paths[i])), random_start=True).astype(np.float64)
        proc = process(clean) if random.random() < P_PROCESS else clean
        return (torch.from_numpy(clean.astype(np.float32)), torch.from_numpy(np.asarray(crop(proc), np.float32)),
                1 - self.labels[i])

def _seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)

loader = torch.utils.data.DataLoader(PairSet(train_df), batch_size=BS, shuffle=True, drop_last=True,
                                     num_workers=min(4, os.cpu_count() or 2), worker_init_fn=_seed_worker,
                                     persistent_workers=True)
lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
opt = torch.optim.AdamW([{"params": lora_params, "lr": LR_LORA, "weight_decay": 0.01},
                         {"params": model.head.parameters(), "lr": LR_HEAD, "weight_decay": 0.0}])
total = EPOCHS * len(loader) // ACCUM
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARMUP) * max(0.05, 1 - s / total))
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

def trainable_state(m):
    return {k: v.detach().half().cpu() for k, v in m.state_dict().items() if "lora_" in k or k.startswith("head.")}

def save(m):
    torch.save(trainable_state(m), BEST)

import copy
T_HEAD = copy.deepcopy(model.head).eval()        # the ORIGINAL head, frozen: the teacher never drifts
for p in T_HEAD.parameters():
    p.requires_grad = False

def teacher_logits(xc):
    """The ORIGINAL model (LoRA off, original head, eval mode) on the clean crops."""
    model.eval()
    with torch.no_grad(), model.ssl.disable_adapter(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        h = model.ssl(F.layer_norm(xc, xc.shape[-1:])).last_hidden_state.mean(1)
        return T_HEAD(h).float()

save(model)                                      # LoRA starts at zero: this IS the original model
best, best_step, bad, step = zs_metric, 0, 0, 0
history = [{"step": 0, "dev_mean_EER": zs_metric, **zs_dev_eer}]
t0, stop = time.time(), False
logs = {"ce": [], "kd": [], "gate": []}
print(f"{len(train_df)} training clips, {len(loader)} batches/epoch, {total} optimizer steps max")
for epoch in range(EPOCHS):
    for i, (xc, xp, y) in enumerate(loader):
        xc, xp, y = xc.to(DEVICE), xp.to(DEVICE), y.to(DEVICE)
        t_log = teacher_logits(xc)
        model.train()
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            s_log = model(xp)
        s_log = s_log.float()
        ce = F.cross_entropy(s_log, y)
        gate = (t_log.argmax(-1) == y).float()                     # teach only what the teacher gets right
        kl = F.kl_div(F.log_softmax(s_log / TAU, -1), F.softmax(t_log / TAU, -1), reduction="none").sum(-1)
        kd = (kl * gate).sum() / gate.sum().clamp(min=1.0) * TAU ** 2
        loss = (ce + LAMBDA * kd) / ACCUM
        scaler.scale(loss).backward()
        logs["ce"].append(ce.item()); logs["kd"].append(kd.item()); logs["gate"].append(gate.mean().item())
        if (i + 1) % ACCUM:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        step += 1
        if step % 25 == 0:
            print(f"epoch {epoch + 1} step {step}/{total} CE {np.mean(logs['ce']):.4f} KD {np.mean(logs['kd']):.4f} "
                  f"teacher-right {100 * np.mean(logs['gate']):.0f}% ({(time.time() - t0) / 60:.1f} min)")
            logs = {"ce": [], "kd": [], "gate": []}
        if step % EVAL_EVERY == 0:
            _, dev_eer = evaluate(model, DEV)
            metric = float(np.mean(list(dev_eer.values())))
            history.append({"step": step, "dev_mean_EER": metric, **dev_eer})
            print(f"  >> dev mean EER {metric:.2f}% (best {best:.2f}% at step {best_step})",
                  {k: round(float(v), 2) for k, v in dev_eer.items()})
            if metric < best:
                best, best_step, bad = metric, step, 0
                save(model)
                print("  >> saved new best")
            else:
                bad += 1
                if bad >= PATIENCE:
                    stop = True
                    break
    if stop:
        print("Early stopping: dev EER stopped improving.")
        break
pd.DataFrame(history).to_csv(f"{WORK}/step8b_history.csv", index=False)
print(f"Done in {(time.time() - t0) / 60:.1f} min. Best dev mean EER {best:.2f}% at step {best_step} "
      f"(original model: {zs_metric:.2f}%)")
''')

c8 = (S8[7].replace("step8_results.csv", "step8b_results.csv")
           .replace("step8_threshold.json", "step8b_threshold.json")
           .replace('("original", zs_test_scores, zs_thr), ("DSP-aware LoRA", ft_test_scores, ft_thr)',
                    '("original", zs_test_scores, zs_thr), ("DSP-aware LoRA", ft_test_scores, ft_thr)'))
c8 = c8.replace(
    'print("Best checkpoint step:", best_step, "(0 = training did not beat the original model)")',
    'fa = res.pivot(index="test set", columns="model", values="false_alarm_on_real_%").reindex(list(TEST))\n'
    'print(f"Mean false alarms on REAL voices, DSP test sets: original {fa.loc[dsp_rows, \'original\'].mean():.1f}% -> '
    'LoRA {fa.loc[dsp_rows, \'DSP-aware LoRA\'].mean():.1f}%")\n'
    'print("Best checkpoint step:", best_step, "(0 = training did not beat the original model)")')
c8 = c8.replace('"lora": {"r": 8, "alpha": 16, "targets": LORA_TARGETS}',
                '"lora": {"r": 8, "alpha": 16, "targets": LORA_TARGETS}, '
                '"method": {"loss": "CE + lambda*tau^2*KL(teacher_clean||student_processed), teacher-correct gated", '
                '"lambda": LAMBDA, "tau": TAU, "p_process": P_PROCESS}')
assert c8.count("step8b_") == 2 and "Mean false alarms" in c8 and '"method"' in c8
code(c8)

if __name__ == "__main__":
    b4.CELLS[:] = CELLS
    b4.build(sys.argv[1])
