"""Builds step8c_frozen_backend.ipynb: layer analysis, a frozen-detector back-end, and processing-aware calibration.

Reuses Step 8 Cells 1-5 (installs, setup + DSP effects, Step 3 data, model, train/dev/test sets).
The detector is run ONCE (frozen) to cache the time-average of all 25 hidden layers for every clip; everything
after that trains small models on the cached features, so the original detector can never change.

    python build_step8c.py ../step8c_frozen_backend.ipynb
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_step4 as b4  # noqa: E402
import build_step8 as b8  # noqa: E402

S8 = [src for kind, src in b8.CELLS if kind == "code"]      # S8[0] = Cell 1 ... S8[4] = Cell 5
CELLS = []


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


md(r"""
# Step 8c: where the deepfake evidence survives consumer processing, a frozen-detector back-end, and automatic processing-aware calibration

**Why:** Steps 8 and 8b changed the detector's weights (LoRA) and both times it forgot real-world fakes (In-the-Wild dev EER 4 -> 9.5%).
Here the detector is **never changed**. We run it once, keep the time-average of each of its 25 layers for every clip, and train small models on top.

**What this notebook measures and builds:**
1. **Layer analysis:** one linear probe per layer, so we can see *which layers still carry the deepfake evidence* after noise suppression, Bluetooth and Opus.
2. **Back-end + fusion:** a learned weighting of all 25 layers plus a linear layer, fused with the original head as `(1-w) * original + w * back-end`.
   `w = 0` is exactly the original model, and `w` is chosen on the dev sets, so the original can only be replaced if the dev data say so.
3. **Processing-aware calibration ("automatic line profile"):** a second small probe recognises which processing the audio went through (9 classes).
   Each processing gets its own threshold, calibrated on held-out speakers. This targets score *shifts* such as Opus, where 32.7% of real voices were flagged.

The test sets are the same unseen sets as in Step 8.

**Inputs:** GPU T4 x2, Internet ON, secret `HF_TOKEN`; attach your Step 4 notebook (Add Input -> Your Work) to reuse `step3_data.zip`.
**Outputs (small, please download):** `step8c_results.csv`, `step8c_layers.csv`, `step8c_fusion.csv`, `step8c_shift.csv`, `step8c_summary.json`, `step8c_backend.npz`.
**Large, download only when asked:** `voiceshield_layers.onnx` (the same detector, with the 25 layer averages as an extra output, for the live server).
""")

for i in range(5):                                          # Cells 1-5 unchanged from Step 8
    code(S8[i])

# ---------------------------------------------------------------- Cell 6: frozen features
code(r'''
# CELL 6: run the FROZEN detector once; keep, for every clip, the time-average of all hidden layers
# (CNN output + 24 transformer blocks) and the original model's score. Every later cell trains small models
# on these features, so the original detector itself never changes (no forgetting by construction).
import copy
random.seed(0); np.random.seed(0); torch.manual_seed(0)
CONDS = ["clean", "phone line (G.711)", "noise suppression", "spectral subtraction", "auto-gain",
         "auto-gain + noise suppression", "Bluetooth CVSD", "Bluetooth mSBC", "Opus"]
VIEWS, P_PROCESSED, FEAT_BS = 3, 0.85, 16          # 1 clean + 2 processed views of every training clip
model.to(DEVICE).eval()
for p in model.parameters():
    p.requires_grad = False
NL, D = model.ssl.config.num_hidden_layers + 1, model.ssl.config.hidden_size

@torch.no_grad()
def layer_feats(x):
    """x: (B, T) -> layer averages (B, NL, D) as fp16 numpy, and the original score llr = logit_fake - logit_real."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        out = model.ssl(F.layer_norm(x, x.shape[-1:]), output_hidden_states=True)
        # the last entry is taken from last_hidden_state (after the final layer norm), because transformers
        # versions differ on whether hidden_states[-1] is before or after that norm
        H = torch.stack([h.mean(1) for h in out.hidden_states[:-1]] + [out.last_hidden_state.mean(1)], 1)
    lo = model.head(H[:, -1].float())
    return H.half().cpu().numpy(), (lo[:, 0] - lo[:, 1]).float().cpu().numpy()

def set_feats(wavs):
    Hs, Ls = [], []
    for i in range(0, len(wavs), FEAT_BS):
        x = torch.from_numpy(np.stack([crop(w) for w in wavs[i:i + FEAT_BS]]).astype(np.float32)).to(DEVICE)
        h, l = layer_feats(x)
        Hs.append(h); Ls.append(l)
    return np.concatenate(Hs), np.concatenate(Ls)

# check: the last layer + the original head reproduce the detector's own output
xb = torch.from_numpy(np.stack([crop(w) for w in DEV["held-out speakers, clean"][0][:8]]).astype(np.float32)).to(DEVICE)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
    ref = model(xb).float()
diff = float(np.abs((ref[:, 0] - ref[:, 1]).cpu().numpy() - layer_feats(xb)[1]).max())
print(f"layer features + original head vs detector output: max llr difference {diff:.4f}")
assert diff < 0.1, "the feature path does not reproduce the detector"

# extra held-out sets so that every processing has its own calibration set
EXTRA = {f"held-out speakers, {c}": dsp_set(dev3, c, 31 + i)
         for i, c in enumerate(["spectral subtraction", "auto-gain", "Bluetooth mSBC"])}
SETS = {**DEV, **EXTRA, **TEST}
FE, t0 = {}, time.time()
for name, (w, y) in SETS.items():
    H, L = set_feats(w)
    FE[name] = (H, L, np.asarray(y))
print(f"features of {len(FE)} dev/calibration/test sets in {(time.time() - t0) / 60:.1f} min")

ORDER = ["auto-gain + noise suppression", "spectral subtraction", "noise suppression", "auto-gain",
         "Bluetooth CVSD", "Bluetooth mSBC", "Opus"]
def cond_of(name):
    """The processing a dev/test set was made with (In-the-Wild and commercial clips count as 'clean')."""
    for c in ORDER:
        if c in name:
            return c
    return "phone line (G.711)" if "phone" in name else "clean"

class ViewSet(torch.utils.data.Dataset):
    """View 0: clean crop. Views 1..: background noise (half the time) + one processing, kept as a label."""
    def __init__(self, df):
        self.items = [(p, int(l), v) for p, l in zip(df.path, df.label) for v in range(VIEWS)]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        path, label, v = self.items[i]
        x = crop(trim(load_wav(path)), random_start=True).astype(np.float64)
        c = 0
        if v > 0:
            if random.random() < 0.5:
                x = add_noise(x, random.uniform(5, 30))
            if random.random() < P_PROCESSED:
                c = random.randrange(1, len(CONDS))
                x = DSP[CONDS[c]](x)
        return torch.from_numpy(np.asarray(crop(x), np.float32)), label, c

def _seed_worker(_):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)

views = ViewSet(train_df)
view_loader = torch.utils.data.DataLoader(views, batch_size=FEAT_BS, shuffle=False,
                                          num_workers=min(4, os.cpu_count() or 2), worker_init_fn=_seed_worker)
N = len(views)
Htr = np.empty((N, NL, D), np.float16); Ltr = np.empty(N, np.float32)
Ytr = np.empty(N, np.int64); Ctr = np.empty(N, np.int64)
k, t0 = 0, time.time()
for b, (x, y, c) in enumerate(view_loader):
    h, l = layer_feats(x.to(DEVICE))
    n = len(y)
    Htr[k:k + n], Ltr[k:k + n], Ytr[k:k + n], Ctr[k:k + n] = h, l, y.numpy(), c.numpy()
    k += n
    if b % 200 == 0:
        print(f"training views {k}/{N} ({(time.time() - t0) / 60:.1f} min)")
print(f"{N} training views ({len(train_df)} clips x {VIEWS}) in {(time.time() - t0) / 60:.1f} min; processing used:",
      dict(zip(CONDS, np.bincount(Ctr, minlength=len(CONDS)).tolist())))
''')

# ---------------------------------------------------------------- Cell 7: layer analysis
code(r'''
# CELL 7: where does the deepfake evidence survive consumer processing? One linear probe per layer,
# trained on the training views, EER on every dev and test set.
torch.manual_seed(0)
Xtr = torch.from_numpy(Htr).to(DEVICE)                       # fp16 on the GPU
ytr = torch.from_numpy(Ytr).float().to(DEVICE)               # 1 = fake
MU = torch.stack([Xtr[:, l].float().mean(0) for l in range(NL)])
SD = torch.stack([Xtr[:, l].float().std(0) for l in range(NL)])
SD = torch.maximum(SD, 0.01 * SD.median(1, keepdim=True).values) + 1e-6   # floor: near-constant dims stay harmless

def eer(y, s):
    return float(eer_auc(y, s)["EER_%"])

def fit_linear(x, y, steps=300):
    lin = nn.Linear(x.shape[1], 1).to(DEVICE)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(steps):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(lin(x).squeeze(1), y).backward()
        opt.step()
    return lin.eval()

def torch_layer(H, l):
    return (torch.from_numpy(np.ascontiguousarray(H[:, l])).to(DEVICE).float() - MU[l]) / SD[l]

LAYER_NAMES = [f"layer {l:02d}" for l in range(NL)] + ["original head"]
rows = []
for l in range(NL):
    lin = fit_linear((Xtr[:, l].float() - MU[l]) / SD[l], ytr)
    with torch.no_grad():
        for name, (H, L, y) in FE.items():
            rows.append({"layer": LAYER_NAMES[l], "set": name,
                         "EER_%": eer(y, lin(torch_layer(H, l)).squeeze(1).cpu().numpy())})
for name, (H, L, y) in FE.items():
    rows.append({"layer": "original head", "set": name, "EER_%": eer(y, L)})
layers = pd.DataFrame(rows)
layers.to_csv(f"{WORK}/step8c_layers.csv", index=False)
piv = layers.pivot(index="layer", columns="set", values="EER_%").reindex(LAYER_NAMES)
piv["dev mean"] = piv[list(DEV)].mean(1)
piv["test mean"] = piv[list(TEST)].mean(1)
show = [k for k in ["held-out speakers, clean", "In-the-Wild (dev sample)", "held-out speakers, noise suppression",
                    "held-out speakers, Bluetooth CVSD", "held-out speakers, Opus"] if k in piv]
print("Linear-probe EER % per layer (layer 00 = CNN output, layer 24 = last block):")
print(piv[show + ["dev mean", "test mean"]].round(2).to_string())
''')

# ---------------------------------------------------------------- Cell 8: back-end + fusion
code(r'''
# CELL 8: our back-end: a learned weighting of ALL layers + a linear layer, on top of the frozen detector,
# fused with the original head. Fusion weight w = 0 is exactly the original model; w is chosen on dev.
class LayerProbe(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(NL))                  # layer weights (softmax)
        self.drop = nn.Dropout(0.2)
        self.lin = nn.Linear(D, n_out)
    def forward(self, H):
        z = (H.float() - MU) / SD
        return self.lin(self.drop((torch.softmax(self.a, 0)[None, :, None] * z).sum(1)))

@torch.no_grad()
def probe_out(probe, H, bs=512):
    probe.eval()
    return torch.cat([probe(torch.from_numpy(H[i:i + bs]).to(DEVICE)) for i in range(0, len(H), bs)]).float().cpu().numpy()

def train_probe(n_out, target, metric, epochs, lr=1e-3, bs=256):
    """Trains on the training views; keeps the epoch with the LOWEST metric(probe) on dev data."""
    torch.manual_seed(0)
    probe = LayerProbe(n_out).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    best, best_ep, best_state, hist = float("inf"), 0, None, []
    for ep in range(1, epochs + 1):
        probe.train()
        for idx in torch.randperm(len(Xtr), device=DEVICE).split(bs):
            out = probe(Xtr[idx])
            loss = (F.binary_cross_entropy_with_logits(out.squeeze(1), target[idx]) if n_out == 1
                    else F.cross_entropy(out, target[idx]))
            opt.zero_grad(); loss.backward(); opt.step()
        m = metric(probe)
        hist.append(m)
        if m < best:
            best, best_ep, best_state = m, ep, copy.deepcopy(probe.state_dict())
    probe.load_state_dict(best_state)
    return probe.eval(), best_ep, hist

DEV_NAMES = list(DEV)
def dev_mean_eer(scores):
    return float(np.mean([eer(FE[k][2], scores[k]) for k in DEV_NAMES]))

det, det_ep, det_hist = train_probe(1, ytr, lambda p: dev_mean_eer({k: probe_out(p, FE[k][0])[:, 0] for k in DEV_NAMES}),
                                    epochs=30)
O = {k: L for k, (H, L, y) in FE.items()}                              # original score (positive = fake)
P = {k: probe_out(det, H)[:, 0] for k, (H, L, y) in FE.items()}        # back-end score (positive = fake)
print(f"back-end: best epoch {det_ep}, dev mean EER {min(det_hist):.2f}% (original model {dev_mean_eer(O):.2f}%)")
print("layer weights learned by the back-end:", np.round(torch.softmax(det.a, 0).detach().cpu().numpy(), 3).tolist())

o_dev = np.concatenate([O[k] for k in DEV_NAMES]); p_dev = np.concatenate([P[k] for k in DEV_NAMES])
NORM = dict(mo=float(o_dev.mean()), so=float(o_dev.std()), mp=float(p_dev.mean()), sp=float(p_dev.std()))
def fuse(o, p, w):
    return (1 - w) * (o - NORM["mo"]) / NORM["so"] + w * (p - NORM["mp"]) / NORM["sp"]
GRID = [round(float(w), 2) for w in np.linspace(0, 1, 21)]
fusion = pd.DataFrame([{"w": w, "dev_mean_EER": dev_mean_eer({k: fuse(O[k], P[k], w) for k in DEV_NAMES})}
                       for w in GRID])
W = float(fusion.sort_values(["dev_mean_EER", "w"]).iloc[0]["w"])   # ties -> the smaller w (closer to the original)
fusion.to_csv(f"{WORK}/step8c_fusion.csv", index=False)
print(fusion.iloc[::2].round(2).to_string(index=False))
print(f"chosen fusion weight w = {W}  (0 = original model only, 1 = back-end only)")
S = {"original": O, "back-end": P, "fused": {k: fuse(O[k], P[k], W) for k in FE}}
dev_tab = pd.DataFrame({tag: {k: eer(FE[k][2], sc[k]) for k in DEV_NAMES} for tag, sc in S.items()})
dev_tab.loc["MEAN"] = dev_tab.mean()
print("\nDev EER % (the data all choices were made on):")
print(dev_tab.round(2).to_string())
''')

# ---------------------------------------------------------------- Cell 9: processing-aware calibration
code(r'''
# CELL 9: processing-aware calibration ("automatic line profile"). A second small probe on the same frozen
# features recognises which processing the audio went through; every processing has its own threshold,
# calibrated on held-out speakers. This corrects score SHIFTS (e.g. Opus); it cannot undo information loss.
ctr = torch.from_numpy(Ctr).to(DEVICE)
CAL_SETS = {"clean": ["held-out speakers, clean", "Commercial, unseen Hume", "In-the-Wild (dev sample)"],
            **{c: [k for k in list(DEV) + list(EXTRA) if k.startswith("held-out") and cond_of(k) == c]
               for c in CONDS[1:]}}
assert all(CAL_SETS[c] for c in CONDS), CAL_SETS
CAL_NAMES = sorted({k for v in CAL_SETS.values() for k in v})
cond_true = {k: CONDS.index(cond_of(k)) for k in FE}

def cond_metric(probe):
    return -float(np.mean([(probe_out(probe, FE[k][0]).argmax(1) == cond_true[k]).mean() for k in CAL_NAMES]))

cls, cls_ep, cls_hist = train_probe(len(CONDS), ctr, cond_metric, epochs=20)
PRED = {k: probe_out(cls, FE[k][0]).argmax(1) for k in FE}
acc = {k: 100 * float((PRED[k] == cond_true[k]).mean()) for k in FE}
print(f"processing classifier: best epoch {cls_ep}, mean accuracy on held-out speakers {-100 * min(cls_hist):.1f}%")
for k in TEST:
    top = int(np.bincount(PRED[k], minlength=len(CONDS)).argmax())
    print(f"  {k:46s} true: {cond_of(k):30s} accuracy {acc[k]:5.1f}%  (most often predicted: {CONDS[top]})")

def thr_at_eer(names, sc):
    y = np.concatenate([FE[k][2] for k in names]); s = np.concatenate([sc[k] for k in names])
    return float(eer_auc(y, s)["threshold_at_EER"])

THR = {tag: {"single": thr_at_eer(DEV_NAMES, sc), **{c: thr_at_eer(CAL_SETS[c], sc) for c in CONDS}}
       for tag, sc in S.items()}
print("\nOriginal model thresholds (llr): single", round(THR["original"]["single"], 2), "| per processing:",
      {c: round(THR["original"][c], 2) for c in CONDS})

def rates(y, s, thr):
    """% of REAL clips flagged (false alarms) and % of FAKE clips missed, at a threshold (scalar or per clip)."""
    y = np.asarray(y); flag = np.asarray(s) >= thr
    return 100 * float(flag[y == 0].mean()), 100 * float((~flag[y == 1]).mean())
''')

# ---------------------------------------------------------------- Cell 10: final test + saves
code(r'''
# CELL 10: final test (every set unseen): original vs back-end vs fused; one threshold vs automatic per-processing
rows = []
for k in TEST:
    y = FE[k][2]
    for tag, sc in S.items():
        s, r = sc[k], eer_auc(y, sc[k])
        fa1, mr1 = rates(y, s, THR[tag]["single"])
        fa2, mr2 = rates(y, s, np.array([THR[tag][CONDS[c]] for c in PRED[k]]))
        fa3, mr3 = rates(y, s, THR[tag][cond_of(k)])
        rows.append({"test set": k, "scorer": tag, "EER_%": r["EER_%"], "AUC": r["AUC"],
                     "FAR_single_%": fa1, "MR_single_%": mr1, "FAR_auto_%": fa2, "MR_auto_%": mr2,
                     "FAR_oracle_%": fa3, "MR_oracle_%": mr3, "processing_accuracy_%": acc[k]})
res = pd.DataFrame(rows).round(3)
res.to_csv(f"{WORK}/step8c_results.csv", index=False)
shift = pd.DataFrame([{"test set": k, "median_llr_real": float(np.median(O[k][FE[k][2] == 0])),
                       "median_llr_fake": float(np.median(O[k][FE[k][2] == 1]))} for k in TEST]).round(2)
shift.to_csv(f"{WORK}/step8c_shift.csv", index=False)

pd.set_option("display.width", 250)
print("EER % on unseen test sets (lower is better):")
print(res.pivot(index="test set", columns="scorer", values="EER_%").reindex(list(TEST))[list(S)].round(2).to_string())
print("\nOriginal model, false alarms on REAL voices / missed FAKES (%), one threshold -> automatic per-processing:")
o = res[res.scorer == "original"].set_index("test set")
print(o[["FAR_single_%", "FAR_auto_%", "MR_single_%", "MR_auto_%", "processing_accuracy_%"]].round(1).to_string())
print("\nFull table:"); print(res.to_string(index=False))

dsp_rows = [k for k in TEST if cond_of(k) != "clean"]
def m(tag, col, names):
    return float(res[(res.scorer == tag) & res["test set"].isin(names)][col].mean())
summary = {
    "fusion_w": W, "backend_best_epoch": det_ep, "processing_classifier_best_epoch": cls_ep,
    "dev_mean_EER": {tag: float(dev_tab.loc["MEAN", tag]) for tag in S},
    "test_mean_EER_all": {tag: m(tag, "EER_%", list(TEST)) for tag in S},
    "test_mean_EER_processed": {tag: m(tag, "EER_%", dsp_rows) for tag in S},
    "processed_test_FAR_single_vs_auto": {tag: [m(tag, "FAR_single_%", dsp_rows), m(tag, "FAR_auto_%", dsp_rows)] for tag in S},
    "processed_test_MR_single_vs_auto": {tag: [m(tag, "MR_single_%", dsp_rows), m(tag, "MR_auto_%", dsp_rows)] for tag in S},
    "processing_accuracy_heldout_%": -100 * min(cls_hist),
    "processing_accuracy_test_%": {k: acc[k] for k in TEST},
    "thresholds_llr_original": THR["original"], "thresholds_fused": THR["fused"],
    "backend_layer_weights": torch.softmax(det.a, 0).detach().cpu().numpy().round(4).tolist(),
    "classifier_layer_weights": torch.softmax(cls.a, 0).detach().cpu().numpy().round(4).tolist(),
    "conds": CONDS, "views_per_clip": VIEWS, "train_views": int(N),
}
json.dump(summary, open(f"{WORK}/step8c_summary.json", "w"), indent=2)
np.savez(f"{WORK}/step8c_backend.npz", conds=np.array(CONDS), mu=MU.cpu().numpy(), sd=SD.cpu().numpy(),
         det_a=det.a.detach().cpu().numpy(), det_w=det.lin.weight.detach().cpu().numpy()[0],
         det_b=det.lin.bias.detach().cpu().numpy(), cls_a=cls.a.detach().cpu().numpy(),
         cls_w=cls.lin.weight.detach().cpu().numpy(), cls_b=cls.lin.bias.detach().cpu().numpy(),
         fusion_w=W, mo=NORM["mo"], so=NORM["so"], mp=NORM["mp"], sp=NORM["sp"],
         thr_original=np.array([THR["original"][c] for c in CONDS]), thr_fused=np.array([THR["fused"][c] for c in CONDS]),
         thr_single_original=THR["original"]["single"], thr_single_fused=THR["fused"]["single"])
print("\n==== SUMMARY (please copy everything below) ====")
for tag in S:
    print(f"{tag:9s} dev mean EER {summary['dev_mean_EER'][tag]:6.2f}% | test mean EER all {summary['test_mean_EER_all'][tag]:6.2f}%"
          f" | processed {summary['test_mean_EER_processed'][tag]:6.2f}% | false alarms on processed sets "
          f"{summary['processed_test_FAR_single_vs_auto'][tag][0]:.1f}% -> {summary['processed_test_FAR_single_vs_auto'][tag][1]:.1f}%"
          f" | misses {summary['processed_test_MR_single_vs_auto'][tag][0]:.1f}% -> {summary['processed_test_MR_single_vs_auto'][tag][1]:.1f}%")
print(f"fusion weight w = {W}; processing classifier accuracy on held-out speakers {summary['processing_accuracy_heldout_%']:.1f}%")
''')

# ---------------------------------------------------------------- Cell 11: ONNX with layer outputs
code(r'''
# CELL 11: export the SAME detector with its layer averages as a second output, for the live server
# (the back-end, the fusion and the processing classifier then run in NumPy from step8c_backend.npz).
EXPORT_ONNX = True
if EXPORT_ONNX:
    import torch.nn.utils.parametrize as _P
    m_cpu = model.float().cpu().eval()
    for mod in [mod for mod in m_cpu.modules() if _P.is_parametrized(mod)]:
        for pname in list(mod.parametrizations.keys()):
            _P.remove_parametrizations(mod, pname, leave_parametrized=True)
    for mod in [mod for mod in m_cpu.modules() if hasattr(mod, "weight_g")]:
        torch.nn.utils.remove_weight_norm(mod)
    class LayerExport(nn.Module):
        """Same maths as the detector; layer_norm over a dynamic length written as mean/var for ONNX."""
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, wav):
            wav = (wav - wav.mean(-1, keepdim=True)) / torch.sqrt(wav.var(-1, keepdim=True, unbiased=False) + 1e-5)
            out = self.m.ssl(wav, output_hidden_states=True)
            layers = torch.stack([h.mean(1) for h in out.hidden_states[:-1]] + [out.last_hidden_state.mean(1)], 1)
            return self.m.head(layers[:, -1]), layers
    onnx_path = f"{WORK}/voiceshield_layers.onnx"
    kw = dict(input_names=["wav"], output_names=["logits", "layers"], opset_version=17,
              dynamic_axes={"wav": {0: "batch"}, "logits": {0: "batch"}, "layers": {0: "batch"}})
    dummy = torch.randn(1, T) * 0.1
    try:
        torch.onnx.export(LayerExport(m_cpu).eval(), dummy, onnx_path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(LayerExport(m_cpu).eval(), dummy, onnx_path, **kw)
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.stack([crop(w) for w in TEST["test speakers, clean"][0][:4]]).astype(np.float32)
    xn = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-7)   # what the live server sends
    with torch.no_grad():
        ref_lo, ref_la = LayerExport(m_cpu).eval()(torch.from_numpy(x))
    lo, la = sess.run(None, {"wav": xn})
    print(f"Exported {onnx_path} ({os.path.getsize(onnx_path) / 1e9:.2f} GB); max difference vs PyTorch: "
          f"logits {np.abs(lo - ref_lo.numpy()).max():.4f}, layers {np.abs(la - ref_la.numpy()).max():.4f}")
''')

if __name__ == "__main__":
    b4.CELLS[:] = CELLS
    b4.build(sys.argv[1])
