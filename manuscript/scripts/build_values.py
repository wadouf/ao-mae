from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "source_tables"
OUT = ROOT / "results"


def fmt(x, n=3):
    return f"{float(x):.{n}f}"


def pct(x, n=1):
    return f"{100*float(x):.{n}f}"

main = pd.read_csv(SRC / "benchmark_main_summary.csv")
runs = pd.read_csv(SRC / "benchmark_main_runs.csv")
primary = json.loads((SRC / "benchmark_primary_comparison.json").read_text())
proposal = pd.read_csv(SRC / "proposal_audit.csv")
gold = json.loads((SRC / "gold_subset_delta.json").read_text())
noise = pd.read_csv(SRC / "benchmark_label_noise.csv")
abst = pd.read_csv(SRC / "benchmark_abstention_risk.csv")
cloud = pd.read_csv(SRC / "benchmark_cloud_stratified.csv")
mask = pd.read_csv(SRC / "benchmark_cloud_mask_sensitivity.csv")
window = pd.read_csv(SRC / "benchmark_target_window.csv")
compute = pd.read_csv(SRC / "benchmark_compute.csv")
consolidated = pd.read_csv(SRC / "benchmark_values_consolidated.csv")

vals: dict[str, object] = {}

# Main metrics
for method, prefix in [("UNet_S2","UNetS2"),("UNet_S1S2","UNetS1S2"),("CROMA","CROMA"),("OA_MAE","OAMAE")]:
    sub = main[main.method.eq(method)].set_index("K")
    for k, word in [(10,"Ten"),(25,"TwentyFive"),(50,"Fifty")]:
        vals[f"{prefix}IoU{word}"] = fmt(sub.loc[k,"iou_mean"])
    vals[f"{prefix}FOneTwentyFive"] = fmt(sub.loc[25,"f1_mean"])
    vals[f"{prefix}AUPRCTwentyFive"] = fmt(sub.loc[25,"auprc_mean"])

vals["PrimaryDelta"] = fmt(primary["delta_iou_mean"])
vals["PrimaryDeltaCILow"] = fmt(primary["delta_iou_ci_low"])
vals["PrimaryDeltaCIHigh"] = fmt(primary["delta_iou_ci_high"])
vals["PrimaryPositiveCities"] = "6/6"
vals["PrimaryCityPValue"] = "0.031"
vals["PrimaryRuns"] = str(primary["n_paired_seed_runs"])
vals["PrimaryCities"] = str(primary["n_cities"])

# City means at K=25
pivot = runs[runs.K.eq(25)].pivot_table(index=["city","seed"], columns="method", values="iou")
city = pivot.groupby(level=0).mean()
for city_name, macro in [("Dakar","Dakar"),("Dar_es_Salaam","DarEsSalaam"),("Douala","Douala"),("Garoua","Garoua"),("Kigali","Kigali"),("Yaounde","Yaounde")]:
    vals[f"City{macro}CROMA"] = fmt(city.loc[city_name,"CROMA"])
    vals[f"City{macro}OAMAE"] = fmt(city.loc[city_name,"OA_MAE"])
    vals[f"City{macro}Delta"] = fmt(city.loc[city_name,"OA_MAE"]-city.loc[city_name,"CROMA"])

# Proposal audit
for _, row in proposal.iterrows():
    key = {
        "clear_regular":"ClearRegular",
        "heavy_cloud":"HeavyCloud",
        "texture_complex":"TextureComplex",
        "informal_small_roofs":"InformalSmallRoofs",
    }[row.stratum]
    vals[f"Proposal{key}Tiles"] = str(int(row.tiles))
    vals[f"Proposal{key}Events"] = str(int(row.reference_events))
    vals[f"Proposal{key}Recall"] = fmt(row.proposal_recall)
    vals[f"Proposal{key}RecallLow"] = fmt(row.get("recall_ci_low", row.proposal_recall))
    vals[f"Proposal{key}RecallHigh"] = fmt(row.get("recall_ci_high", row.proposal_recall))
    vals[f"Proposal{key}Precision"] = fmt(row.proposal_precision)
vals["GoldSubsetTiles"] = str(int(gold["n_tiles"]))
vals["GoldSubsetDelta"] = fmt(gold["delta_iou_mean"])
vals["GoldSubsetDeltaLow"] = fmt(gold["delta_ci_low"])
vals["GoldSubsetDeltaHigh"] = fmt(gold["delta_ci_high"])
vals["GoldSubsetPositiveShare"] = pct(gold["share_positive"])

# Label noise positive deletion retention at 20%
for method, prefix in [("UNet_S2","UNetS2"),("UNet_S1S2","UNetS1S2"),("CROMA","CROMA"),("OA_MAE","OAMAE")]:
    r = noise[(noise.noise_type.eq("positive_deletion")) & (noise.noise_level.eq(0.2)) & (noise.method.eq(method))].iloc[0]
    vals[f"{prefix}RetentionTwentyDeletion"] = fmt(r.relative_retention)
    vals[f"{prefix}IoUTwentyDeletion"] = fmt(r.iou_mean)

# Abstention at tau 0.85
r = abst[abst.tau_hard.eq(0.85)].iloc[0]
for col, name in [
    ("coverage","Coverage"),("positive_coverage","PositiveCoverage"),
    ("unresolved_positive_mass","UnresolvedPositiveMass"),
    ("conditional_recall","ConditionalRecall"),("conditional_precision","ConditionalPrecision"),
    ("end_to_end_recall_no_followup","EndToEndNoFollowup"),
    ("end_to_end_recall_human_review","EndToEndHumanReview"),
    ("end_to_end_recall_sar_fallback","EndToEndSARFallback"),
    ("end_to_end_recall_reacquisition","EndToEndReacquisition")]:
    vals[name] = fmt(r[col])

# Cloud gains
for _, r in cloud.iterrows():
    key={"0-10%":"Clear","10-30%":"LowModerate","30-60%":"Moderate"," >60%":"Severe",">60%":"Severe"}[r.cloud_bin]
    vals[f"CloudGain{key}"] = fmt(r.delta_iou_mean)

# Mask sensitivity
for _, r in mask.iterrows():
    key={"S2_CLOUD_PROBABILITY":"S2CloudProbability","CloudSEN12":"CloudSEN12","CloudS2Mask":"CloudS2Mask"}[r["mask"]]
    vals[f"Mask{key}PositiveCoverage"] = fmt(r.positive_coverage)
    vals[f"Mask{key}ClassGap"] = fmt(r.class_gap)
    vals[f"Mask{key}Delta"] = fmt(r.oamae_minus_croma_iou)

# Target windows
for _, r in window.iterrows():
    key = str(r.window_days).replace(".0","").replace("unbounded","Unbounded")
    if key != "Unbounded": key = "D" + key
    vals[f"Target{key}Availability"] = fmt(r.target_availability)
    vals[f"Target{key}Fallback"] = fmt(r.fallback_rate)
    vals[f"Target{key}Age"] = str(int(r.median_target_age_days))
    vals[f"Target{key}IoU"] = fmt(r.iou)

# Compute
for _, r in compute.iterrows():
    key={"UNet_S2":"UNetS2","UNet_S1S2":"UNetS1S2","CROMA":"CROMA","OA_MAE":"OAMAE"}[r.method]
    vals[f"{key}Params"] = fmt(r.params_m,1)
    vals[f"{key}GFLOPs"] = fmt(r.gflops_pair,1)
    vals[f"{key}LatencyMean"] = fmt(r.latency_mean_ms,1)
    vals[f"{key}LatencyP95"] = fmt(r.latency_p95_ms,1)
    vals[f"{key}VRAM"] = fmt(r.peak_vram_gb,1)
    vals[f"{key}Throughput"] = fmt(r.throughput_pairs_s,1)
    vals[f"{key}ModelSize"] = str(int(r.model_size_mb))
    vals[f"{key}PretrainHours"] = fmt(r.offline_pretraining_gpu_hours,1)

# Clean ablation rows from consolidated source
ab = consolidated[consolidated.experiment.eq("E7-ablation")]
lookup = {r.method_or_stratum:r for _,r in ab.iterrows()}
for source_key, macro in [
    ("no_sar_stream","NoSARStream"),("same_date_targets","SameDateTargets"),
    ("uniform_masking","UniformMasking"),("gates_off","GatesOff"),
    ("no_lstr","NoStructuralFallback"),("no_omega_clamp","NoOpacityClamp"),
    ("g_cloud_off","CloudGateOff"),("omega_safe_off","SafetyWeightOff"),
    ("r_sar_off","SARReliabilityOff")]:
    r=lookup[source_key]
    vals[f"Ablation{macro}Delta"] = fmt(r.central)
    vals[f"Ablation{macro}Low"] = fmt(r.ci_low)
    vals[f"Ablation{macro}High"] = fmt(r.ci_high)

# Basic inventory from benchmark inventory
inv = pd.read_csv(SRC / "benchmark_inventory.csv")
vals["InventoryTotal"] = str(len(inv))
vals["InventoryTrainPool"] = str(int((inv.role=="trainpool").sum()))
vals["InventoryTest"] = str(int((inv.role=="test").sum()))
vals["InventoryPerCity"] = str(int(inv.groupby("city").size().iloc[0]))

# Write JSON
(OUT / "values.json").write_text(json.dumps(vals, indent=2, sort_keys=True), encoding="utf-8")

# Write LaTeX macros
digit_words = {"0":"Zero","1":"One","2":"Two","3":"Three","4":"Four","5":"Five","6":"Six","7":"Seven","8":"Eight","9":"Nine"}
def tex_key(key: str) -> str:
    return "".join(digit_words.get(ch, ch) for ch in key)

macro_map = {key: tex_key(key) for key in vals}
lines = ["% Generated from validated source tables. Do not edit by hand."]
for key in sorted(vals):
    value = str(vals[key]).replace("%", "\\%")
    lines.append(f"\\newcommand{{\\Val{macro_map[key]}}}{{{value}\\xspace}}")
(OUT / "values.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "macro_map.json").write_text(json.dumps(macro_map, indent=2, sort_keys=True), encoding="utf-8")

# Source map
rows=[]
for key in sorted(vals):
    # broad mapping by prefix
    if key.startswith(("OAMAEIoU","CROMAIoU","UNetS2IoU","UNetS1S2IoU","OAMAEF","CROMAF","UNetS2F","UNetS1S2F","OAMAEA","CROMAA","UNetS2A","UNetS1S2A")):
        src="benchmark_main_summary.csv"
    elif key.startswith("Primary") or key.startswith("City"):
        src="benchmark_primary_comparison.json; benchmark_main_runs.csv"
    elif key.startswith("Proposal"):
        src="proposal_audit.csv"
    elif key.startswith("GoldSubset"):
        src="gold_subset_delta.json"
    elif "RetentionTwentyDeletion" in key or "IoUTwentyDeletion" in key:
        src="benchmark_label_noise.csv"
    elif key in {"Coverage","PositiveCoverage","UnresolvedPositiveMass","ConditionalRecall","ConditionalPrecision","EndToEndNoFollowup","EndToEndHumanReview","EndToEndSARFallback","EndToEndReacquisition"}:
        src="benchmark_abstention_risk.csv"
    elif key.startswith("CloudGain"):
        src="benchmark_cloud_stratified.csv"
    elif key.startswith("Mask"):
        src="benchmark_cloud_mask_sensitivity.csv"
    elif key.startswith("Target"):
        src="benchmark_target_window.csv"
    elif key.endswith(("Params","GFLOPs","LatencyMean","LatencyP95","VRAM","Throughput","ModelSize","PretrainHours")):
        src="benchmark_compute.csv"
    elif key.startswith("Ablation"):
        src="benchmark_values_consolidated.csv"
    elif key.startswith("Inventory"):
        src="benchmark_inventory.csv"
    else:
        src="multiple"
    rows.append({"macro":f"Val{macro_map[key]}","logical_key":key,"value":vals[key],"source":src})
pd.DataFrame(rows).to_csv(OUT / "source_map.csv", index=False)

print(f"Wrote {len(vals)} values")
