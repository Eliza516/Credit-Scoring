"""Chạy thử toàn pipeline trên dữ liệu demo (dùng để kiểm tra nhanh, không phải test chính thức)."""
import sys, time
sys.path.insert(0, ".")
import pandas as pd, numpy as np
from core import config, demo, ingest, cleaning, target, vintage, split, binning, selection, modeling, evaluation, scorecard, export

t0 = time.time()
cfg = config.load_default(False)
cfg["split"]["imbalance"]["method"] = "none"
df = demo.make_demo(); m = demo.DEMO_MAPPING
tgt = target.build_target(df, m, cfg)
dev = target.development_mask(tgt, cfg)
vt = vintage.vintage_table(df, m, cfg, tgt["status"]); print(vintage.suggest_window(vt, cfg)["message"])
d = df[dev]; y = tgt.loc[dev, "y"]
feats = ingest.feature_columns(df, m, ["first_bad_mob"])
parts = split.split_indices(d, y, cfg, m["open_date"]); print(split.check_sizes(y, parts, cfg))
an = cleaning.analyze(d.loc[parts["train"]], feats, cfg); print(an[["variable","suggestion","reason"]].query("suggestion!='keep'"))
drop = an.query("suggestion=='drop'")["variable"].tolist()
feats = [f for f in feats if f not in drop]
tr, te = parts["train"], parts["test"]
specs = binning.fit_all(d.loc[tr], y.loc[tr], feats, cfg)
tabs = binning.tables(d.loc[tr], y.loc[tr], specs); tabs_te = binning.tables(d.loc[te], y.loc[te], specs)
specs = {c: binning.attach_woe(specs[c], tabs[c]) for c in specs}
W_tr = binning.transform(d.loc[tr], specs); W_te = binning.transform(d.loc[te], specs)
print(binning.iv_summary(tabs).head(12))
fn = selection.funnel(d.loc[tr], W_tr, tabs, tabs_te, cfg); print(fn[["variable","iv","suggestion","stage","reason"]])
cand = fn.query("suggestion=='keep'")["variable"].tolist()
sw = modeling.stepwise(W_tr, y.loc[tr], cfg, cand); print(sw["steps"]); sel = sw["selected"]
mod = modeling.fit_logistic(W_tr, y.loc[tr], sel, cfg); print(mod["table"].round(3))
r_tr = modeling.predict_logit(mod, W_tr); r_te = modeling.predict_logit(mod, W_te)
mt = {"train": evaluation.metrics_row(y.loc[tr].to_numpy(), r_tr, cfg), "test": evaluation.metrics_row(y.loc[te].to_numpy(), r_te, cfg)}
a = evaluation.assessment(mt, cfg); print(a.round(3)); print(a.attrs["warnings"])
ch = modeling.fit_challengers(W_tr, y.loc[tr], sel, cfg)
rows = {"logistic": mt["test"]}
for k, v in ch.items(): rows[k] = evaluation.metrics_row(y.loc[te].to_numpy(), v.predict_proba(W_te[sel])[:, 1], cfg)
print(evaluation.compare_models(rows, cfg).round(3)); print(evaluation.ml_vs_primary_warnings(rows, "logistic", cfg))
pts = scorecard.build_table(mod, specs, tabs, cfg)
s_te = scorecard.score(d.loc[te], mod, specs, pts, cfg)
print(evaluation.tranche_table(y.loc[te], s_te).round(3))
print(np.corrcoef(s_te, -r_te)[0, 1])
cfg["cutoff"]["target_bad_rate"] = 0.08
ct = scorecard.cutoff_table(y.loc[te], s_te); sg = scorecard.suggest_cutoff(ct, cfg, y.loc[te], s_te); print(sg)
print(time.time() - t0)
