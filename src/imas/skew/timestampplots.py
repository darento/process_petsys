"""IMAS side timestamp rule diagnostics in the point-source CTR (spec 014 T9, FR-7). Diagnostics only, not
validation evidence.

For each CTR table applied under a rule other than ``top`` (``ctr.RULE_STAGE`` tables), on the near-source
events of the nominal variant:

* the **lead** of each side: the earliest candidate's corrected time minus the top's (ps; 0 when the top is the
  earliest), with that table's skew and walk (``timestamp.side_offset``, twice the ``top_earliest_mean`` offset);
* the residual ``dt - TOF`` per event lead bin (the more negative of the two sides), oriented so that this
  leading side's late top is positive (negated when side 2 leads), **before** (the same tables under ``top``)
  and **after** (under the table's rule), fitted like the CTR: model-free and Gaussian FWHM and the Gaussian
  mean. Within a bin a late top shifts the residual rather than widening it, so the rule shows in the mean;
* per minimodule (SM, mm of the side): sides and sides where the rule acts.

``RuleDiagnostics`` accumulates them per declared position; ``summary`` reduces them; ``write_plots`` writes
``<stem>_<table>_lead.png``, ``<stem>_<table>_ctr_by_lead.png`` and ``<stem>_<table>_minimodule_share.png`` next
to the CTR report (headless, Agg).
"""
from __future__ import annotations

import os
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.imas.skew import timestamp
from src.imas.skew import walk as iw
from src.timing import ctr as core

LEAD_EDGES_PS = np.arange(-3000.0, 0.0 + 1e-9, 25.0)       # side lead histogram; 0 and below-range counted apart
OFFSET_EDGES_PS = (0.0, 100.0, 250.0, 500.0, 1000.0, np.inf)   # event |lead| bins after "rule does not act"
NO_ACT = "rule does not act"


def offset_labels():
    labels = [NO_ACT]
    for low, high in zip(OFFSET_EDGES_PS[:-1], OFFSET_EDGES_PS[1:]):
        labels.append(f"> {low:g} ps" if not np.isfinite(high) else f"{low:g}-{high:g} ps")
    return labels


def side_leads(features, skew, walk):
    """Per side (1, 2) the earliest-minus-top corrected time (ps, <= 0) and whether a candidate is earlier."""
    out = []
    for s in (1, 2):
        side = {"q_top": features.column(f"q_top_{s}"), "q_time": features.column(f"q_time_{s}"),
                "doi_ratio": features.column(f"doi_ratio_{s}"), "kev": features.column(f"energy_kev_{s}")}
        offset, acts, _ = timestamp.side_offset(features.column(f"time_channel_{s}"), features.candidates[:, s - 1],
                                                side, skew, walk, "top_earliest_mean")
        out.append((2.0 * offset, acts))
    return out


class RuleDiagnostics:
    """FR-7 histograms per rule table x declared position (``groups``: {position: acquisition indices})."""

    def __init__(self, tables, groups, edges):
        self.tables, self.groups = tables, list(groups)
        self.group_of = {k: g for g, index in enumerate(groups.values()) for k in index}
        self.edges = np.asarray(edges, np.float64)
        n_g, n_o, bins = len(self.groups), len(OFFSET_EDGES_PS), len(self.edges) - 1
        self.lead = {name: np.zeros((n_g, len(LEAD_EDGES_PS) - 1)) for name in tables}
        self.zero = {name: np.zeros(n_g, np.int64) for name in tables}
        self.below = {name: np.zeros(n_g, np.int64) for name in tables}
        self.before = {name: np.zeros((n_g, n_o, bins)) for name in tables}
        self.after = {name: np.zeros((n_g, n_o, bins)) for name in tables}
        self.minimodules = {name: {} for name in tables}

    def add(self, index, features, near, tof, after):
        """``near``: near-source events of the batch; ``tof``: their geometric TOF; ``after``: {table: corrected
        dt of the batch under its rule}."""
        g, bins = self.group_of[index], len(self.edges) - 1
        for name, table in self.tables.items():
            before, _, _ = iw.correct_features(features, table.skew, table.walk, policy="drop", rule="top",
                                               mismatch=True)
            leads = side_leads(features, table.skew, table.walk)
            for lead, acts in leads:
                lead = lead[near]
                self.zero[name][g] += int(np.count_nonzero(lead == 0))
                self.below[name][g] += int(np.count_nonzero(lead < LEAD_EDGES_PS[0]))
                inside = (lead < 0) & (lead >= LEAD_EDGES_PS[0])
                k = np.clip(np.searchsorted(LEAD_EDGES_PS, lead[inside], side="right") - 1, 0, len(LEAD_EDGES_PS) - 2)
                self.lead[name][g] += np.bincount(k, minlength=len(LEAD_EDGES_PS) - 1)
            event = -np.minimum(leads[0][0], leads[1][0])[near]
            o = np.where(event > 0, np.searchsorted(OFFSET_EDGES_PS, event, side="left"), 0)
            sign = np.where(leads[0][0] <= leads[1][0], 1.0, -1.0)[near]      # the leading side's late top > 0
            for store, values in ((self.before, before), (self.after, after[name])):
                b = np.floor((sign * (values[near] - tof) - self.edges[0]) / (self.edges[1] - self.edges[0]))
                ok = np.isfinite(b) & (b >= 0) & (b < bins)
                store[name][g] += np.bincount(o[ok] * bins + b[ok].astype(np.int64),
                                              minlength=len(OFFSET_EDGES_PS) * bins).reshape(len(OFFSET_EDGES_PS), bins)
            counts = self.minimodules[name]
            for s, (_, acts) in zip((1, 2), leads):
                key = features.column(f"sm_{s}")[near].astype(np.int64) * 65536 + features.column(f"mm_{s}")[near]
                acting = acts[near]
                keys, sides = np.unique(key, return_counts=True)
                hits = np.unique(key[acting], return_counts=True)
                hit = dict(zip(hits[0].tolist(), hits[1].tolist()))
                for k, n in zip(keys.tolist(), sides.tolist()):
                    entry = counts.setdefault(k, [0, 0])
                    entry[0] += n
                    entry[1] += hit.get(k, 0)

    def _bins(self, before, after):
        out = []
        for label, b, a in zip(offset_labels(), before, after):
            fits = {key: core._public(core.fit_histogram(h, self.edges, crossings=core.local_crossings))
                    for key, h in (("before", b), ("after", a))}
            out.append({"bin": label, "events": int(a.sum()), **fits})
        return out

    def _group(self, name, select):
        lead = select(self.lead[name])
        return {"lead_counts": lead.astype(int).tolist(), "top_earliest_sides": int(select(self.zero[name])),
                "below_range_sides": int(select(self.below[name])),
                "offset_bins": self._bins(select(self.before[name]), select(self.after[name]))}

    def summary(self):
        out = {}
        for name, table in self.tables.items():
            rows = [{"sm": k // 65536, "mm": k % 65536, "sides": n, "acting": a, "share": a / n}
                    for k, (n, a) in sorted(self.minimodules[name].items())]
            out[name] = {"rule": table.rule, "lead_edges_ps": LEAD_EDGES_PS.tolist(),
                         "offset_edges_ps": [float(e) if np.isfinite(e) else None for e in OFFSET_EDGES_PS],
                         "pooled": self._group(name, lambda h: h.sum(axis=0)),
                         "per_position": {p: self._group(name, lambda h, g=g: h[g]) for g, p in enumerate(self.groups)},
                         "per_minimodule": rows}
        return out


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def write_plots(output, diagnostics):
    """The three PNGs per rule table next to the CTR report ``output``; returns their paths."""
    output, paths = Path(output), []
    for name, entry in diagnostics.items():
        stem = output.with_name(f"{output.stem}_{_safe(name)}")
        prefix = os.path.commonprefix(list(entry["per_position"])) if len(entry["per_position"]) > 1 else ""
        groups = {name[len(prefix):] or name: group for name, group in entry["per_position"].items()}
        groups["pooled"] = entry["pooled"]                                # pooled drawn last, in black

        def colour(k, label):
            return "k" if label == "pooled" else f"C{k}"
        edges = np.asarray(entry["lead_edges_ps"])
        figure, ax = plt.subplots(figsize=(8, 4.8))
        for k, (label, group) in enumerate(groups.items()):
            sides = sum(group["lead_counts"]) + group["top_earliest_sides"] + group["below_range_sides"]
            ax.stairs(group["lead_counts"], edges, lw=1.4 if label == "pooled" else 0.8, color=colour(k, label),
                      label=f"{label}: top earliest {group['top_earliest_sides'] / max(sides, 1):.1%}, "
                            f"< {edges[0]:g} ps {group['below_range_sides']}")
        ax.set(yscale="log", xlabel="earliest - top corrected time per side (ps)", ylabel="sides per bin",
               title=f"{name} ({entry['rule']}): lead of the earliest candidate")
        ax.legend(fontsize=7, title=prefix.strip() or None, title_fontsize=7)
        figure.tight_layout()
        paths.append(stem.with_name(stem.name + "_lead.png"))
        figure.savefig(paths[-1], dpi=120)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        labels = [b["bin"] for b in entry["pooled"]["offset_bins"]]
        x = np.arange(len(labels))
        for ax, key, ylabel in ((axes[0], "model_free_fwhm_ps", "model-free FWHM (ps)"),
                                (axes[1], "mean_ps", "Gaussian mean, leading side late > 0 (ps)")):
            for k, (label, group) in enumerate(groups.items()):
                for when, style in (("before", "--"), ("after", "-")):
                    y = [b[when].get(key) if b[when].get("status") == "fitted" else None for b in group["offset_bins"]]
                    ax.plot(x, [np.nan if v is None else v for v in y], style, marker="o", color=colour(k, label),
                            lw=1.6 if label == "pooled" else 0.9,
                            label=f"{label} {'top' if when == 'before' else entry['rule']}")
            events = ax.twinx()
            events.bar(x, [b["events"] for b in entry["pooled"]["offset_bins"]], color="0.85", zorder=0)
            events.set(ylabel="pooled events per bin")
            ax.set_zorder(events.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.set_xticks(x, labels, fontsize=8)
            ax.set(xlabel="event lead (the more negative side)", ylabel=ylabel)
            ax.legend(fontsize=6, ncol=2, loc="upper left", title=prefix.strip() or None, title_fontsize=6)
        axes[1].axhline(0, color="k", lw=0.6)
        figure.suptitle(f"{name}: CTR per lead bin, before (top) and after ({entry['rule']})")
        figure.tight_layout()
        paths.append(stem.with_name(stem.name + "_ctr_by_lead.png"))
        figure.savefig(paths[-1], dpi=120)
        plt.close(figure)

        rows = entry["per_minimodule"]
        sms = sorted({r["sm"] for r in rows})
        mms = sorted({r["mm"] for r in rows})
        grid = np.full((len(mms), len(sms)), np.nan)
        for r in rows:
            grid[mms.index(r["mm"]), sms.index(r["sm"])] = r["share"]
        figure, ax = plt.subplots(figsize=(max(6, 0.3 * len(sms) + 2), 4.8))
        image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=0)
        figure.colorbar(image, ax=ax, label="share of sides where the rule acts")
        ax.set_xticks(range(len(sms)), sms, fontsize=7, rotation=90)
        ax.set_yticks(range(len(mms)), mms, fontsize=7)
        ax.set(xlabel="SM", ylabel="minimodule", title=f"{name} ({entry['rule']}): rule acts per minimodule")
        figure.tight_layout()
        paths.append(stem.with_name(stem.name + "_minimodule_share.png"))
        figure.savefig(paths[-1], dpi=120)
        plt.close(figure)
    return paths
