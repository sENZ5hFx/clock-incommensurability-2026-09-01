#!/usr/bin/env python3
"""Iteration on CIC: clock-kind split and recalibrated remaining surface.

The first engine treated 'age of life' as τ_process. That inflates every
phylogenetic item. This function classifies each item into a clock KIND,
then rescores only with the kind that actually applies.

Kinds
-----
  cadence   — same object/process, τ_p >> τ_o (true freeze-frame)
  two_clock — two legitimate metrologies answer the same question differently
  loc       — naming/catalog clock slower than loss/extinction clock
  poc       — a snapshot was promoted to an ontological class
  control   — negative control / prior-class reclaim; remaining must be ~0

This is the missing capability the first function did not cover.
"""

from __future__ import annotations

import json
from pathlib import Path

IN = Path("/workspace/artifacts/cic_scores.json")
OUT = Path("/workspace/artifacts/cic_recalibrated.json")

# Manual kind assignment — the iteration's actual scientific work.
# Cadence_log from the first pass is kept as a FEATURE, not as the score.
KIND = {
    "aerobe_before_photosynthesis": ("two_clock", "poc"),
    "lrd_black_hole_star_phase": ("poc", "cadence"),
    "cryptic_species_morph_clock": ("poc", "loc"),
    "insect_dark_taxa_loc": ("loc",),
    "dark_genome_two_clocks": ("two_clock", "poc"),
    "soil_prokaryote_function": ("loc", "cadence"),
    "inner_core_state_as_class": ("control",),  # NCR reclaim
    "kola_contact_fraction": ("control",),
    "crustal_radiolytic_oxygen": ("control",),  # TAC converter
    "deep_seafloor_unseen": ("control",),  # undersampling
    "deep_biosphere_generation_clock": ("cadence",),  # real CIC but prior overlap
    "geomagnetic_reversal_unwatched": ("control",),  # NCR
    "microbial_uncultured_fraction": ("control",),
    "seafloor_mapping_residual": ("control",),
    "hadal_microbiome": ("control",),
    "nodule_dark_oxygen": ("control",),
    "hubble_tension": ("control",),
    "omnitrophota": ("control",),
    "consciousness_discriminator": ("control",),
    "magnetoreceptor": ("control",),
    "anesthesia_maintenance": ("control",),
}

# Recalibrated remaining: start from first-pass remaining, then:
#   control → 0 (reclaim)
#   cadence with heavy prior overlap (dark_biosphere) → * 0.45
#   two_clock items: replace inflated cadence with a bounded two-clock score
#   loc / poc keep remaining if not control


def two_clock_score(confidence: float, spokenness: float, prior_n: int) -> float:
    """Two valid clocks, incompatible closures. Not cadence.

    Bounded 0-55 so a 3.5 Gyr evolutionary τ cannot dominate the ranking.
    """
    base = 48.0
    base *= 0.5 + 0.5 * confidence
    base *= 1.0 - 0.25 * min(prior_n, 2)
    base *= 1.0 - 0.30 * (spokenness / 100.0)
    return round(max(0.0, min(55.0, base)), 2)


def loc_score(item: dict) -> float:
    """Prefer the first-pass LOC-aware remaining, bounded."""
    return round(min(60.0, float(item["remaining"])), 2)


def poc_cadence_score(item: dict) -> float:
    return round(min(58.0, float(item["remaining"])), 2)


def main() -> None:
    data = json.loads(IN.read_text())
    items = data["items"]
    by_id = {i["id"]: i for i in items}

    missing = set(by_id) - set(KIND)
    extra = set(KIND) - set(by_id)
    if missing or extra:
        raise SystemExit(f"kind map mismatch missing={missing} extra={extra}")

    rows = []
    for i in items:
        kinds = list(KIND[i["id"]])
        prior_n = len(i.get("prior_classes") or [])
        if "control" in kinds:
            rec = 0.0
            reason = "control/reclaim — zero remaining novelty"
        elif kinds == ["loc"] or kinds[0] == "loc":
            rec = loc_score(i)
            reason = "LOC operator"
        elif "two_clock" in kinds:
            rec = two_clock_score(i["confidence"], i["spokenness"], prior_n)
            reason = "two-clock metrology split (cadence inflation removed)"
        elif "poc" in kinds:
            rec = poc_cadence_score(i)
            reason = "phase-object collapse ± cadence"
        elif kinds == ["cadence"]:
            rec = round(float(i["remaining"]) * 0.45, 2)
            reason = "true cadence but prior-class overlap (dark biosphere)"
        else:
            rec = round(float(i["remaining"]) * 0.3, 2)
            reason = "unclassified remainder"

        # Never load-bear retain items.
        if i.get("retain"):
            rec = 0.0
            reason = "RETAIN — zero"

        rows.append(
            {
                "id": i["id"],
                "realm": i["realm"],
                "title": i["title"],
                "kinds": kinds,
                "first_pass_remaining": i["remaining"],
                "recalibrated": rec,
                "reason": reason,
                "status": i["status"],
                "source": i["source"],
                "adversarial_flags": i["adversarial_flags"],
            }
        )

    rows.sort(key=lambda r: r["recalibrated"], reverse=True)
    allowed = [r for r in rows if r["recalibrated"] >= 12.0]
    primary = allowed[0] if allowed else None

    # Discriminators that are actually runnable (iteration fix).
    discriminators = {
        "insect_dark_taxa_loc": (
            "Compare description rate vs independently estimated extinction/"
            "range-loss rate for the same dark-taxa families (Diptera-heavy 20). "
            "If unnamed species disappear faster than they are named in a "
            "protected-area time series, LOC is confirmed. If speeding up DNA "
            "barcoding without reducing loss closes the unnamed fraction, this "
            "was undersampling, not LOC."
        ),
        "lrd_black_hole_star_phase": (
            "Do not wait 300 Myr. Use a redshift baseline of the SAME selection "
            "function: if the LRD fraction collapses into ordinary hosts + a "
            "short-lived black-hole-star phase across z, POC is confirmed. If "
            "LRDs remain a disjoint class at every z they appear, POC is wrong. "
            "Naidu 12 Aug 2026 and the Saguaro host-hiding study are the live tests."
        ),
        "cryptic_species_morph_clock": (
            "For a closed morphological morphospecies set, add nuclear + mitochondrial "
            "delimitation. If mean cryptic factor holds near 3.1 and unnamed splits "
            "are the range-restricted ones, POC+LOC stands. If barcodes collapse to "
            "morphology, the freeze-frame was already the object."
        ),
        "aerobe_before_photosynthesis": (
            "Two-clock test, not a time-series: if independent phylogenomic clocks "
            "(not the Murali tree) still place aerobic respiration before "
            "oxygenic photosynthesis, the textbook sequence is the freeze-frame. "
            "The ancestral ENERGY SOURCE of those aerobes is a separate unproved "
            "claim — do not treat crustal O2 or nodule O2 as settled."
        ),
        "dark_genome_two_clocks": (
            "Ask one question with two clocks: (assay) fraction of the genome with "
            "reproducible biochemical activity vs (selection) fraction with "
            "purifying-selection signatures. If they remain irreconcilable, "
            "two-clock stands. If a single operational definition of 'function' "
            "makes them agree, this was NCR naming, not dual metrology."
        ),
        "soil_prokaryote_function": (
            "Resample the same soil plot across seasons and years with matched "
            "metagenome+metabolome. If 'unknown function' is a seasonal phase of "
            "known taxa, cadence/POC. If new uncultured lineages keep appearing "
            "faster than functions are assigned, LOC."
        ),
    }

    out = {
        "session_id": data["session_id"],
        "timestamp": data["timestamp"],
        "iteration": "clock_kind_split",
        "why": (
            "First pass used evolutionary age as τ_process, inflating two-clock "
            "items. This split removes that artifact and zeroes reclaim/control rows."
        ),
        "primary": primary,
        "allowed": allowed,
        "all_rows": rows,
        "discriminators": discriminators,
        "structural_claim": {
            "name": "Clock-Incommensurability Class (CIC)",
            "operators": {
                "cadence_mismatch": "same object; observation window too short",
                "phase_object_collapse": "freeze-frame promoted to a type",
                "loss_outruns_catalog": "naming clock slower than loss clock",
                "two_clock": "two valid metrologies, incompatible closures",
            },
            "not": [
                "undersampling (WHERE)",
                "TAC (missing converter)",
                "NCR/IM (named cause does not close; initiation vs maintenance)",
            ],
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print("=" * 80)
    print("ITERATION — clock-kind split")
    print("=" * 80)
    print(f"{'recal':>7} {'pass1':>7}  kinds                    id")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['recalibrated']:7.2f} {r['first_pass_remaining']:7.2f}  "
            f"{','.join(r['kinds']):22}  {r['id']}"
        )
        print(f"         {r['reason']}")
    print("-" * 80)
    print("PRIMARY:", None if primary is None else f"{primary['id']}  {primary['recalibrated']}")
    print("ALLOWED:")
    for a in allowed:
        print(f"  [{a['realm']}] {a['id']}  {a['recalibrated']}  {a['kinds']}")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
