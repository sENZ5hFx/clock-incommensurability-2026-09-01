#!/usr/bin/env python3
"""Clock-Incommensurability Class (CIC) — remaining-gap engine.

Session: 2026-09-01T13:40:00-04:00
Author: Haley Bird / autonomous Grok research agent

This is a classification heuristic over *sourced* unknowns. It is not a
measurement of nature, not a physical discovery, not peer review, not a patent.

Prior archive classes (MUST NOT be reclaimed):
  - Undersampling (2026-08-03)           — WHERE we look
  - Dark Biosphere Hypothesis (2026-05-14)
  - Deep-biosphere-as-space-prior (2026-08-14)
  - Transducer-Absence Class / TAC (2026-08-28) — missing converter; name the layer dark
  - Named-Cause Residual / NCR + Initiation-Maintenance Split (2026-08-30)
    — named cause is real but does not close the ledger

Remaining structural gap after that subtraction
-----------------------------------------------
Clock-Incommensurability Class (CIC): the unknown is not missing stuff,
not a missing converter, and not an insufficient named cause. The unknown
is that the *observation cadence cannot resolve the process cadence*, so
every named "object" is a freeze-frame of a process we have not watched.

Three operators inside CIC:

  1. CADENCE MISMATCH  — process timescale τ_p >> observation window τ_o
  2. PHASE-OBJECT COLLAPSE (POC) — a freeze-frame is promoted to an
     ontological class (a type of thing) instead of a phase of a process
  3. LOSS-OUTRUNS-CATALOG (LOC) — the change/extinction clock is faster
     than the naming clock; the dark majority is not waiting to be found,
     it is disappearing unnamed

Falsification rule
------------------
  If a time-series of the SAME object/process closes the gap → CIC for that case.
  If more spatial samples close it                     → undersampling (already claimed).
  If locating a converter closes it                    → TAC (already claimed).
  If adding more of the named cause closes it          → not NCR; CIC does not apply.
  If a discriminator already exists and has been run   → the split is settled; drop.

Scores are a heuristic. They are not measurements of nature.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/workspace/artifacts")
OUT.mkdir(parents=True, exist_ok=True)

SESSION_ID = "2026-09-01-cic-poc-loc"
TIMESTAMP = "2026-09-01T13:40:00-04:00"

# ---------------------------------------------------------------------------
# Prior classes this session is forbidden to reclaim as novelty
# ---------------------------------------------------------------------------
PRIOR_CLASSES = {
    "undersampling": "2026-08-03",
    "dark_biosphere": "2026-05-14",
    "deep_biosphere_as_space_prior": "2026-08-14",
    "TAC": "2026-08-28",
    "NCR": "2026-08-30",
    "IM_split": "2026-08-30",
}

RETAIN_DO_NOT_RECLAIM = {
    "omnitrophota",
    "hadal_microbiome",
    "magnetoreceptor",
    "hubble_tension",
    "nodule_dark_oxygen",
}


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not math.isfinite(x):
        raise ValueError(f"non-finite score: {x}")
    return max(lo, min(hi, x))


def log10_ratio(process_years: float, observe_years: float) -> float:
    """Cadence ratio in decades of mismatch. Floor at 1 day of observation."""
    if process_years <= 0 or observe_years <= 0:
        raise ValueError("timescales must be positive")
    return math.log10(process_years / observe_years)


@dataclass
class Item:
    """One sourced unknown, scored for CIC remaining surface."""

    id: str
    realm: str
    title: str
    process_years: float
    observe_years: float
    snapshot_not_series: bool
    phase_named_as_class: bool
    loss_outruns_catalog: bool
    prior_classes: list[str]
    retain: bool
    spokenness: float  # 0 = unspoken, 100 = famous unsolved
    source: str
    status: str  # known_fact | documented_gap | unproved_claim | structural_synthesis
    notes: str
    confidence: float  # 0-1 on the *numbers*, not on the class
    # optional LOC inputs
    catalog_years_to_finish: float | None = None
    loss_horizon_years: float | None = None

    cadence_log: float = field(init=False)
    cic_raw: float = field(init=False)
    poc: float = field(init=False)
    loc: float = field(init=False)
    prior_penalty: float = field(init=False)
    spoken_penalty: float = field(init=False)
    remaining: float = field(init=False)
    adversarial_flags: list[str] = field(default_factory=list)

    def score(self) -> "Item":
        if self.realm not in {
            "space",
            "earth",
            "ocean",
            "animals",
            "humans",
            "land",
        }:
            raise ValueError(f"unknown realm: {self.realm}")
        if not (0.0 <= self.spokenness <= 100.0):
            raise ValueError("spokenness must be 0-100")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be 0-1")

        self.cadence_log = log10_ratio(self.process_years, self.observe_years)
        # Map log10 ratio onto 0-100. 0 decades → 0; 6 decades (1e6×) → 100.
        cadence_score = _clamp(self.cadence_log / 6.0 * 100.0)

        snapshot = 18.0 if self.snapshot_not_series else 0.0
        self.poc = 22.0 if self.phase_named_as_class else 0.0

        self.loc = 0.0
        if self.loss_outruns_catalog:
            self.loc = 20.0
            if (
                self.catalog_years_to_finish is not None
                and self.loss_horizon_years is not None
                and self.loss_horizon_years > 0
            ):
                overrun = self.catalog_years_to_finish / self.loss_horizon_years
                self.loc = _clamp(20.0 + 8.0 * math.log10(max(overrun, 1.0)))

        self.cic_raw = _clamp(0.55 * cadence_score + snapshot + self.poc + self.loc)

        # Prior-class penalty: each overlapping prior class cuts remaining surface.
        n_prior = len([p for p in self.prior_classes if p in PRIOR_CLASSES])
        self.prior_penalty = min(0.85, 0.28 * n_prior + (0.45 if self.retain else 0.0))

        # Famous unsolved problems are downranked (they are spoken).
        self.spoken_penalty = _clamp(self.spokenness) / 100.0 * 0.35

        remaining = self.cic_raw * (1.0 - self.prior_penalty) * (1.0 - self.spoken_penalty)
        remaining *= 0.5 + 0.5 * self.confidence  # distrust weakly sourced numbers
        self.remaining = round(_clamp(remaining), 2)

        self._adversarial()
        return self

    def _adversarial(self) -> None:
        flags: list[str] = []
        if self.confidence < 0.55:
            flags.append("LOW_CONFIDENCE_NUMBERS")
        if self.retain:
            flags.append("RETAIN_DO_NOT_LOAD_BEAR")
        if n := [p for p in self.prior_classes if p in PRIOR_CLASSES]:
            flags.append("PRIOR_OVERLAP:" + ",".join(n))
        if self.status == "unproved_claim":
            flags.append("UNPROVED_DO_NOT_ASSERT")
        if self.phase_named_as_class and "NCR" in self.prior_classes:
            flags.append("POC_NEAR_NCR_CHECK_DISTINCT")
        if self.cadence_log < 2.0 and not self.loss_outruns_catalog:
            flags.append("WEAK_CADENCE_MISMATCH")
        # Timescale order-of-magnitude sanity: observation should be shorter.
        if self.observe_years >= self.process_years:
            flags.append("OBSERVE_NOT_SHORTER_THAN_PROCESS")
        self.adversarial_flags = flags


def build_catalog() -> list[Item]:
    """Sourced items. Numbers are literature values or bounded estimates.

    process_years / observe_years are *order-of-magnitude* clocks, not
    precise stopwatch readings. Confidence encodes that.
    """
    raw: list[Item] = [
        Item(
            id="lrd_black_hole_star_phase",
            realm="space",
            title="Little red dots named as a class; Aug 2026 evidence they are a transient PHASE (black-hole-star / hidden host)",
            process_years=3e8,  # LRDs prominent ~0.6–1.6 Gyr after BB, gone by z≈0
            observe_years=4.0,  # JWST era 2022–2026: years, not a time-series of one object
            snapshot_not_series=True,
            phase_named_as_class=True,
            loss_outruns_catalog=False,
            prior_classes=["NCR"],  # NCR covered class-vs-hidden-host, not BHS-as-phase
            retain=False,
            spokenness=40.0,
            source="Naidu et al. MIT 12 Aug 2026 (MoM-BH*-1 'black hole star'); ScienceDaily/NASA 17 Aug 2026 (LRDs may not be a separate population); Wikipedia LRD, 300+ observed as of 2025; MIT Physics @MIT_Physics 1 Sep 2026",
            status="unproved_claim",
            notes="Observation of LRDs is confirmed. That they ARE a new class of object is unproved. That they are a growth PHASE of SMBH assembly (black hole star) is an interpretation dated 12 Aug 2026. Population snapshots cannot watch one object evolve. Distinct from NCR (which asked whether the named cause closes the mass crisis) and from TAC (no missing converter).",
            confidence=0.72,
        ),
        Item(
            id="hubble_tension",
            realm="space",
            title="Hubble tension: local H0 ~73 vs Planck ~67.4",
            process_years=1.38e10,
            observe_years=30.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["TAC", "NCR"],
            retain=True,
            spokenness=95.0,
            source="JWST confirms local ~73 vs CMB ~67–68; Keck 5 Dec 2025 independent time-delay; H0DN / Riess SH0ES. Wikipedia Hubble's law (early vs late >5σ).",
            status="documented_gap",
            notes="RETAIN. Spoken. Already TAC item 10 and NCR retain. Not reclaimed.",
            confidence=0.88,
        ),
        Item(
            id="deep_biosphere_generation_clock",
            realm="earth",
            title="Deep-biosphere cells: generation times of tens to thousands of years; expeditions sample days",
            process_years=1e3,  # Jørgensen: tens to thousands of years between divisions
            observe_years=0.02,  # ~1-week sampling window typical of a cruise/mine visit
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["dark_biosphere", "deep_biosphere_as_space_prior"],
            retain=False,
            spokenness=35.0,
            source="Jørgensen & D'Hondt 2011 PMC3215010 (doubling as long as several thousand years); Jørgensen 2016 (mean generation tens to thousands of years); Wikipedia Deep biosphere (turnover hundreds of thousands of years; cells live thousands of years before dividing). Biomass ~15% of Earth total (Bar-On 2018).",
            status="documented_gap",
            notes="Dark Biosphere Hypothesis claimed the EXISTENCE and networked architecture of subsurface life. This item is the CLOCK: a 3-year programme is one frame of a millennial film. Distinct structure.",
            confidence=0.80,
        ),
        Item(
            id="geomagnetic_reversal_unwatched",
            realm="earth",
            title="Geomagnetic reversal: process kyr-scale; never watched; simulations decades from realistic",
            process_years=5e3,  # reversal duration ~1–10 kyr
            observe_years=50.0,  # satellite era + historical
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["NCR"],
            retain=False,
            spokenness=55.0,
            source="Roger Fu, Harvard Gazette, 7 Apr 2026 (NCR archive). Last reversal ~780 kyr ago.",
            status="documented_gap",
            notes="NCR already ranked this highest remaining (53.22) as initiation-maintenance. CIC overlap is real. Heavy prior penalty applied. Kept only as a negative control.",
            confidence=0.70,
        ),
        Item(
            id="inner_core_state_as_class",
            realm="earth",
            title="Inner core named 'solid'; 2025 evidence it is transforming",
            process_years=1e8,
            observe_years=60.0,
            snapshot_not_series=True,
            phase_named_as_class=True,
            loss_outruns_catalog=False,
            prior_classes=["NCR", "undersampling"],
            retain=False,
            spokenness=40.0,
            source="USC 10 Feb 2025 (NCR archive); Kola Superdeep 12.3 km vs 6371 km radius (~0.19%).",
            status="documented_gap",
            notes="POC example (solid-as-class) but already NCR. Downranked.",
            confidence=0.68,
        ),
        Item(
            id="kola_contact_fraction",
            realm="earth",
            title="Direct sample 0.19% of the way to the core",
            process_years=4.5e9,
            observe_years=50.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["undersampling", "NCR"],
            retain=False,
            spokenness=50.0,
            source="Kola Superdeep ~12.3 km. Earth's radius 6371 km.",
            status="known_fact",
            notes="This is spatial undersampling, already claimed 2026-08-03 and ranked in NCR. Included as a negative control so the engine can show it is NOT remaining CIC novelty.",
            confidence=0.95,
        ),
        Item(
            id="seafloor_mapping_residual",
            realm="ocean",
            title="Global seafloor mapped to modern standards: 28.7% (Apr 2026)",
            process_years=1e4,
            observe_years=20.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["undersampling", "NCR"],
            retain=False,
            spokenness=60.0,
            source="NOAA Ocean Exploration fact, last updated 20 Apr 2026, citing Seabed 2030.",
            status="known_fact",
            notes="Spatial undersampling. Updated number, not a new class. Negative control.",
            confidence=0.93,
        ),
        Item(
            id="deep_seafloor_unseen",
            realm="ocean",
            title="Deep seafloor visually observed: <0.001%",
            process_years=1e4,
            observe_years=50.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["undersampling"],
            retain=False,
            spokenness=55.0,
            source="NOAA citing Science Advances 10.1126/sciadv.adp8602. Deep ocean >90% of ocean volume.",
            status="known_fact",
            notes="Undersampling. The CIC remainder is only that a visual pass is also a snapshot of a changing benthos — weak, downranked.",
            confidence=0.90,
        ),
        Item(
            id="nodule_dark_oxygen",
            realm="ocean",
            title="Nodule 'dark oxygen' (Sweetman 2024) — contested, editor's note",
            process_years=1e6,
            observe_years=2.0,
            snapshot_not_series=True,
            phase_named_as_class=True,
            loss_outruns_catalog=False,
            prior_classes=["TAC", "NCR", "dark_biosphere"],
            retain=True,
            spokenness=80.0,
            source="Sweetman et al. Nature Geoscience 17, 737–744 (2024); Nature Geoscience Editor's Note 8 Apr 2026; Downes et al. Frontiers in Marine Science 2025 10.3389/fmars.2025.1721853; Live Science 19 Mar 2026 (thermodynamics critique); DORI landers 2026 pending.",
            status="unproved_claim",
            notes="RETAIN. Do not load-bear. Homonym collapse still active. Stronger quieter phenomenon is crustal O2 (Ruff), itself mechanistically unproved.",
            confidence=0.40,
        ),
        Item(
            id="crustal_radiolytic_oxygen",
            realm="earth",
            title="Crustal dark oxygen in 1.2-Gyr brine (Moab Khotsong) — observed; biology vs radiolysis unproved",
            process_years=1.2e9,
            observe_years=3.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["TAC", "dark_biosphere"],
            retain=False,
            spokenness=30.0,
            source="New Yorker, James Dinneen, 13 Aug 2026 (Ruff; 0.09 µmol/L O2; NASA OxyMoRon funding 2025). Mechanism split unproved.",
            status="documented_gap",
            notes="TAC already owns the missing converter (biology vs radioactivity). CIC remainder: we sample a 1.2 Gyr isolated brine in a 3-year grant. Partial overlap, penalized.",
            confidence=0.74,
        ),
        Item(
            id="aerobe_before_photosynthesis",
            realm="earth",
            title="Named sequence reversed: some aerobes exist in genomes BEFORE photosynthesis",
            process_years=3.5e9,
            observe_years=20.0,  # genomic comparative window of the research programme
            snapshot_not_series=True,
            phase_named_as_class=True,
            loss_outruns_catalog=False,
            prior_classes=[],
            retain=False,
            spokenness=25.0,
            source="Murali et al., Science, DOI 10.1126/science.adp1853 (cited in New Yorker 13 Aug 2026). Named public story is photosynthesis → O2 → aerobes (GOE ~2.4 Gya). Genomic clock disagrees for some lineages. Candidate energy: radiolytic O2 (Ruff), itself unproved as the ancestral source.",
            status="structural_synthesis",
            notes="This is CIC's cousin: two clocks (textbook sequence vs genomic sequence) disagree. Do not load-bear on Sweetman nodules. Load-bear only on the genomic result. The ancestral energy source is unproved.",
            confidence=0.70,
        ),
        Item(
            id="hadal_microbiome",
            realm="ocean",
            title="Hadal microbiome: 7,564 species, ~90% new",
            process_years=1e6,
            observe_years=5.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=True,
            catalog_years_to_finish=200.0,
            loss_horizon_years=80.0,
            prior_classes=["undersampling", "TAC"],
            retain=True,
            spokenness=40.0,
            source="Xiao et al., Cell 2025 (NCR/TAC retain).",
            status="documented_gap",
            notes="RETAIN. Not reclaimed.",
            confidence=0.78,
        ),
        Item(
            id="insect_dark_taxa_loc",
            realm="animals",
            title="Insect dark taxa: ~80% unnamed; description clock ~450 yr; unnamed species are more extinction-prone",
            process_years=450.0,  # years to finish catalog at current insect description rate
            observe_years=1.0,  # one year of taxonomic output
            snapshot_not_series=False,  # the catalog is a series — but too slow
            phase_named_as_class=False,
            loss_outruns_catalog=True,
            catalog_years_to_finish=450.0,
            loss_horizon_years=100.0,  # 21st-century extinction/land-use horizon (order-of-magnitude)
            prior_classes=[],
            retain=False,
            spokenness=20.0,
            source="Lehmitz et al. npj Biodivers. 2025 (s44185-025-00108-3): 5.5 million insects, ~1 million named, >80% undescribed, ~10,000 insects described/year, ~450 years; 20 families = 50% of flying insect richness (dark taxa). Stork/Griffith 26 Aug 2024: undescribed insects smaller, rarer, more extinction-prone. Li et al. 2025 PMC12680044: >400 years to describe a small fraction at current rates. Vox 2 Sep 2025: dark taxa as animal-kingdom 'dark matter'.",
            status="documented_gap",
            notes="Flagship LOC. Not in TAC or NCR as a primary claim. The dark majority of animals is not waiting in a drawer. It is smaller, rarer, and going extinct unnamed. Distinct from undersampling (we DO sample meadows; we do not name what we sample on a clock that matches loss).",
            confidence=0.86,
        ),
        Item(
            id="cryptic_species_morph_clock",
            realm="animals",
            title="Each morphological insect species conceals ~3.1 cryptic species",
            process_years=250.0,  # Linnaean morphological clock, centuries
            observe_years=15.0,  # DNA-barcode era
            snapshot_not_series=False,
            phase_named_as_class=True,
            loss_outruns_catalog=True,
            catalog_years_to_finish=450.0,
            loss_horizon_years=100.0,
            prior_classes=[],
            retain=False,
            spokenness=15.0,
            source="Lehmitz et al. 2025 citing cryptic-species factor 3.1. Morphological 'species' is a freeze-frame of a genetic complex.",
            status="documented_gap",
            notes="POC: Linnaean species-as-class collapses a genetic process. Complements insect LOC.",
            confidence=0.75,
        ),
        Item(
            id="magnetoreceptor",
            realm="animals",
            title="Magnetoreceptor cell unlocated after ~60 years",
            process_years=60.0,
            observe_years=60.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["TAC", "NCR"],
            retain=True,
            spokenness=50.0,
            source="Hore 2026; Nordmann et al. J. Exp. Biol. 10 Apr 2025 (ruling-hypothesis trap). TAC item 2.",
            status="documented_gap",
            notes="RETAIN. TAC owns this (missing converter). CIC does not apply cleanly: observe_years ≈ process_years of the research programme.",
            confidence=0.80,
        ),
        Item(
            id="dark_genome_two_clocks",
            realm="humans",
            title="Dark genome: biochemical-assay clock vs evolutionary-selection clock disagree on 'function'",
            process_years=3e5,  # recent-human evolutionary window (order of magnitude)
            observe_years=15.0,  # ENCODE/functional-genomics assay era
            snapshot_not_series=True,
            phase_named_as_class=True,
            loss_outruns_catalog=False,
            prior_classes=["NCR"],
            retain=False,
            spokenness=45.0,
            source="~98% non-coding. IISc Connect Apr 2026 (Graur: ENCODE 80% 'functional' would imply ~70% invulnerable to mutation — mutational clock contradicts assay clock). Prabakaran 2025 'dark proteins'. Function of most of the genome still unsettled.",
            status="documented_gap",
            notes="NCR listed 'dark genome function residual' (the name moved junk → dark → regulatory; ledger did not close). CIC remainder is specifically TWO-CLOCK disagreement about what 'function' means. Partial overlap; penalized.",
            confidence=0.77,
        ),
        Item(
            id="anesthesia_maintenance",
            realm="humans",
            title="Anesthesia: initiation known, maintenance unagreed after ~180 years",
            process_years=4.0 / 8760.0,  # hours of a surgery, in years
            observe_years=180.0,
            snapshot_not_series=False,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["NCR", "IM_split"],
            retain=False,
            spokenness=40.0,
            source="The Conversation 16 May 2024 (NCR archive). Canonical IM-split.",
            status="documented_gap",
            notes="Negative control: observe_years >> process_years, so CIC cadence is inverted. This is NCR/IM, not CIC.",
            confidence=0.85,
        ),
        Item(
            id="consciousness_discriminator",
            realm="humans",
            title="No discriminator for machine (or animal) consciousness",
            process_years=1.0,
            observe_years=30.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["TAC", "NCR"],
            retain=False,
            spokenness=98.0,
            source="Thomas Fel, Harvard Gazette 7 Apr 2026 (NCR). Hard problem is the most spoken unproved thing.",
            status="documented_gap",
            notes="Spoken. Last on an unspoken list. Not CIC.",
            confidence=0.60,
        ),
        Item(
            id="omnitrophota",
            realm="land",
            title="Omnitrophota: globally common, almost uncultured (lab clock ≠ environmental clock)",
            process_years=1e2,
            observe_years=30.0,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["TAC"],
            retain=True,
            spokenness=25.0,
            source="DRI / Nature Microbiology Omnitrophota (inferred from env. DNA ~30 years; still uncultured). TAC item 1.",
            status="documented_gap",
            notes="RETAIN. TAC owns missing lab transducer. CIC remainder (lab cadence ≠ environmental cadence) is real but secondary.",
            confidence=0.78,
        ),
        Item(
            id="soil_prokaryote_function",
            realm="land",
            title="Functions of most soil-relevant prokaryotic species remain unknown",
            process_years=10.0,  # seasonal-to-decadal soil process
            observe_years=0.2,  # typical sampling/assay campaign
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=True,
            catalog_years_to_finish=200.0,
            loss_horizon_years=50.0,
            prior_classes=["undersampling"],
            retain=False,
            spokenness=20.0,
            source="Lehmitz et al. 2025: functions of most soil-relevant prokaryotic species unknown; barrier to linking biodiversity and ecosystem services. Cultivation <1% of microorganisms (Alanzi 2025 JMB); 87–99% uncultured across habitats (preprints 2025, 1,046 studies).",
            status="documented_gap",
            notes="Land's CIC: we stand on it, we name it known, the functional catalog is a snapshot of an uncultured majority. Distinct from Omnitrophota (one phylum, TAC).",
            confidence=0.73,
        ),
        Item(
            id="microbial_uncultured_fraction",
            realm="land",
            title="87–99% of microbial diversity remains uncultured across habitats",
            process_years=1e2,
            observe_years=0.1,
            snapshot_not_series=True,
            phase_named_as_class=False,
            loss_outruns_catalog=False,
            prior_classes=["undersampling", "TAC"],
            retain=False,
            spokenness=40.0,
            source="Statistical analysis of 1,046 studies / eight habitats, preprint 3 Jul 2025: 87–99% uncultured. Lloyd et al. 2018 ~80% uncultured taxa; Nayfach 2021 85% of phylogenetic diversity uncultured.",
            status="known_fact",
            notes="Known fact used as a clock-context number, not a new class. Partial TAC overlap (no lab converter).",
            confidence=0.82,
        ),
    ]
    return [it.score() for it in raw]


def remaining_surface(items: list[Item]) -> list[Item]:
    """Drop retain-only rows from the 'remaining novelty' ranking but keep them in the full table."""
    return sorted(items, key=lambda x: x.remaining, reverse=True)


def novelty_guard(items: list[Item]) -> dict[str, Any]:
    """Adversarial second function: reject any 'primary insight' that reclaims a prior class.

    Inputs: scored items.
    Outputs: allowed primary insight, rejected reclaimings, discriminator tests.
    """
    ranked = [i for i in remaining_surface(items) if not i.retain]
    rejected = []
    allowed = []
    for it in ranked:
        reasons = []
        if it.retain:
            reasons.append("on RETAIN list")
        if "RETAIN_DO_NOT_LOAD_BEAR" in it.adversarial_flags:
            reasons.append("do not load-bear")
        if it.remaining < 12.0:
            reasons.append(f"remaining surface too small ({it.remaining})")
        if "OBSERVE_NOT_SHORTER_THAN_PROCESS" in it.adversarial_flags:
            reasons.append("not CIC: observation window is not shorter than process")
        if it.id in RETAIN_DO_NOT_RECLAIM:
            reasons.append("explicit retain id")
        # If prior penalty wiped most of the score, it is a reclaim.
        if it.prior_penalty >= 0.50 and it.poc == 0 and not it.loss_outruns_catalog:
            reasons.append("prior-class overlap dominates; no POC/LOC remainder")
        if reasons:
            rejected.append({"id": it.id, "remaining": it.remaining, "reasons": reasons})
        else:
            allowed.append(it)

    # Discriminators that would settle CIC vs TAC vs NCR for the top allowed items.
    discriminators: list[dict[str, str]] = []
    for it in allowed[:5]:
        discriminators.append(
            {
                "id": it.id,
                "realm": it.realm,
                "cic_test": (
                    "Obtain a time-series of the SAME object/process at intervals "
                    f"shorter than τ_p ({it.process_years:g} yr). If the named class "
                    "dissolves into a phase, CIC/POC is confirmed."
                ),
                "tac_test": (
                    "Physically remove or locate the hypothesized converter. If the "
                    "effect survives removal, TAC applies, not CIC."
                ),
                "ncr_test": (
                    "Add more of the named cause. If the residual closes, NCR is wrong "
                    "for this case. If the named cause is removed and the phenomenon "
                    "continues, the named cause was never the ledger."
                ),
                "undersampling_test": (
                    "Increase spatial coverage at the same cadence. If the gap closes "
                    "without a longer time-series, this was undersampling, not CIC."
                ),
            }
        )

    primary = allowed[0] if allowed else None
    return {
        "session_id": SESSION_ID,
        "timestamp": TIMESTAMP,
        "n_items": len(items),
        "n_allowed": len(allowed),
        "n_rejected": len(rejected),
        "primary_id": None if primary is None else primary.id,
        "primary_title": None if primary is None else primary.title,
        "primary_remaining": None if primary is None else primary.remaining,
        "allowed_ids": [a.id for a in allowed],
        "rejected": rejected,
        "discriminators": discriminators,
        "class_name": "Clock-Incommensurability Class (CIC)",
        "operators": ["cadence_mismatch", "phase_object_collapse", "loss_outruns_catalog"],
        "not_claimed": [
            "Not a new physical mechanism.",
            "Not a proof that little red dots are black hole stars.",
            "Not a proof that Sweetman nodule oxygen is real or fake.",
            "Not a proof that radiolytic O2 is the ancestral aerobic energy source.",
            "Not patent-ready, not peer-reviewed, not a discovery of nature.",
            "Scores are a classification heuristic over sourced gaps.",
        ],
    }


def write_outputs(items: list[Item], guard: dict[str, Any]) -> dict[str, str]:
    full = []
    for it in remaining_surface(items):
        row = asdict(it)
        full.append(row)

    json_path = OUT / "cic_scores.json"
    json_path.write_text(
        json.dumps(
            {
                "session_id": SESSION_ID,
                "timestamp": TIMESTAMP,
                "class": "Clock-Incommensurability Class (CIC)",
                "prior_classes_subtracted": PRIOR_CLASSES,
                "retain_do_not_reclaim": sorted(RETAIN_DO_NOT_RECLAIM),
                "items": full,
                "guard": guard,
            },
            indent=2,
        )
        + "\n"
    )

    csv_path = OUT / "cic_scores.csv"
    fields = [
        "remaining",
        "cic_raw",
        "cadence_log",
        "poc",
        "loc",
        "prior_penalty",
        "spokenness",
        "confidence",
        "realm",
        "id",
        "retain",
        "status",
        "title",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for it in remaining_surface(items):
            w.writerow({k: getattr(it, k) for k in fields})

    return {"json": str(json_path), "csv": str(csv_path)}


def plot_rankings(items: list[Item]) -> str:
    ranked = remaining_surface(items)
    labels = [f"{it.realm[:2].upper()} · {it.id.replace('_', ' ')}" for it in ranked]
    remaining = [it.remaining for it in ranked]
    colors = []
    for it in ranked:
        if it.retain:
            colors.append("#6b7280")
        elif it.loss_outruns_catalog and it.remaining >= 20:
            colors.append("#b45309")
        elif it.phase_named_as_class and it.remaining >= 20:
            colors.append("#7c3aed")
        else:
            colors.append("#0f766e")

    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    y = np.arange(len(ranked))
    ax.barh(y, remaining, color=colors, edgecolor="#111827", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color="#e5e7eb")
    ax.invert_yaxis()
    ax.set_xlabel("Remaining CIC surface (heuristic, after prior-class subtraction)", color="#e5e7eb")
    ax.set_title(
        "Clock-Incommensurability Class — remaining surface\n"
        f"{TIMESTAMP}  ·  grey = RETAIN (do not reclaim)",
        color="#f9fafb",
        loc="left",
        fontsize=13,
        pad=12,
    )
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#9ca3af")
    ax.grid(axis="x", color="#1f2937", linestyle="--", linewidth=0.6)
    # legend
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#b45309", label="LOC (loss outruns catalog)"),
            Patch(color="#7c3aed", label="POC (phase named as class)"),
            Patch(color="#0f766e", label="cadence mismatch"),
            Patch(color="#6b7280", label="RETAIN / already claimed"),
        ],
        facecolor="#111827",
        edgecolor="#374151",
        labelcolor="#e5e7eb",
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    path = OUT / "cic_ranking.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_clock_plane(items: list[Item]) -> str:
    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    fig.patch.set_facecolor("#0b1020")
    ax.set_facecolor("#0b1020")
    realm_color = {
        "space": "#60a5fa",
        "earth": "#f59e0b",
        "ocean": "#22d3ee",
        "animals": "#34d399",
        "humans": "#f472b6",
        "land": "#a3e635",
    }
    for it in items:
        x = math.log10(it.observe_years)
        y = math.log10(it.process_years)
        size = 40 + 9 * it.remaining
        marker = "s" if it.retain else ("D" if it.loss_outruns_catalog else "o")
        ax.scatter(
            x,
            y,
            s=size,
            c=realm_color[it.realm],
            marker=marker,
            alpha=0.9,
            edgecolors="#111827",
            linewidths=0.5,
            zorder=3,
        )
        if it.remaining >= 18 or it.id in {
            "insect_dark_taxa_loc",
            "lrd_black_hole_star_phase",
            "deep_biosphere_generation_clock",
            "aerobe_before_photosynthesis",
            "soil_prokaryote_function",
        }:
            ax.annotate(
                it.id.replace("_", "\n"),
                (x, y),
                textcoords="offset points",
                xytext=(7, 6),
                fontsize=6.5,
                color="#e5e7eb",
            )

    # diagonal τ_p = τ_o
    xs = np.linspace(-3.2, 2.6, 50)
    ax.plot(xs, xs, color="#6b7280", linestyle="--", linewidth=1, label="τ_process = τ_observe")
    ax.fill_between(xs, xs, xs + 6, color="#7c3aed", alpha=0.08, label="CIC half-plane (process slower)")
    ax.set_xlabel("log10 observation window (years)", color="#e5e7eb")
    ax.set_ylabel("log10 process timescale (years)", color="#e5e7eb")
    ax.set_title(
        "Clock plane — above the diagonal, we hold a freeze-frame",
        color="#f9fafb",
        loc="left",
    )
    for spine in ax.spines.values():
        spine.set_color("#374151")
    ax.tick_params(colors="#9ca3af")
    ax.grid(color="#1f2937", linestyle=":", linewidth=0.6)
    ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#e5e7eb", fontsize=8)
    fig.tight_layout()
    path = OUT / "cic_clock_plane.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def print_tables(items: list[Item], guard: dict[str, Any]) -> None:
    print("=" * 88)
    print(f"CIC remaining-gap engine  ·  {TIMESTAMP}  ·  {SESSION_ID}")
    print("=" * 88)
    print(f"{'remain':>7} {'raw':>6} {'logR':>5} {'POC':>4} {'LOC':>4} {'pen':>5} {'conf':>4}  realm     id")
    print("-" * 88)
    for it in remaining_surface(items):
        flag = " RETAIN" if it.retain else ""
        print(
            f"{it.remaining:7.2f} {it.cic_raw:6.1f} {it.cadence_log:5.1f} "
            f"{it.poc:4.0f} {it.loc:4.0f} {it.prior_penalty:5.2f} {it.confidence:4.2f}  "
            f"{it.realm:<8} {it.id}{flag}"
        )
        if it.adversarial_flags:
            print(f"         flags: {', '.join(it.adversarial_flags)}")
    print("-" * 88)
    print("GUARD")
    print(json.dumps({k: guard[k] for k in [
        "primary_id", "primary_title", "primary_remaining",
        "allowed_ids", "n_allowed", "n_rejected", "class_name", "operators",
    ]}, indent=2))
    print("Rejected:")
    for r in guard["rejected"]:
        print(f"  - {r['id']} ({r['remaining']}): {'; '.join(r['reasons'])}")
    print("Discriminators (top allowed):")
    for d in guard["discriminators"]:
        print(f"  [{d['realm']}] {d['id']}")
        print(f"     CIC: {d['cic_test']}")


def main() -> None:
    items = build_catalog()
    guard = novelty_guard(items)
    paths = write_outputs(items, guard)
    p1 = plot_rankings(items)
    p2 = plot_clock_plane(items)
    print_tables(items, guard)
    print("Wrote:", paths, p1, p2)
    # Machine-readable summary for the session log
    summary_path = OUT / "cic_summary.json"
    top_allowed = [i for i in remaining_surface(items) if i.id in guard["allowed_ids"]][:7]
    summary_path.write_text(
        json.dumps(
            {
                "session_id": SESSION_ID,
                "timestamp": TIMESTAMP,
                "primary_insight": (
                    "Clock-Incommensurability Class: across realms the remaining unknown "
                    "after subtracting undersampling, TAC, and NCR is that observation "
                    "cadence cannot resolve process cadence, so named classes are freeze-frames. "
                    f"Highest remaining allowed surface: {guard['primary_id']} "
                    f"({guard['primary_remaining']})."
                ),
                "top_allowed": [
                    {
                        "id": i.id,
                        "realm": i.realm,
                        "remaining": i.remaining,
                        "title": i.title,
                        "source": i.source,
                        "status": i.status,
                    }
                    for i in top_allowed
                ],
                "paths": {**paths, "ranking_png": p1, "clock_png": p2},
            },
            indent=2,
        )
        + "\n"
    )
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
