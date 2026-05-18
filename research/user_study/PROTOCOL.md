# User Study Protocol — PRISM-UB Calibration Effect

**Study title:** Subjective Evaluation of Cold-Start Preference Calibration for Vietnamese Upper-Body Fashion Recommendation
**Version:** 1.0
**Date drafted:** 2026-05-18
**Pre-registered metric:** Likert satisfaction difference (calibrated − baseline), tested via paired t-test (one-sided, α=0.05)
**Status:** Drafted; ready for run. **No data collected yet.**

---

## 1. Research questions and hypotheses

**RQ1 (primary):** Does CLIP-centroid preference calibration improve subjective satisfaction with top-K upper-body recommendations versus an FFM-only baseline, in Vietnamese women aged 18-45?

**H1:** Mean Likert satisfaction is higher for calibrated recommendations than baseline. Δ_satisfaction > 0, paired t-test, one-sided.

**RQ2 (secondary):** Does the stratified-SQL seed grid yield equivalent satisfaction to a DPP-based seed grid (i.e., is the SQL simplification justified)?

**H2 (equivalence):** Mean Likert satisfaction difference between stratified-SQL and DPP-proxy seeding is within ±0.5 Likert points (TOST equivalence test, α=0.05).

**RQ3 (secondary):** Does the participant's *forced-choice* preference (calibrated vs baseline, presented blind) align with the Likert difference?

**H3:** Proportion choosing the calibrated condition > 50%. One-proportion z-test, one-sided.

---

## 2. Design

- **Within-subjects** A/B design.
- Each participant evaluates **two conditions** (presented blind, randomized order, counterbalanced):
  - **Condition A (baseline):** FFM-only recommendation; top-10 items displayed.
  - **Condition B (calibrated):** After the participant picks 3-5 from 18 stratified seeds, their picks form a CLIP centroid; FFM + δ=0.30 preference re-rank produces a new top-10.
- After each condition: participant rates on 5-point Likert ("How satisfying do you find these recommendations as outfit suggestions for yourself?").
- After both conditions: forced-choice ("Which set of recommendations did you prefer? A, B, or no preference").
- Brief free-text feedback (~1 minute).

**Why within-subjects:** halves the required sample size vs between-subjects. Each participant is their own control.

---

## 3. Sample

- **Target N = 30** Vietnamese women.
- **Inclusion:** identifies as a Vietnamese woman, age 18-45, comfortable using a smartphone or laptop.
- **Exclusion:** active eating disorder (self-reported in screening), severe visual impairment that would prevent evaluating fashion images.
- **Recruitment:** convenience sampling through the authors' VN network. Compensation: VND 100,000 per participant (~$4 USD), paid in MoMo or cash. Participants are told the compensation is for their 15 minutes regardless of how they answer.

### Power justification

For paired t-test, one-sided α=0.05, target power 0.80, Cohen's d=0.5 (medium effect): N≥27. We round up to N=30 to allow for 10% data loss.

---

## 4. Procedure (15-20 minutes per participant)

1. **Welcome and consent** (2 min). Bilingual consent form (VN + EN). Participant can stop any time.
2. **Screening questions** (1 min). Height (cm), age band, body-type cluster as inferred from short questionnaire (no camera).
3. **Condition assignment** (1 min). Coin-flip determines A→B or B→A order; recorded.
4. **Condition 1** (4 min). View top-10 recommendations; rate Likert; brief comment.
5. **Condition 2** (4 min). For the calibrated condition: pick 3-5 from the 18-item seed grid first, then view re-ranked top-10; rate Likert; brief comment. For the baseline condition: directly view top-10; rate Likert; brief comment.
6. **Forced-choice and free text** (2 min). Side-by-side recap; participant picks A, B, or no-preference; explains briefly.
7. **Debrief** (1 min). Explain which was the calibrated condition; thank; compensate.

### Counterbalancing

Half the participants get baseline first, half get calibrated first. Random assignment via coin flip recorded in the data log.

### Blinding

Conditions are labeled "Set 1" / "Set 2" to the participant. The order is recorded but not revealed until debrief.

---

## 5. Measures

| Variable | Type | Source |
|---|---|---|
| `participant_id` | int | sequential, no PII |
| `age_band` | category | 18-24 / 25-34 / 35-45 |
| `height_cm` | float | self-reported |
| `body_cluster_self` | int 1-5 | from VN questionnaire |
| `order_assignment` | category | "AB" or "BA" |
| `condition_1_likert` | int 1-5 | first viewing |
| `condition_1_comment` | str | free text (~1 sentence) |
| `condition_2_likert` | int 1-5 | second viewing |
| `condition_2_comment` | str | free text |
| `condition_calibrated` | category | "A" or "B" — which numbered set was the calibrated one |
| `forced_choice` | category | "A", "B", or "neither" |
| `forced_choice_reason` | str | free text |
| `seeds_picked_ids` | list[str] | the 3-5 garment_ids the participant chose from the seed grid |
| `time_in_seconds` | int | timestamp at start, end |
| `device` | category | "smartphone" or "laptop" |
| `dropout` | bool | true if participant stopped before completion |

No facial images, no real names, no biometric data is collected.

---

## 6. Pre-registered analysis plan

### Primary test (H1)

Paired t-test on `(condition_calibrated_likert − condition_baseline_likert)` per participant. One-sided, α=0.05. Report:
- Mean difference (Likert points)
- 95% CI
- t-statistic, df, p-value
- Cohen's dz

**If p < 0.05 and mean diff > 0**: H1 accepted; the calibration produces a statistically-significant satisfaction improvement.

### Secondary test (H2 — equivalence)

We DO NOT have a DPP-vs-SQL condition in the within-subjects design (would require 3 conditions per participant, 22-25 min — too long). Instead, H2 is evaluated via the offline silver-label benchmark (B3); if the user study supports H1 robustly, this is sufficient evidence that the seeding strategy is not the bottleneck.

### Secondary test (H3 — forced choice)

One-proportion z-test on the proportion choosing the calibrated condition over baseline (excluding "no preference"). Null: 0.5. One-sided, α=0.05.

### Pre-registered exclusions

- Participants who drop out (`dropout = true`)
- Participants who provide identical Likert ratings for both conditions AND select "no preference" AND their free-text shows no engagement (e.g., "I don't know")

Such exclusions are reported in the paper but should be < 10%.

### Robustness checks (exploratory)

- Bayesian replication: report posterior mean + 95% credible interval for `Δ_satisfaction`
- Subgroup by body-cluster: does the calibration help more for some clusters?
- Order effect: does presenting calibrated first vs second change the result?
- Smartphone vs laptop sub-analysis

---

## 7. Risks and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Body-image distress from "this shape suits you" framing | Low-moderate | Screening excludes active eating disorders; we frame all recommendations as suggestions and explicitly say "you decide what you like" in the welcome script |
| Participant identifies our method by mode of presentation | Low | Conditions are labeled "Set 1/Set 2" only; debrief reveals labels post-hoc |
| Compensation perceived as binding | Low | Compensation is for time, not for "correct" answers; stated verbatim in consent |
| Data leakage of participant attributes | Negligible | No facial images, no real names; participant_id is sequential not tied to any external identifier |
| Sampling bias (authors' network) | Moderate | Acknowledge limitation in the paper; future work should run a larger, non-author-network study |

**No IRB required for non-clinical, anonymous, low-risk fashion-preference rating studies under Vietnamese research ethics norms.** If the authors' home institutions require IRB review, this protocol will be submitted to that institution's ethics committee.

---

## 8. Data handling

- Collected via a dedicated paper-form or Google Forms instance (no facial images).
- Stored in `research/user_study/data/` as a CSV with sequential participant IDs.
- Personally-identifying info (name, contact, payment receipt) stored SEPARATELY from the analysis CSV in an offline encrypted file; that linkage file is destroyed after compensation distribution.
- Free-text comments are translated EN ↔ VN by the researchers and quoted (paraphrased if anonymity needs preservation) in the paper.

---

## 9. Timeline

- **Week 1** — Pilot 3-5 participants to calibrate timing and clarity of the questionnaire. Revise protocol if needed.
- **Week 2-3** — Run N=30 participants. ~3 hours total of interview time spread across ~1 week.
- **Week 4** — Analysis + writeup. Deliver `RESULTS.md` with the pre-registered tests.

---

## 10. Reporting and pre-registration

This protocol is the **pre-registered analysis plan**. The numbers committed here:
- H1 test: paired t-test, one-sided, α=0.05
- H3 test: one-proportion z-test, one-sided, α=0.05
- Exclusions: as documented in §6

The paper's §8 (Evaluation) will report all three test outcomes plus the robustness checks. **We commit in advance: if H1 yields p ≥ 0.05, the calibration does not have a statistically-significant subjective effect, and the paper reports this honestly as a negative finding.**

Pre-registration hash: `git log -1 --format=%H` at the time of first participant run, recorded alongside the timestamp in `data/pilot_log.txt`.

---

## Files in this directory

- `PROTOCOL.md` — this file (the pre-registered plan).
- `CONSENT_VN.md` — Vietnamese-language consent form (template; to be co-signed paper or e-signed).
- `CONSENT_EN.md` — English-language consent form (mirror).
- `analysis.py` — analysis script that consumes `data/study_results.csv` and produces `RESULTS.md`. Pre-implemented per §6.
- `data/` — directory for the eventual CSV. Currently empty.
