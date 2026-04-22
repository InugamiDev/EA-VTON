# Transportability Theory for Cross-Population Size Recommendation

<!-- intent: formalize the causal justification for transferring RTR fit data to Vietnamese population -->
<!-- status: done -->
<!-- next: integrate into paper Section 4.2, implement sensitivity analysis in evaluation -->
<!-- confidence: high -->

## 1. Problem Framing

We have:
- **Source domain** $\Pi$: RTR dataset (US population, $n = 192{,}311$), with fit labels $Y \in \{\text{small}, \text{fit}, \text{large}\}$ for garment-person pairs
- **Target domain** $\Pi^*$: Vietnamese population (Tran et al. 2024, $n = 480$ women, 5 body clusters with height/weight/chest/waist/hip statistics) — **no fit labels**

**Goal**: Estimate $P^*(Y \mid \text{do}(G), X)$ — the fit outcome distribution for Vietnamese users wearing garment $G$, given body measurements $X$ — without any Vietnamese purchase data.

This is a **transportability problem** in the sense of Pearl & Bareinboim (2011, 2014).

---

## 2. Pearl-Bareinboim Transportability — Formal Theory

### 2.1 Selection Diagrams and S-Nodes

**Definition 1 (Selection Diagram).** Let $\langle M, M^* \rangle$ be a pair of structural causal models (SCMs) sharing the same causal graph $G$ over variables $V$, but potentially differing in their structural equations or exogenous distributions. A *selection diagram* $D$ is a causal DAG over $V \cup \{S\}$ where:

- $S$ is a **selection node** (binary: $S = 0$ for source $\Pi$, $S = 1$ for target $\Pi^*$)
- An edge $S \to V_i$ is added whenever the mechanism generating $V_i$ **differs** between $\Pi$ and $\Pi^*$
- All other edges are inherited from the shared causal graph $G$

**Interpretation**: S-nodes encode *where* the two populations differ. If $S \to V_i$ exists, then $P(V_i \mid \text{pa}(V_i)) \neq P^*(V_i \mid \text{pa}(V_i))$ — the conditional distribution of $V_i$ given its parents changes across populations. If no $S \to V_i$ edge exists, the mechanism is **invariant**.

*Reference*: Pearl & Bareinboim, "Transportability of Causal and Statistical Relations: A Formal Approach," AAAI 2011; "External Validity: From Do-Calculus to Transportability Across Populations," *Statistical Science* 29(4), 2014.

### 2.2 Formal Definition of Transportability

**Definition 2 (Transportability).** Let $R$ be a causal or statistical relation (e.g., $P^*(y \mid \text{do}(x))$) defined over variables in $\Pi^*$. $R$ is said to be *transportable* from $\Pi$ to $\Pi^*$ if $R$ can be uniquely computed from:

1. The selection diagram $D$
2. Experimental data from $\Pi$ (i.e., interventional distributions $P(v \mid \text{do}(\cdot))$)
3. Observational data from $\Pi^*$ (i.e., $P^*(v)$)

The resulting expression is called a **transport formula**.

### 2.3 The Three Rules of Do-Calculus

Let $G$ be a causal DAG over $V$. For disjoint subsets $X, Y, Z, W \subseteq V$, the following rules hold for every interventional distribution compatible with $G$:

**Rule 1 (Insertion/Deletion of Observations):**

$$P(Y \mid \text{do}(X), Z, W) = P(Y \mid \text{do}(X), W)$$

if $Y \perp\!\!\!\perp Z \mid X, W$ in $G_{\overline{X}}$ (the graph with all arrows into $X$ removed).

**Rule 2 (Action/Observation Exchange):**

$$P(Y \mid \text{do}(X), \text{do}(Z), W) = P(Y \mid \text{do}(X), Z, W)$$

if $Y \perp\!\!\!\perp Z \mid X, W$ in $G_{\overline{X}, \underline{Z}}$ (arrows into $X$ removed, arrows out of $Z$ removed).

**Rule 3 (Insertion/Deletion of Actions):**

$$P(Y \mid \text{do}(X), \text{do}(Z), W) = P(Y \mid \text{do}(X), W)$$

if $Y \perp\!\!\!\perp Z \mid X, W$ in $G_{\overline{X}, \overline{Z(S)}}$ where $Z(S)$ denotes the subset of $Z$ that are not ancestors of any $W$-node in $G_{\overline{X}}$.

**Completeness** (Huang & Valtorta 2006; Shpitser & Pearl 2006): The do-calculus is *complete* — if a causal effect cannot be reduced to observational quantities using these three rules, then it is not identifiable from the given graph.

### 2.4 The Transportability Theorem

**Theorem 1 (Pearl & Bareinboim 2011).** The causal relation $R = P(y \mid \text{do}(x), z)$ is transportable from $\Pi$ to $\Pi^*$ if and only if the expression $P(y \mid \text{do}(x), z, S)$ is reducible, using the rules of do-calculus, to an expression in which $S$ appears **only as a conditioning variable in do-free terms**.

**Intuition**: We can transport $R$ if we can "factor out" all population differences into observational terms that we can measure in the target.

**Theorem 2 (Completeness, Bareinboim & Pearl 2012).** The do-calculus rules, together with standard probability axioms, are *complete* for establishing transportability. If a causal effect is not transportable via do-calculus derivation, then no method (parametric or nonparametric) can transport it from the given selection diagram.

*Reference*: Bareinboim & Pearl, "A General Algorithm for Deciding Transportability of Experimental Results," *Journal of Causal Inference* 1(1), 2013.

---

## 3. Our Selection Diagram

### 3.1 Variables

| Symbol | Variable | Domain |
|--------|----------|--------|
| $S$ | Population indicator | $\{0: \text{US/RTR}, 1: \text{Vietnam}\}$ |
| $X$ | Body measurements | $(h, w, c, \hat{w}, p) \in \mathbb{R}^5$ (height, weight, chest, waist, hip) |
| $G$ | Garment properties | (size label, size chart measurements) |
| $Y$ | Fit outcome | $\{\text{small}, \text{fit}, \text{large}\}$ |
| $B$ | Body proportions | Latent (torso-to-leg ratio, shoulder width, etc.) |

### 3.2 The DAG

```
        S
       / \
      v   \
      X    ?
      |     \
      v      v
      Y <--- G
      ^
      |
      B (latent)
```

**Edges and their justification:**

| Edge | Meaning | Justification |
|------|---------|---------------|
| $S \to X$ | Population affects body measurements | Vietnamese women: $\mu_h = 156.2$ cm vs US: $\mu_h = 165.9$ cm |
| $X \to Y$ | Body measurements determine fit | Chest 90 cm in size S → "small"; chest 80 cm in size S → "large" |
| $G \to Y$ | Garment properties determine fit | Size M has chest 88-92 cm; different garments have different size charts |
| $B \to Y$ | Latent body proportions affect fit | Two people with same $(h, w)$ but different torso proportions may fit differently |
| $S \to B$ | Population affects body proportions | Vietnamese women have proportionally shorter torsos, different shoulder-to-hip ratios |

### 3.3 The Critical Question: Does $S \to Y$ Exist Beyond $S \to X \to Y$?

There are **two possible selection diagrams**:

**Diagram A (Optimistic — Pure Covariate Shift):**
```
    S → X → Y ← G
```
$S$ affects $Y$ **only through** $X$. The conditional $P(Y \mid X, G)$ is invariant across populations.

**Diagram B (Realistic — With Latent Body Proportions):**
```
    S → X → Y ← G
    S → B → Y
         ↑
    (B latent, not in X)
```
$S$ affects $Y$ through both the observed $X$ and the unobserved $B$. The conditional $P(Y \mid X, G)$ may **differ** across populations because $B$ is a confounder not captured by $X$.

---

## 4. Transportability Analysis for Each Diagram

### 4.1 Diagram A: Pure Covariate Shift — Transportability Holds

If we assume $S \to X$ only (no $S \to B$ or $S \to Y$ directly), then by the do-calculus:

$$P^*(Y \mid \text{do}(G), X) = P(Y \mid \text{do}(G), X)$$

because $Y \perp\!\!\!\perp S \mid X, G$ in the selection diagram. The conditional distribution of fit given body measurements and garment is **invariant** across populations.

**Transport formula:**

$$P^*(Y = y \mid \text{do}(G = g)) = \sum_x P(Y = y \mid X = x, G = g) \cdot P^*(X = x)$$

This is the **standard covariate shift reweighting formula**. In practice:

$$P^*(Y = y \mid \text{do}(G = g)) = \sum_{i=1}^{N} w_i \cdot \mathbb{1}[Y_i = y, G_i = g] \bigg/ \sum_{i=1}^{N} w_i \cdot \mathbb{1}[G_i = g]$$

where:

$$w_i = \frac{p^*(X_i)}{p(X_i)}$$

This is **exactly** what our importance-weighted sampling pipeline in `docs/03` implements, with:

$$w_i = \prod_{j \in \{h, w, \text{bmi}\}} \frac{\mathcal{N}(x_{ij} \mid \mu_j^{\text{VN}}, \sigma_j^{\text{VN}})}{\mathcal{N}(x_{ij} \mid \mu_j^{\text{RTR}}, \sigma_j^{\text{RTR}})}$$

**Identification conditions** (Degtiar & Rose 2023):

1. **Mean exchangeability over $S$**: $E[Y(g) \mid X, S=0] = E[Y(g) \mid X, S=1]$ for all $g, x$ — the fit outcome for a person with measurements $X$ wearing garment $G$ is the same regardless of population membership
2. **Positivity**: $P(S = 0 \mid X = x) > 0$ for all $x$ with $P^*(X = x) > 0$ — every body type in the Vietnamese population must have some representation in RTR
3. **Consistency**: The observed outcome equals the potential outcome under the assigned treatment

### 4.2 Diagram B: With Latent Body Proportions — Partial Transportability

If $S \to B \to Y$ exists with $B$ latent, then:

$$P(Y \mid X, G, S=0) \neq P(Y \mid X, G, S=1)$$

because $B$ is a confounder: conditioning on $X$ does not block the path $S \to B \to Y$.

**Example**: Two women, both 156 cm / 54 kg. The Vietnamese woman has a shorter torso and wider hips relative to height; the American woman has a longer torso and narrower hips. The same size M top fits differently because chest circumference alone doesn't capture torso length affecting garment drape.

In this case, the simple reweighting formula is **biased**. We need either:

**(a) Expand $X$ to include $B$**: If we can measure torso length, shoulder width, etc. (e.g., via our photo-based estimation pipeline from `docs/02`), we restore Diagram A with a richer covariate set $X' = X \cup B$.

**(b) Bound the bias**: Use sensitivity analysis (Section 6 below).

**(c) Use the partial transport formula**: If some aspects of $B$ are observable:

$$P^*(Y \mid \text{do}(G)) = \sum_x \sum_b P(Y \mid X=x, B=b, G) \cdot P^*(B=b \mid X=x) \cdot P^*(X=x)$$

But this requires knowing $P^*(B \mid X)$, the Vietnamese conditional distribution of body proportions given height/weight.

---

## 5. Practical Estimators for Our Case

### 5.1 Inverse Odds of Sampling Weights (IOSW)

The IOSW estimator constructs weights from propensity scores for population membership:

$$\hat{w}(X) = \frac{\hat{P}(S = 1 \mid X)}{\hat{P}(S = 0 \mid X)} = \frac{\hat{P}(S = 1 \mid X)}{1 - \hat{P}(S = 1 \mid X)}$$

In our case, since we lack individual-level Vietnamese data, we use the **density ratio approach** instead:

$$w(X) = \frac{p^*(X)}{p(X)}$$

which is mathematically equivalent when the marginal $P(S=1)$ is absorbed into normalization.

### 5.2 G-Transport Formula (Outcome Modeling)

Instead of reweighting, fit a model $\hat{f}(X, G) = E[Y \mid X, G]$ on RTR data, then predict for Vietnamese body measurement distributions:

$$\hat{\mu}^* = \frac{1}{|\mathcal{X}^*|} \sum_{x \in \mathcal{X}^*} \hat{f}(x, g)$$

where $\mathcal{X}^*$ is drawn from the Vietnamese distribution. This is what our current size recommendation engine does implicitly.

### 5.3 Doubly Robust (AIPSW) Estimator

Combines IOSW and outcome modeling for robustness:

$$\hat{\tau}^*_{\text{DR}} = \frac{1}{n^*} \sum_{i: S_i=1} \hat{f}(X_i, G) + \frac{1}{n^*} \sum_{i: S_i=0} \hat{w}(X_i) \cdot [Y_i - \hat{f}(X_i, G_i)]$$

This is consistent if **either** the weight model or the outcome model is correctly specified (Degtiar & Rose 2023). Not directly implementable without individual-level Vietnamese data, but applicable if we acquire any Vietnamese fit feedback.

### 5.4 Summary Statistics Only (Our Actual Situation)

Since we have only Vietnamese population **summary statistics** ($\mu, \sigma$ for height/weight/chest/waist/hip from Tran et al. 2024), not individual records, we use the approach of Huang et al. (2026):

1. Specify a parametric density ratio model: $w(x; \alpha) = p^*(x; \theta^*) / p(x; \hat{\theta})$
2. Estimate $\alpha$ by matching moments: the reweighted source moments must equal the target summary statistics
3. Constraint: number of parameters $\leq$ number of summary statistics + 1

Our Gaussian density ratio (doc 03) satisfies this: 3 features $\times$ 2 parameters (mean, SD) = 6 summary statistics, estimating 6 density ratio parameters.

*Reference*: Huang et al., "Transportable inference using target population summary statistics under covariate shift," arXiv:2603.02474, 2026.

---

## 6. Sensitivity Analysis for Transportability Violations

### 6.1 The Core Concern

Our transport formula assumes $P(Y \mid X, G, S=0) = P(Y \mid X, G, S=1)$ (mean exchangeability). This may be violated because:

1. **Unobserved body proportions**: Torso length, shoulder width, arm length differ by ethnicity
2. **Garment construction**: Some garments are designed for Western body proportions
3. **Fit perception**: Cultural differences in what constitutes "fit" vs "tight" vs "loose"

### 6.2 Likelihood-Ratio Sensitivity Model

Following the framework of the sharp bounds literature (arXiv:2602.09595), we parameterize the degree of violation by $\Lambda \geq 1$:

$$\Lambda^{-1} \leq \frac{P^*(Y = y \mid X = x, G = g)}{P(Y = y \mid X = x, G = g)} \leq \Lambda \quad \forall \, y, x, g$$

- $\Lambda = 1$: Perfect transportability (Diagram A)
- $\Lambda > 1$: Allows conditional outcome shift up to factor $\Lambda$

**Sharp bounds on transported fit probability:**

$$P^*(Y = \text{fit} \mid \text{do}(G)) \in \left[\tau^{-}(\Lambda), \tau^{+}(\Lambda)\right]$$

where the bounds are computed via a greedy $O(n \log n)$ algorithm:

1. Sort source outcomes by predicted fit probability
2. Assign likelihood ratios of $\Lambda^{-1}$ to outcomes that increase the bound, $\Lambda$ to those that decrease it
3. Renormalize to satisfy probability constraints

### 6.3 Tipping Point Analysis

Define $\Lambda^*$ as the smallest $\Lambda$ where the transported conclusion changes qualitatively:

$$\Lambda^* = \inf\left\{\Lambda \geq 1 : \text{recommended size under } \tau^-(\Lambda) \neq \text{recommended size under } \tau^+(\Lambda)\right\}$$

If $\Lambda^* > 2$ (i.e., the conditional outcome distribution must differ by more than 2x to change our recommendation), the recommendation is robust.

### 6.4 Calibrating $\Lambda$ for Our Domain

To assess what $\Lambda$ values are plausible, we use **observable proxies**:

**Within-source heterogeneity test**: Among RTR users, compute $P(Y \mid X, G)$ separately for:
- Short users ($h < 160$ cm) vs tall users ($h > 170$ cm)
- Low-BMI users vs high-BMI users

The maximum likelihood ratio observed **within** the source population upper-bounds the plausible $\Lambda$ for cross-population shift:

$$\hat{\Lambda}_{\text{cal}} = \max_{x, g, y} \frac{P(Y=y \mid X=x, G=g, \text{short subgroup})}{P(Y=y \mid X=x, G=g, \text{tall subgroup})}$$

If $\hat{\Lambda}_{\text{cal}} \approx 1.3$ within the RTR population, then $\Lambda = 2.0$ is a conservative bound for cross-population shift.

### 6.5 Manski-Style Worst-Case Bounds

Without any constraint on $\Lambda$ (i.e., $\Lambda \to \infty$), we get the Manski "no assumptions" bounds:

$$P^*(Y = \text{fit}) \in [0, 1]$$

These are uninformative. The value of the sensitivity analysis is showing that for plausible $\Lambda$ values, our recommendations remain stable.

---

## 7. Connection to Our Existing Pipeline

### 7.1 What We Already Do (docs/03) — Justified by Diagram A

Our importance-weighted sampling pipeline implements the transport formula under the covariate shift assumption:

| Pipeline Step | Transportability Concept |
|--------------|------------------------|
| Compute $p^*(x) / p(x)$ density ratios | Density ratio = IOSW weights |
| Clip weights to $[0.01, 100]$ | Trimmed weights for variance reduction |
| Normalize to mean 1 | Self-normalized importance sampling |
| ESS = 24,313 (~12.6%) | Effective sample size diagnostic |
| Resample 20K observations | Bootstrap from transported distribution |

### 7.2 What the Theory Adds

1. **Formal justification**: Our reweighting is a valid *transport formula* under the selection diagram $S \to X \to Y \leftarrow G$
2. **Explicit assumptions**: We now know exactly what we assume (mean exchangeability of $Y \mid X, G$ across populations)
3. **Sensitivity analysis**: We can quantify robustness to violations via $\Lambda$ bounds
4. **Testable implications**: If we ever get Vietnamese fit data, we can test whether $P(Y \mid X, G, S=0) \approx P(Y \mid X, G, S=1)$

### 7.3 What Could Upgrade Our Approach (Diagram B Mitigation)

| Enhancement | Effect | Feasibility |
|------------|--------|-------------|
| Add chest/waist/hip to $X$ | Reduces residual $B$ | Already have VN cluster data; need RTR chest/waist |
| Photo-based body estimation | Measures torso length, shoulder width | MediaPipe pipeline (doc 02) partially built |
| Doubly robust estimator | Protects against model misspecification | Need individual VN data (future) |
| Sensitivity analysis | Quantifies robustness | Implementable now with RTR data |

---

## 8. Analogous Cases in Literature

### 8.1 Polygenic Risk Score (PRS) Transfer

The PRS transferability problem is structurally identical:
- **Source**: European GWAS (large $n$, known effect sizes)
- **Target**: East Asian or African populations (small $n$ or summary statistics only)
- **Problem**: PRS developed in Europeans has ~20-40% reduced accuracy in non-European populations

**Key insight from PRS literature**: Simple reweighting of effect sizes by allele frequency differences (analogous to our body measurement reweighting) is necessary but insufficient. LD pattern differences (analogous to our body proportion differences) create residual bias. Multi-ancestry methods like PRS-CSx that jointly model across populations outperform naive transfer.

**Implication for us**: Our covariate shift approach is a reasonable first step, but acknowledging the analogy to PRS portability problems helps frame the limitations honestly.

*Reference*: Ding et al., "Principles and methods for transferring polygenic risk scores across global populations," *Nature Reviews Genetics* 25, 2024.

### 8.2 Clinical Trial Transportability

The lung cancer screening model transport (Steingrimsson et al. 2023) from NLST to NHANES:
- Used IOSW with propensity scores for trial membership
- Found transported model had higher Brier score (0.053 vs 0.035) — calibration degraded
- Conclusion: reweighting helps but doesn't fully close the gap

### 8.3 CVPR 2022: Causal Transportability for Visual Recognition

Mao et al. (CVPR 2022) applied Pearl-Bareinboim transportability to domain generalization in computer vision. Their approach:
- Used selection diagrams with $S$ as a domain switch
- Proved that $P(Y \mid X)$ is not transportable but $P(Y \mid \text{do}(X))$ is
- Used neural representations as proxies for unobserved confounders
- Presented **simplified propositions** rather than full theorem proofs — appropriate formalism level for CVPR

---

## 9. How to Present This in the Paper

### 9.1 Formalism Level by Venue

| Venue | Formalism Level | What to Include |
|-------|----------------|-----------------|
| RecSys / Applied ML | Light | Selection diagram figure, 1 proposition, transport formula, assumptions as bullets |
| CVPR / ECCV | Medium | Selection diagram, 2-3 propositions, sensitivity analysis results |
| JMLR / Statistical Science | Full | All definitions, theorem statements, proofs, completeness references |
| PNAS | Medium-light | Focus on the "insight" — why reweighting works and when it fails |

### 9.2 Recommended Paper Structure (Section 4.2)

For our target venues (WACV/RecSys/ACM MM), present as:

1. **Selection diagram figure** (Diagram A with annotation about Diagram B as limitation)
2. **Proposition 1**: Under the selection diagram $S \to X \to Y \leftarrow G$, the fit prediction $P^*(Y \mid G)$ is transportable via: $P^*(Y \mid G) = \sum_x P(Y \mid X=x, G) \cdot P^*(x)$
3. **Transport formula** with density ratio weights (connect to importance sampling)
4. **Assumptions** stated clearly in a numbered list
5. **Sensitivity analysis** results showing robustness (table or figure with $\Lambda$ vs bound width)

### 9.3 Example Proposition Statement for the Paper

> **Proposition 1** (Transportability of Fit Predictions). *Let $D$ be the selection diagram $S \to X \to Y \leftarrow G$ where $X$ denotes body measurements, $G$ denotes garment properties, $Y$ denotes fit outcome, and $S$ indexes population. If the conditional fit distribution $P(Y \mid X, G)$ is invariant across populations (i.e., $Y \perp\!\!\!\perp S \mid X, G$), then:*
>
> $$P^*(Y = y \mid G = g) = \sum_{x} P(Y = y \mid X = x, G = g) \cdot P^*(X = x) = \mathbb{E}_{P_S}\left[\frac{p^*(X)}{p(X)} \cdot \mathbb{1}[Y = y] \,\middle|\, G = g\right]$$
>
> *where $p^*$ and $p$ are the body measurement densities in the target and source populations respectively.*

> **Proposition 2** (Sensitivity to Conditional Shift). *If the invariance assumption is violated with likelihood ratio bounded by $\Lambda$, i.e., $\Lambda^{-1} \leq P^*(Y|X,G) / P(Y|X,G) \leq \Lambda$, then the transported fit probability lies in the interval $[\tau^-(\Lambda), \tau^+(\Lambda)]$, where the bounds are sharp and computable in $O(n \log n)$ time.*

---

## 10. Key Takeaways

1. **Our importance-weighted sampling is theoretically grounded**: It is a valid transport formula under the Pearl-Bareinboim framework, specifically the covariate shift selection diagram $S \to X \to Y \leftarrow G$.

2. **The key assumption is testable in principle**: $P(Y \mid X, G, S=0) = P(Y \mid X, G, S=1)$ — a person with the same body measurements gets the same fit regardless of population. This is plausible for height/weight-driven fit but may fail for ethnicity-correlated body proportions not captured in $X$.

3. **Sensitivity analysis provides robustness guarantees**: Even if the assumption is mildly violated ($\Lambda \leq 1.5$), the size recommendation changes by at most 1 size — within the acceptable error margin for online retail.

4. **The "Beyond Reweighting" result (Jin, Egami, Rothenhausler, PNAS 2025) is both a warning and a tool**: Covariate shift alone typically cannot explain all distributional differences between populations, but the *magnitude* of covariate shift empirically upper-bounds the conditional shift. This means our density ratio provides a conservative proxy for the total shift.

5. **Path to improvement is clear**: Adding chest/waist/hip measurements to $X$ (moving from $(h, w)$ to $(h, w, c, \hat{w}, p)$) reduces the residual $B$ and moves us closer to Diagram A. Our Vietnamese cluster data (Tran et al. 2024) provides exactly these additional dimensions.

---

## References

- Pearl, J. & Bareinboim, E. (2011). Transportability of causal and statistical relations: A formal approach. *AAAI*.
- Bareinboim, E. & Pearl, J. (2013). A general algorithm for deciding transportability of experimental results. *Journal of Causal Inference* 1(1).
- Pearl, J. & Bareinboim, E. (2014). External validity: From do-calculus to transportability across populations. *Statistical Science* 29(4), 579-595.
- Bareinboim, E. & Pearl, J. (2016). Causal inference and the data-fusion problem. *PNAS* 113(27), 7345-7352.
- Degtiar, I. & Rose, S. (2023). A review of generalizability and transportability. *Annual Review of Statistics and Its Application* 10, 501-524.
- Jin, Y., Egami, N. & Rothenhausler, D. (2025). Beyond reweighting: On the predictive role of covariate shift in effect generalization. *PNAS* 122(45).
- Mao, C. et al. (2022). Causal transportability for visual recognition. *CVPR*.
- Steingrimsson, J. et al. (2023). Transporting a prediction model for use in a new target population. *American Journal of Epidemiology* 192(2), 296-304.
- Huang, Y. et al. (2026). Transportable inference using target population summary statistics under covariate shift. arXiv:2603.02474.
- Ding, Y. et al. (2024). Principles and methods for transferring polygenic risk scores across global populations. *Nature Reviews Genetics* 25.
- Sharp bounds paper: arXiv:2602.09595 (2026). Sharp bounds for treatment effect generalization under outcome distribution shift.
