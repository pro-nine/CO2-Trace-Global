# CO2 Trace: audited carbon accounting, 1990-2022

Data: committed OWID snapshot, 162 countries (manifest sha256 `2b29d86f403e`). Every number below is computed by `python -m scripts.run_analysis`; none is typed by hand.

## 1. Audit of the v1 report

| # | v1 claim | verdict |
|---|---|---|
| C1 | Germany clusters with Ukraine, not with Denmark and Sweden. | **not supported** |
| C2 | Most countries that cut CO2 did so mainly through efficiency, not fuel switching. | **supported** |
| C3 | Absolute decoupling is real but never sustained (longest run 7 years). | **definition-sensitive** |
| C4 | 41 countries show the signature of offshoring. | **misleading** |
| C5 | The decomposition covers all 162 countries. | **denominator differs** |

### C1: Germany and Ukraine
On the RMS distance between log-trajectories (1990 = 1), Germany is 0.53 from Ukraine, 0.21 from Denmark and 0.11 from Sweden. Under a moving-block bootstrap over years (k = 3), Germany is co-clustered with Denmark in 100% of draws, with Sweden in 100%, and with Ukraine in 46%. The v1 K-means run did put Germany and Ukraine together, but Germany's nearest centre was only 7% closer than the next, so the assignment was a boundary case, and 16 near-zero emitters were excluded here because their 1990-based ratios are not meaningful.

### C2: efficiency versus fuel switching
Of 39 countries whose CO2 fell, 82% had efficiency as their larger reducing effect; 78% to 85% under 3% log-normal noise on all inputs. This is a sensitivity range, not a sampling confidence interval.

### C3: how long does absolute decoupling last?
Strictly consecutive strong-decoupling spells: 1036, median length 1 year, 70% last a single year, longest 7 years. Allowing one interrupted year, the longest spell becomes 16 years. The claim holds only under the strict definition.

### C4: the offshoring flag
41 of 162 countries meet the v1 rule (energy intensity down more than 30%, carbon intensity of energy down less than 10%). 24 of them (59%) have rising CO2 per capita and a median GDP per capita growth of 131%: fast growth lowers energy per GDP mechanically. The flag does not isolate offshoring and is not used here.

### C5: coverage
The panel has 162 countries; 154 have positive values at both endpoints and 153 in every year, so the chain-linked decomposition covers the smaller set.

## 2. Where emissions changed: chain-linked LMDI

Year-by-year LMDI closes exactly (largest residual 3.3e-11 Mt). Its totals differ from the two-endpoint version because each year is weighted by that year's emissions. The eight largest differences:

| Country | Efficiency, endpoint (Mt) | Efficiency, chain (Mt) | Difference (Mt) |
|---|---|---|---|
| China | -2089 | -3312 | -1224 |
| Russia | -1796 | -1276 | +520 |
| United States | -2985 | -3291 | -305 |
| India | -445 | -593 | -148 |
| Kazakhstan | -270 | -165 | +105 |
| Ukraine | -412 | -332 | +80 |
| South Korea | -28 | -80 | -52 |
| Japan | -318 | -362 | -44 |

For large reducers, the share of the cumulative efficiency effect that fell in 1991-1998: Germany 36%, Poland 34%, Romania 33%, Netherlands 27%, Czechia 26%; for comparison United Kingdom 16%. A front-loaded efficiency effect is consistent with transition-era restructuring, but this panel has no sector split and cannot separate it from technology.

## 3. Do decoupling spells last? Survival analysis

1036 strong-decoupling spells, right-censored where a spell was still running in 2022. Kaplan-Meier survival past one year: high-income 42%, middle 31%, low 30% (1990 GDP per capita tertiles). Log-rank high vs low: chi2 = 11.9, p = 0.0005.

## 4. Synthetic control: what can and cannot be said

Outcome: log carbon intensity of energy. Donors: the 15 countries closest before the policy, weights on the probability simplex, fit scored on the last five pre-policy years. Cases and years are well-known policy dates picked by the analyst; point estimates were seen before this report was written, so inference is exploratory. p-values rank the standardised post-policy gap among in-space placebos.

| Case | Effect vs synthetic | Held-out pre-fit RMSPE | Placebo p |
|---|---|---|---|
| UK Carbon Price Support (Apr 2013) | -9.2% | 0.049 | 0.25 |
| Australia carbon price (Jul 2012) | -1.6% | 0.022 | 0.72 |
| Ireland carbon tax (2010) | -1.9% | 0.037 | 0.78 |
| Japan nuclear shutdowns after Fukushima (2011, positive control) | +14.5% | 0.026 | 0.10 |

Japan 2011 is a positive control: a large, known increase in carbon intensity. If the test cannot flag that at conventional levels, it has low power here, and a non-significant UK, Australian or Irish estimate is not evidence of no effect.

## Limitations

- CO2 is territorial; consumption-based emissions and trade flows are not in the panel.
- Kaya LMDI is accounting. Efficiency mixes technology, structure and offshoring.
- Spell and distance results depend on stated definitions (strict spells, 4x winsorisation, k chosen by silhouette).
- Synthetic control is exploratory: few treated units, short pre-periods, and EU donors share common policy shocks.
