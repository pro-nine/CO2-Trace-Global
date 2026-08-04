# GreenTrace — Deconstructing the Global Energy Transition

**Source data:** Our World in Data (CC BY) — 162 countries, 1990–2022.
**Data download date:** Live at runtime from raw.githubusercontent.com/owid

## The Central Finding

> **Of the 39 countries that reduced absolute CO₂ emissions between 1990 and 2022, the majority achieved this primarily through energy efficiency gains and economic restructuring — not through fuel switching to clean energy. Only a minority achieved reductions primarily driven by replacing fossil fuels with renewables or nuclear.**

## Finding 1: Germany and Ukraine cluster together — not with Denmark and Sweden

Unsupervised K-means clustering on CO₂/capita trajectory SHAPES (1990=1.0) finds 3 natural groups:
- **Cluster 0 (collapse-then-plateau, 27 countries):** sharp early drop, flat or modest subsequent decline. Includes: Germany, Ukraine, Estonia, Latvia, Lithuania, Romania, Czechia, Bulgaria — i.e. post-Soviet deindustrialisation AND Western European 'Energiewende' in the same cluster.
- **Cluster 2 (peak-and-decline, 44 countries):** rose through the 2000s, peaked, then declined. Includes UK, Denmark, Sweden, France, Austria, Switzerland, USA.
- **What this means:** Germany's trajectory shape is statistically closer to Ukraine's economic collapse than to Denmark's deliberate clean-energy transition. The CO₂ absolute numbers look similar but the mechanisms are fundamentally different. The Kaya decomposition makes this explicit.

## Finding 2: The Kaya Decomposition reveals the drivers

_(Negative values = CO₂-reducing contribution; positive = CO₂-increasing)_

| Country | Net change | Fuel switch (Mt) | Efficiency (Mt) | Affluence (Mt) | Population (Mt) |
|---|---|---|---|---|---|
| Austria | -1.2% | -9.0 | -31.9 | +30.0 | +10.3 |
| China | +371.6% | -1030.9 | -2088.6 | +11089.8 | +1258.0 |
| Denmark | -45.6% | -22.8 | -28.9 | +21.8 | +5.5 |
| France | -25.2% | -52.0 | -215.6 | +116.3 | +51.7 |
| Germany | -36.7% | -223.2 | -724.7 | +515.7 | +45.3 |
| Sweden | -36.7% | -19.4 | -34.9 | +23.9 | +9.4 |
| Ukraine | -79.8% | +15.6 | -412.4 | -82.9 | -83.8 |
| United Kingdom | -48.3% | -193.2 | -340.9 | +167.2 | +76.1 |
| United States | -1.5% | -887.5 | -2985.5 | +2275.9 | +1520.8 |

**Germany interpretation:** efficiency gains (−725 Mt) and fuel switching (−223 Mt) overcame affluence growth (+516 Mt). But efficiency here captures both genuine productivity improvements AND deindustrialisation — two very different processes with the same Kaya signature.
**UK interpretation:** efficiency (−341 Mt) is actually the larger term, with fuel switching (−193 Mt) second — the same efficiency/deindustrialisation ambiguity flagged for Germany applies here too, since the UK's manufacturing base also shrank over this period. What IS more confidently attributable to deliberate policy is the fuel-switching component specifically: carbon-intensity-of-energy is less confounded by economic restructuring than energy-intensity-of-GDP is, so the coal-closure-and-offshore-wind story is real, just smaller in Mt terms than the headline 'efficiency' number.
**Ukraine interpretation:** efficiency component (−412 Mt) reflects industrial collapse, not technological progress. Framing this as 'decarbonisation' would be misleading.

## Finding 3: Absolute decoupling is real but almost never sustained

- Countries currently in absolute decoupling (GDP/cap rising, CO₂/cap falling, 2018–2022): **57**
- Of these, the longest consecutive streak of absolute decoupling ranges 2–7 years — not sustained throughout the 32-year window.
- Longest consecutive absolute-decoupling streak so far — Sweden: 7. Germany: 4. Denmark: 4. UK: 7.
- Every country that shows net CO₂ reduction also shows periods of re-coupling — growth booms typically increase CO₂ even in 'green' economies.

## Finding 4: The offshoring ambiguity

- 41 of 162 countries are flagged as **suspect offshoring**: energy intensity fell >30% while carbon intensity of energy barely changed (<10%).
- Signature: a service-sector economy increasingly importing manufactured goods from high-emission countries. The country's *territorial* CO₂ falls but its *consumption-based* footprint may not.
- Flagged countries include: Lithuania, Italy, New Zealand, Canada, Norway, Cyprus, UAE, and others.
- This is not a conclusion — the OWID data does not have consumption-based emissions for all countries — but it is a directional diagnostic worth investigating.

## Limitations (honest)

- CO₂ is territorial (production-based), not consumption-based. A country that moved manufacturing abroad looks better than it really is.
- Kaya decomposition is accounting, not causal — it does not explain *why* energy intensity fell, only that it did.
- The 'suspect offshoring' flag is heuristic (no structural break test, no trade flow validation).
- Clustering is on trajectory shape, not mechanism — two countries can have the same shape for entirely different reasons.
