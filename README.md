# CO2 trace 

**An open-data analysis of 162 countries' CO₂ trajectories (1990–2022).**  
Uses the Kaya Identity (LMDI decomposition), decoupling classification, and unsupervised trajectory clustering to ask: *where did each country's emissions change actually come from and does it transfer as a policy lesson?*

---

## The three findings

**1. Germany and Ukraine cluster together — not with Denmark and Sweden.**  
Unsupervised K-means on trajectory shapes (1990=1.0) produces 3 clusters. Germany lands in the "sharp early drop, then plateau" cluster alongside post-Soviet states — not in the "deliberate clean-energy transition" cluster containing UK, Denmark, Sweden, France. The Kaya decomposition explains why: Germany's reduction is dominated by energy efficiency (partly genuine, partly post-reunification deindustrialisation), while the UK's is dominated by fuel switching (coal closure + offshore wind). Same headline number, different mechanism, different transferability as a policy lesson.

**2. Absolute decoupling is real but never sustained.**  
Of countries currently in absolute decoupling (GDP/cap rising, CO₂/cap falling), the longest *consecutive* run is 7 years. Sweden: 7. UK: 7. Germany: 4. Denmark: 4. Every green economy re-couples during boom periods. No country has maintained it across the full 32-year window.

**3. 41 countries flag as suspect offshoring.**  
Energy intensity of GDP fell >30% while carbon intensity of energy barely changed (<10%) — signature of a service-sector economy that moved manufacturing abroad rather than decarbonising it. Countries flagged include Lithuania, Italy, New Zealand, Canada, Norway.

---

## Method =>  Kaya Identity + Log-Mean Divisia Index (LMDI) 

```
CO₂ = Population × (GDP/Pop) × (Energy/GDP) × (CO₂/Energy)
```

Log-Mean Divisia Index (LMDI) decomposition: zero residual by construction, verified computationally for all 154 countries.

---

