# Mineirinho Creek: Flash-Flood Timing

Source: Ranieri, Souza, Nishijima, Krishnamachari & Ueyama, "A deep learning workflow
enhanced with optical flow fields for flood risk estimation," Applied Intelligence, 2024
(the source paper for this benchmark's dataset and annotation methodology). During
periods of flash flood, the water level at this site rises fast and can overflow in less
than one hour, then returns to its low/dry-weather state within a couple of hours once
rainfall stops. This is why the creek is well-modeled as a series of short, independent
rainfall-driven events rather than a slowly-evolving time series -- and why a rain-gauge
reading from even a few hours before or after a frame's timestamp says little about that
frame's water level unless it falls within the same short event window.

# São Carlos: Local Flood-Type Terminology

Source: IPT (Instituto de Pesquisas Tecnologicas), "Mapeamento de Areas de Alto e Muito
Alto Risco a Deslizamentos e Inundacoes do Municipio de Sao Carlos-SP," Relatorio Tecnico
no. 144.443-205, 2015 -- an official municipal risk-mapping report. The report defines
five related but distinct flood-adjacent phenomena, in the Brazilian regulatory/technical
vocabulary for this region:

- **Enchente** (flood/flood surge): a temporary rise in water level within a channel.
- **Inundacao** (inundation): overflow of a channel's water beyond its banks into
  marginal/adjacent areas -- the closest local-terminology equivalent to this benchmark's
  `flood` category (water crossing onto the grass bank).
- **Alagamento** (waterlogging/ponding): water accumulation caused by a drainage system's
  insufficient capacity, not necessarily channel overflow.
- **Enxurrada** (flash flood / surface runoff): concentrated, high-energy surface runoff,
  typically the trigger mechanism behind a fast-rising `enchente` in a small urban
  watershed like Mineirinho Creek's.
- **Erosao marginal / solapamento** (bank erosion / bank collapse): secondary structural
  processes caused by repeated high-flow events, distinct from the water-level phenomena
  above but often co-occurring at the same sites.

# Mineirinho Creek: Official Site Classification

Source: IPT Relatorio Tecnico no. 144.443-205 (2015), p.50, section 5.2.2 (area SCA-02,
Centro/Botafogo). The report identifies Mineirinho Creek (corrego Mineirinho) as a
tributary of the Monjolinho Creek, converging near the "Rotatoria do Cristo" (Cristo
roundabout) in downtown Sao Carlos, alongside another tributary, Gregorio Creek. The
report notes observed bank scouring ("solapamentos de margem") on the Mineirinho and
Monjolinho creeks, and classifies this area as **R3 - Risco Alto (High Risk)** for
flooding, citing frequent inundation and dense commercial/residential occupation of the
area. (See the separate "Regional Risk-Probability Classification" entry for what the
R1-R4 scale actually measures -- it is not the same scale as this benchmark's
low/medium/high/flood water-level categories.)

# Regional Risk-Probability Classification (IPT) -- Not This Benchmark's Water-Level Scale

Source: IPT Relatorio Tecnico no. 144.443-205 (2015), p.21 (Quadro 4), citing methodology
from Ministerio das Cidades / IPT, 2007. Sao Carlos's official municipal risk mapping
uses a four-level scale -- R1 (Baixo/Low), R2 (Medio/Medium), R3 (Alto/High), R4 (Muito
Alto/Very High) -- but this measures something different from this benchmark's water-level
categories: **it is the estimated probability that a destructive flood or landslide event
will occur at a given location within a 1-year period**, based on geological, geotechnical,
and land-occupation factors, not the instantaneous water level visible in a photograph.
R1 means no destructive event is expected within a year under current conditions; R4 means
such an event is judged very likely during intense/prolonged rain within a year. Do not
equate an area's IPT risk classification (e.g. Mineirinho's R3) with this benchmark's
`high` water-level category just because both use the word "high" -- they answer different
questions (annual event probability vs. current observed water level).

# A Documented Severe Flood Event Near Mineirinho Creek (Different Site, Outside This Benchmark's Period)

Source: IPT Relatorio Tecnico no. 144.443-205 (2015), p.56 (area SCA-07). On 22 October
2013, Gregorio Creek -- a different tributary that, like Mineirinho Creek, converges near
the Rotatoria do Cristo in downtown Sao Carlos -- overflowed its banks, passing over the
bridge on Avenida Comendador Alfredo Maffei and reaching a maximum flood depth of 4.5
meters with a 30-meter lateral reach. Recorded rainfall for the event was 108 mm/h. This
event is **not** on Mineirinho Creek itself and predates this benchmark's labeled dataset
(November 2018 - February 2022) by five years -- it is included here only as an
illustrative example of what a severe, well-documented flood event in this same downtown
drainage network looks like in terms of rainfall intensity and flood depth, not as a
historical analog for any specific image in this dataset.

# Verified Seasonal Rainfall Pattern (Sao Carlos, 2018-2022)

Independently computed from INMET (Instituto Nacional de Meteorologia) automatic weather
station A711 (Sao Carlos), hourly precipitation records, 1 January 2018 - 30 December
2022. Of all rainfall recorded at this station across the full 5-year period, **64.4%
fell during November-February** -- the same Nov-Feb window used to define each "season"
in this benchmark's leave-one-season-out evaluation splits. This independently corroborates
the source paper's own rainfall-seasonality analysis (based on a different station,
operated by DAEE, Sao Carlos's Department of Water and Electricity) and confirms Nov-Feb
is genuinely the rainy season driving nearly all flood-relevant water-level variation at
this site.
