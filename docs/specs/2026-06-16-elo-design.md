# Fase 2 — Dynamic Elo (design)

> Fecha: 2026-06-16. Estado: aprobado. Implementación: TDD.

## Objetivo

Calcular un rating Elo dinámico por selección a partir del histórico
`martj42/international_results`, recondicionable con resultados del WC2026. El rating
alimenta el mapeo Elo→goles de Dixon-Coles (Fase 3) y la simulación condicional (Fase 4).

## Decisiones (este diseño)

- **Estructura:** Elo secuencial cronológico. Se procesan los partidos de más antiguo a
  más reciente; cada uno actualiza ambos ratings. La reconditioning live (Fase 4) añade
  los resultados 2026 a la misma lista y re-ejecuta el ajuste (recompute completo).
- **Expectativa:** logística `E_home = 1 / (1 + 10^(-(R_home + HA - R_away)/400))`, con
  `HA = home_advantage` si `neutral=False`, si no `0`.
- **Factor K:** `Δ = K_base(importancia) · G(margen) · recency · (resultado − E)`.
  - `K_base` por importancia del torneo (config `elo.k_factors`).
  - `G(margen)` = piecewise eloratings.net: `1` si `|d|≤1`, `1.5` si `|d|=2`,
    `(offset+|d|)/divisor` si `|d|≥3` (params en config).
  - `recency = 0.5^(edad_meses / half_life)`, edad relativa a `reference_date`
    (= `max(match.date)` por defecto, o explícito). Determinista por snapshot;
    conversión días→meses con constante de calendario documentada (no hiperparámetro).
- **Init:** `initial_rating = 1500` plano para todas las selecciones. Sin prior FIFA.
- **FIFA ranking:** NO se ingiere. El Elo recondicionado live ES, de hecho, un ranking
  estilo-FIFA en tiempo real (FIFA usa Elo desde 2018 y publica ~mensual, congelado
  durante el Mundial). Queda como **baseline de comparación en el backtest (Fase 5)**.
- **Sede (host):** el bono `host_advantage` (USA/MX/CA) es un concepto de *simulación*
  (Fase 4), NO del ajuste histórico. Aquí solo se usa `home_advantage` vía `neutral`.

## Componentes

### `data/historical.py` (nuevo)
- `fetch_martj42(url, dest) -> Path` — I/O (requests perezoso); cachea `results.csv`
  (y `shootouts.csv`) en `data/raw/`. Idempotente.
- `HistoricalMatch` (dataclass): `date, home_team, away_team, home_score, away_score,
  tournament, neutral` (columnas VERIFICADAS de martj42).
- `parse_results_csv(text) -> list[HistoricalMatch]` — **puro**, stdlib `csv` (sin
  pandas → testeable sin dependencias pesadas).

### `features/elo.py` (nuevo) — funciones puras, sin I/O, sin RNG
- `expected_score(r_home, r_away, home_adv) -> float`
- `goal_margin_multiplier(margin, cfg) -> float`
- `recency_weight(match_date, reference_date, half_life_months) -> float`
- `classify_importance(tournament) -> str` — reglas por palabra clave → clave de
  `k_factors`. Documentada y ajustable.
- `fit_elo(matches, config, reference_date=None) -> dict[str, float]` — ajuste
  cronológico; devuelve `team → rating`. Determinista.

### `config.yaml`
- **Añadir** `elo.goal_margin: {two_goal: 1.5, offset: 11.0, divisor: 8.0}`.
- **Mantener** `recency_half_life_months: 18.0` (ahora consumido).

## Flujo de datos

`fetch_martj42 → parse_results_csv → fit_elo → {team: rating}`

## Mapeo de importancia (borrador, ajustable)

| Patrón en `tournament` | Clave `k_factors` |
|---|---|
| contiene "World Cup" + "qualification" | `world_cup_qualifier` |
| contiene "World Cup" (no qualification) | `world_cup` |
| contiene "Nations League" | `nations_league` |
| contiene "Friendly" | `friendly` |
| Euro / Copa América / Gold Cup / Asian Cup / African Cup / Confederations | `continental` |
| resto | `default` |

## Nombres de equipo

martj42 es consistente internamente. El puente a nombres 2026 (p.ej. "USA" ↔ "United
States") reutiliza `clean.apply_team_aliases` y se ejerce al consumir ratings en la
simulación (Fase 4). Fase 2 ajusta sobre los nombres de martj42.

## Tests (TDD — primero los tests)

- `expected_score`: iguales+neutral → 0.5; localía sube `E_home`; simétrico.
- `goal_margin_multiplier`: `d=1→1`, `d=2→1.5`, `d=3→1.75`; empate → 1.
- `recency_weight`: edad 0 → 1; 18 meses → 0.5; más viejo → menor; anclaje determinista.
- update: ganador sube = perdedor baja (G=1, recency=1); upset mueve más; mayor margen
  mueve más.
- `fit_elo`: determinista; orden cronológico; fixture pequeño con orden de ratings esperado.
- `classify_importance`: strings de torneo conocidos → clave correcta.
- `parse_results_csv`: CSV mínimo → records (stdlib).

## Reglas

Sin números mágicos en código de modelo (todo en `config.yaml`). Determinismo: mismos
inputs → mismos ratings (Elo no usa aleatoriedad). Dependencias en una dirección
(`features` puede importar de `data`, no al revés).
