# Fase 3 — Dixon-Coles match model (design)

> Fecha: 2026-06-16. Estado: aprobado. Implementación: TDD.

## Objetivo

Dado el rating Elo de dos selecciones, producir la **matriz de marcadores** (P(home=i,
away=j)) y de ahí las probabilidades 1X2. Es el modelo de partido que consume la
simulación (Fase 4). Interfaz estable `MatchModel` para poder enchufar el alternativo
XGBoost más adelante.

## Decisiones

- **Elo → goles esperados (símetrico, consistente con Fase 2):**
  `d = (R_home + HA) − R_away`; `λ_home = clip(base + d/denom, [min,max])`,
  `λ_away = clip(base − d/denom, [min,max])`. Una sola aplicación de HA, igual que el
  `expected_score` del Elo. (`base=1.35`, `denom=400`, `[min,max]=[0.3,3.5]` de config.)
- **Matriz Poisson independiente** de tamaño `(max_goals+1)²` (9×9). La pmf de Poisson se
  calcula con numpy + stdlib (`exp`, `factorial`) — no se trae scipy para una pmf de 9
  puntos (scipy queda para el posible ajuste de ρ en el backtest, Fase 5).
- **Corrección Dixon-Coles τ** sobre las 4 celdas de marcador bajo, con `ρ=−0.13`:
  `τ(0,0)=1−λμρ`, `τ(0,1)=1+λρ`, `τ(1,0)=1+μρ`, `τ(1,1)=1−ρ`. Tras aplicarla (y por el
  truncado a `max_goals`), se **normaliza** la matriz para que sume 1.
- **Alcance:** solo tiempo reglamentario (matriz + 1X2 + goles esperados). La resolución
  de empates en eliminatoria (prórroga/penales con `extra_time_total_goals`) vive en
  `simulation/match.py` (Fase 4), no aquí.

## Componentes

### `models/base.py`
- `MatchOutcome` (dataclass frozen): `home_win, draw, away_win` (suman 1).
- `outcome_probabilities(score_matrix) -> MatchOutcome` — agrega la matriz (triángulo
  inferior = victoria local, diagonal = empate, superior = visitante).
- `sample_scoreline(score_matrix, rng) -> tuple[int, int]` — muestrea un marcador con el
  RNG sembrado del proyecto (`worldcup.rng`); para la simulación (Fase 4).
- `MatchModel(ABC)`: método abstracto `score_matrix(...)`; método por defecto
  `outcome_proba(...)` derivado de `score_matrix` (los modelos lo heredan gratis).

### `models/dixon_coles.py`
- `DixonColesModel(elo_cfg, dc_cfg)` implementa `MatchModel`.
- `expected_goals(R_home, R_away, home_advantage=0.0) -> (λ_home, λ_away)`.
- `score_matrix(...)` -> matriz 9×9 normalizada con la corrección τ.

## Testing (TDD — primero los tests)

- `outcome_probabilities`: matriz conocida -> home/draw/away correctos y suman 1.
- `sample_scoreline`: reproducible con RNG sembrado; devuelve índices válidos.
- `expected_goals`: iguales+neutral -> ambos λ = base; local más fuerte -> λ_home>λ_away;
  respeta los clips [min,max].
- `score_matrix`: forma 9×9; suma 1; las 4 celdas τ difieren del producto Poisson crudo
  en el signo correcto (ρ<0 sube 0-0 y 1-1, baja 1-0 y 0-1); equipo más fuerte ->
  `outcome_proba.home_win` mayor; simetría en cancha neutral al intercambiar equipos.

## Reglas

Sin números mágicos (todo de `config.yaml`). `models` no importa de `viz`/`app`/
`simulation`; sí puede importar de `config`. Aleatoriedad solo vía `worldcup.rng`.
Determinismo: misma matriz + misma semilla -> mismo marcador muestreado.
