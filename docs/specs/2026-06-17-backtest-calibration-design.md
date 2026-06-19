# Fase 5 — Backtest + calibration (design)

> Fecha: 2026-06-17. Estado: aprobado. Implementación: TDD.

## Objetivo

Validar el **modelo de partido** (1X2 de Dixon-Coles) en walk-forward sobre el histórico
martj42, y medir/corregir su calibración. Unidad evaluable = el 1X2 a nivel partido (miles
de muestras); el Monte Carlo del torneo no es backtesteable (un Mundial = una muestra).

## Decisiones (aprobadas)

- **Ratings walk-forward = refit fiel con recencia, por ventana.** Para cada partido
  histórico M se predice con el MODELO QUE SE PUBLICA: `fit_elo` sobre la ventana
  `[M.date − history_window_days, M.date)` anclada a `M.date`. La media-vida de 18 meses
  hace despreciable (<3%) lo anterior a ~8 años, así que truncar a la ventana es casi sin
  pérdida y vuelve factible el coste. Sin peeking: solo partidos previos.
- **Métricas (set completo):** `log_loss`, `brier` (multiclase), `rps` (ranked probability
  score, respeta el orden local>empate>visita), `accuracy`.
- **Calibración:** curva de fiabilidad + ECE, y recalibración **Platt/logística por
  resultado** (una logística por clase home/draw/away, luego renormalizar a sumar 1).

## Componentes

### `evaluation/backtest.py`
- `Prediction` (frozen): `probs=(p_home,p_draw,p_away)`, `actual ∈ {0,1,2}` (orden
  home/draw/away).
- Métricas puras sobre `list[Prediction]` (numpy/stdlib): `log_loss` (con clip),
  `brier`, `rps` (acumulada), `accuracy`. `Metrics` (dataclass) las agrupa.
- `backtest(history, model, elo_cfg, *, burn_in_matches, history_window_days)
  -> BacktestResult(predictions, metrics)`: walk-forward con `fit_elo` por ventana
  (bisect sobre fechas para acotar la ventana), `home_advantage` salvo `neutral`.
  Determinista (sin RNG).

### `evaluation/calibration.py`
- `reliability_bins(predictions, n_bins) -> [(mean_pred, observed_freq, count)]` (pooled
  por clase: prob predicha de cada resultado vs si ocurrió) y `ece(...)` (número titular).
  Puro numpy.
- `fit_platt(predictions) -> PlattCalibrator`: una `LogisticRegression` (sklearn,
  import perezoso) por clase; `.calibrate(probs)` aplica las tres y renormaliza a 1.

### `config.yaml`
Añadir `evaluation.history_window_days: 2920` (~8 años). Mantener `burn_in_matches: 150`.

## Testing (TDD)

- Métricas en casos a mano: predicción perfecta → log_loss/brier/rps = 0, accuracy = 1;
  RPS penaliza "predijo visita cuando fue local" MÁS que "predijo empate" (prueba el orden).
- `backtest`: no-peek (añadir un partido posterior no cambia las predicciones previas) +
  determinismo.
- `reliability_bins`/`ece` en datos construidos (calibrado → ECE~0; sesgado → ECE>0).
- Platt (`importorskip` sklearn): `calibrate` devuelve probs válidas que suman 1 y no
  empeora el log-loss en datos miscalibrados.

## Notas

Las 4 métricas de puntuación (incl. Brier) viven juntas en `backtest.py` (se computan del
mismo conjunto de predicciones); `calibration.py` se queda con fiabilidad + el mapa de
recalibración. `evaluation` puede importar `features`/`models`/`data`/`config`, no
`viz`/`app`. sklearn solo en el fit de Platt (import perezoso; test con `importorskip`).
