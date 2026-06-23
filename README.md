# worldcup2026-predictor

Predictor **live-updating** del Mundial FIFA 2026 (48 equipos, 12 grupos). Recondiciona
las probabilidades con resultados reales conforme ocurren y produce visuales para
YouTube Shorts / Reels. Modelo: **Elo dinámico → Dixon-Coles → Monte Carlo condicional**.

> Contexto completo y reglas: [`PROJECT.md`](PROJECT.md). Fuentes de datos: [`SOURCES.md`](data/raw/SOURCES.md).

## Quickstart

```bash
make setup                        # crea .venv e instala deps (pyproject.toml)
make run                          # pipeline live completo (necesita FOOTBALL_DATA_TOKEN)
streamlit run app/dashboard.py    # dashboard interactivo (lee la última corrida)
```

Sin clave de API (no toca resultados live) puedes generar el baseline pre-torneo, que
descarga el calendario (openfootball) y el histórico (martj42):

```bash
python scripts/run_pipeline.py --mode pre_tournament --runs 50000
```

## Comandos

```bash
# Predicción puntual de un partido (1X2), usa el histórico martj42.
# Los nombres deben coincidir con martj42 (p.ej. "United States", no "USA").
python scripts/predict_match.py "Brazil" "France"
python scripts/predict_match.py "United States" "Mexico" --host "United States"

# Reproducir EXACTAMENTE un estado (para grabar un video sin que cambie)
python scripts/run_pipeline.py --snapshot 20260616t1830 --runs 50000

# Loop de refresco mientras hay partidos (polling por ventanas; ver triggers.py)
python scripts/run_pipeline.py --watch --interval 600

# Calidad (toolchain canónico = el venv / make)
make test                         # pytest
make lint                         # ruff + black --check + mypy
make fmt                          # black + ruff --fix
```

Refresco automático y robusto (recomendado sobre `--watch`): el workflow
[`.github/workflows/refresh.yml`](.github/workflows/refresh.yml) corre el pipeline en un
schedule de GitHub Actions y publica las figuras + el JSON como artefactos.

## Qué produce una corrida

- `data/processed/probabilities_<ts>.json` (+ `latest.json`): por equipo,
  `P(avance / octavos / cuartos / semis / final / campeón)` y los grupos. El puntero
  `latest.json` habilita los deltas ↑/↓ vs la corrida previa.
- `outputs/figures/*.png`: ranking de campeón, barra 1X2, heatmap de marcadores, bracket,
  tabla de grupos y diagrama de fiabilidad (1080×1920 vertical / 1920×1080 horizontal).
- `outputs/videos/*.mp4`: animación del ranking (el "drumroll" del campeón).
- `data/raw/results_<ts>.parquet`: snapshot inmutable — el registro de *qué sabía el modelo
  y cuándo*.

**Reproducibilidad:** dado el mismo snapshot + la misma semilla, la salida es idéntica. En
modo `live` las figuras cambian al entrar un resultado nuevo (es lo esperado); para grabar,
fija `--snapshot <ts>`.

## Modelo y fases

Todas las fases están construidas, cada una revisada de forma adversarial y endurecida.

- [x] **Fase 0** — Andamiaje: `pyproject.toml`, `Makefile`, `config/config.yaml`, CI, RNG.
- [x] **Fase 1** — Datos + LIVE: `LiveResultsProvider`/`APIFootballProvider`, backbone de
  calendario (openfootball), snapshotting con timestamp, validación + reconciliación,
  `WatchTrigger`/`CronTrigger`.
- [x] **Fase 2** — Elo dinámico: histórico martj42, multiplicador de margen eloratings.net,
  recencia 18m, `fit_elo` secuencial determinista.
- [x] **Fase 3** — Dixon-Coles: Elo→goles simétrico, matriz Poisson, corrección τ.
- [x] **Fase 4** — Monte Carlo condicional: Annex C, desempates Art. 13 (H2H recursivo),
  prórroga + penales, simula solo lo pendiente → P(ronda/campeón).
- [x] **Fase 5** — Backtest + calibración: walk-forward recency-fiel (log-loss/Brier/RPS),
  fiabilidad/ECE, recalibración Platt.
- [x] **Fase 6** — Visualización: `theme` (marca única), charts/bracket/tabla, export
  PNG + MP4.
- [x] **Fase 7** — Pipeline end-to-end: `pipeline.py` (núcleo puro) + CLIs; `reconcile`
  alimenta `build_state`; salida JSON + figuras.
- [x] **Fase 8** — Dashboard Streamlit (`app/dashboard.py`) sobre los artefactos persistidos.

Diseños por fase en [`docs/specs/`](docs/specs/).

## Arquitectura

Separación estricta de capas (`data → features → models → simulation → evaluation → viz`),
con dependencias en una sola dirección; el I/O vive en `data`, `scripts` y `app`. Cada capa
es un **núcleo puro testeado** envuelto en glue fino de I/O. Toda la aleatoriedad pasa por
`worldcup.rng.get_rng()` (determinismo por semilla). Ver el árbol completo en `PROJECT.md §4`.
