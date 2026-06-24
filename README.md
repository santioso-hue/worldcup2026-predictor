# worldcup2026-predictor

Predictor del Mundial 2026 que se actualiza en vivo (48 selecciones, 12 grupos). A medida
que se juegan los partidos, recalcula las probabilidades con los resultados reales y arma
los gráficos para Shorts y Reels. El modelo combina **Elo dinámico → Dixon-Coles → Monte
Carlo condicional**.

> Contexto y reglas en [`PROJECT.md`](PROJECT.md). De dónde salen los datos: [`SOURCES.md`](data/raw/SOURCES.md).

## Para arrancar

```bash
make setup                        # crea el .venv e instala las dependencias
make run                          # corrida live completa (necesita FOOTBALL_DATA_TOKEN)
streamlit run app/dashboard.py    # tablero interactivo (lee la última corrida)
```

Si no hay clave de API, igual se puede correr el baseline previo al torneo, que baja el
calendario (openfootball) y el histórico (martj42):

```bash
python scripts/run_pipeline.py --mode pre_tournament --runs 50000
```

## Comandos

```bash
# Pronóstico de un partido suelto (1X2), con el histórico de martj42.
# Los nombres tienen que coincidir con martj42 (p.ej. "United States", no "USA").
python scripts/predict_match.py "Brazil" "France"
python scripts/predict_match.py "United States" "Mexico" --host "United States"

# Reproducir un estado exacto (para grabar un video sin que cambien los números)
python scripts/run_pipeline.py --snapshot 20260616t1830 --runs 50000

# Refrescar en bucle mientras hay partidos (sondea por ventanas; ver triggers.py)
python scripts/run_pipeline.py --watch --interval 600

# Calidad (todo corre dentro del .venv, vía make)
make test                         # pytest
make lint                         # ruff + black --check + mypy
make fmt                          # black + ruff --fix
```

Para algo más estable que `--watch`, el workflow
[`.github/workflows/refresh.yml`](.github/workflows/refresh.yml) corre el pipeline con
GitHub Actions en un horario fijo y sube las figuras y el JSON como artefactos.

## Qué deja cada corrida

- `data/processed/probabilities_<ts>.json` (y `latest.json`): por selección, la
  probabilidad de avanzar y de llegar a octavos, cuartos, semis, final y título, más los
  grupos. El puntero `latest.json` sirve para mostrar las flechas ↑/↓ contra la corrida
  anterior.
- `outputs/figures/*.png`: ranking de campeón, barra 1X2, mapa de calor de marcadores,
  bracket, tabla de grupos y curva de fiabilidad (1080×1920 vertical o 1920×1080 horizontal).
- `outputs/videos/*.mp4`: la animación que va revelando al campeón.
- `data/raw/results_<ts>.parquet`: el snapshot inmutable, el registro de *qué sabía el
  modelo y cuándo*.

**Reproducibilidad:** con el mismo snapshot y la misma semilla, la salida es idéntica. En
modo `live` las figuras cambian cuando entra un resultado nuevo (es lo que se espera); para
grabar, se fija `--snapshot <ts>`.

## El modelo, por fases

Están todas hechas, y cada una pasó por revisión y pruebas.

- [x] **Fase 0** — Base del proyecto: `pyproject.toml`, `Makefile`, `config/config.yaml`,
  CI y RNG.
- [x] **Fase 1** — Datos y live: `LiveResultsProvider`/`FootballDataProvider`, el calendario
  base (openfootball), snapshots con timestamp, validación y reconciliación,
  `WatchTrigger`/`CronTrigger`.
- [x] **Fase 2** — Elo dinámico: histórico de martj42, multiplicador por margen de gol
  (eloratings.net), peso por recencia y `fit_elo` secuencial y determinista.
- [x] **Fase 3** — Dixon-Coles: de Elo a goles, matriz de Poisson y corrección τ para
  marcadores bajos.
- [x] **Fase 4** — Monte Carlo condicional: Anexo C, desempates del Art. 13 (enfrentamiento
  directo recursivo), prórroga y penales; simula solo lo que falta por jugar y da
  P(ronda/título).
- [x] **Fase 5** — Backtest y calibración: walk-forward fiel a la recencia (log-loss/Brier/
  RPS), fiabilidad/ECE y recalibración de Platt.
- [x] **Fase 6** — Gráficos: tema visual propio, charts/bracket/tabla y export a PNG y MP4.
- [x] **Fase 7** — Pipeline de punta a punta: `pipeline.py` (núcleo puro) más las CLIs;
  `reconcile` alimenta a `build_state`; salida en JSON y figuras.
- [x] **Fase 8** — Tablero en Streamlit (`app/dashboard.py`) sobre los artefactos guardados.

Los diseños de cada fase están en [`docs/specs/`](docs/specs/).

## Arquitectura

Las capas están separadas y las dependencias van en un solo sentido:
`data → features → models → simulation → evaluation → viz`. El I/O vive en `data`, `scripts`
y `app`; cada capa es un núcleo puro y testeado, envuelto en una capa fina de I/O. Toda la
aleatoriedad pasa por `worldcup.rng.get_rng()`, así que con la misma semilla siempre sale lo
mismo. El árbol completo está en `PROJECT.md §4`.
