# worldcup2026-predictor

Predictor **live-updating** del Mundial FIFA 2026 (48 equipos, 12 grupos). Recondiciona
las probabilidades con resultados reales conforme ocurren y produce visuales para
YouTube Shorts / Reels. Modelo: **Elo dinámico → Dixon-Coles → Monte Carlo condicional**.

> Contexto completo y reglas: [`PROJECT.md`](PROJECT.md). Fuentes de datos: [`SOURCES.md`](data/raw/SOURCES.md).

## Quickstart

```bash
make setup                       # crea .venv e instala deps
cp .env.example .env             # añade tu API_FOOTBALL_KEY
make run                         # pipeline live completo
make test && make lint           # calidad
```

Reproducibilidad: dado **el mismo snapshot + semilla**, la salida es idéntica. Para
grabar un video sin que cambie, fija un snapshot:

```bash
python scripts/run_pipeline.py --snapshot 20260616t1830 --runs 50000
```

## Estado de construcción

- [x] **Fase 0** — Andamiaje: `pyproject.toml`, `Makefile`, `config/config.yaml`,
  `.env.example`, `src/worldcup/{config.py,rng.py}`, CI, pre-commit.
- [x] **Fase 1** — Datos + LIVE: interfaz `LiveResultsProvider` + esquema normalizado,
  cliente `APIFootballProvider`, `schedule.py` (backbone openfootball + `validate_schedule`),
  snapshotting con timestamp (`download.py`), `clean.py` (validación + reconciliación),
  `triggers.py` (`WatchTrigger`/`CronTrigger`). Pendiente: cliente `FootballDataProvider`
  (fallback) y fetch del histórico martj42 (se hará junto al Elo).
- [x] **Fase 2** — Elo dinámico: `data/historical.py` (fetch + parse martj42),
  `features/elo.py` (expectativa logística, multiplicador de margen eloratings.net,
  recencia 18m, clasificación de importancia, `fit_elo` secuencial determinista).
  Diseño en `docs/specs/2026-06-16-elo-design.md`.
- [ ] Fases 3–8 — Dixon-Coles, simulación condicional, evaluación, viz, CLI,
  dashboard, notebook. Ver el plan en `PROJECT.md`.

## Arquitectura

Separación estricta de capas (`data` → `features` → `models` → `simulation` →
`evaluation` → `viz`); dependencias en una sola dirección. Toda la aleatoriedad pasa
por `worldcup.rng.get_rng()`. Ver el árbol completo en `PROJECT.md §4`.
