# PROJECT.md — World Cup 2026 Predictor

> Cómo está armado el proyecto: objetivo, decisiones de diseño, arquitectura y fuentes de datos.

---

## 1. Objetivo del proyecto

Construir un **predictor del Mundial FIFA 2026** (48 equipos, 12 grupos) que:

1. **Prediga en vivo:** recondicione las probabilidades con **resultados reales** a
   medida que ocurren (modelo *live-updating*), no un forecast congelado. Simula con
   Monte Carlo solo lo que falta por jugar.
2. Produzca **visualizaciones bonitas y consistentes** (probabilidades, bracket,
   ranking de campeón) en formato vertical (1080×1920) y horizontal, más un
   **dashboard interactivo** que muestre la última actualización.
3. Sea **reproducible y explicable**: dado un snapshot de resultados fijo + semilla, el
   pipeline es determinista, sin "caja negra".

El output está pensado para gente no técnica; el código, para alguien que quiere
entender el modelo de un vistazo.

---

## 2. Decisiones de diseño

- **Lenguaje:** Python 3.11+ (mejor ecosistema de ML y visualización).
- **Modelo primario:** `Elo dinámico → Dixon-Coles (Poisson bivariado) → Monte Carlo`.
  Transparente, calibrado y fácil de explicar. **Es el default.**
- **Modelo alternativo:** clasificador ML (XGBoost) detrás de la **misma interfaz**
  (`models/base.py`) para comparar en el backtest. No es el default.
- **Modo de torneo:** configurable en `config/config.yaml`. **El enfoque del proyecto
  es `live`** (el Mundial está en curso: 11 jun – 19 jul 2026):
  - `mode: live` (**DEFAULT y enfoque principal**) → recondiciona con resultados reales:
    bloquea partidos finalizados, colapsa equipos eliminados a 0%, usa el bracket real
    (incl. mejores terceros) y **solo simula los partidos restantes**. Se re-ejecuta tras
    cada resultado nuevo. Más preciso porque integra información que ya ocurrió.
  - `mode: pre_tournament` (solo fallback) → forecast congelado al kickoff, sin
    recondicionar. Útil como baseline para el backtest y para comparar "lo que el modelo
    decía antes" vs. la realidad. **No es el enfoque del proyecto.**
- **Determinismo dentro del modo live:** la aleatoriedad de Monte Carlo es reproducible
  **dado un snapshot de resultados fijo + semilla**. Cada descarga de resultados se guarda
  como snapshot con timestamp (`data/raw/results_YYYYMMDDtHHMM.parquet`) y el output
  registra qué snapshot usó. Para reproducir un estado exacto, **fija un snapshot**
  (`--snapshot <ts>`) y obtendrás siempre las mismas figuras. Nunca uses
  `random`/`np.random` sin el RNG sembrado del proyecto.

---

## 3. Stack técnico

| Capa | Herramienta |
|------|-------------|
| Datos / cómputo | `pandas`, `numpy`, `pyarrow` |
| Modelo estadístico | `scipy` (Poisson/optimización Dixon-Coles) |
| Modelo ML (opcional) | `xgboost`, `scikit-learn` |
| Calibración / métricas | `scikit-learn` (Brier, log-loss, Platt/isotónica) |
| Viz estática | `matplotlib` (export PNG/MP4) |
| Viz interactiva | `plotly` |
| Dashboard | `streamlit` |
| Config | `pydantic` + `pyyaml` |
| CLI | `typer` |
| Tests / lint | `pytest`, `ruff`, `black`, `mypy` |
| Reproducibilidad | `Makefile`, semilla fija, datos versionados |

---

## 4. Arquitectura de carpetas

```
worldcup2026-predictor/
├── PROJECT.md                  # este archivo
├── README.md                  # quickstart para humanos
├── pyproject.toml             # deps + config de ruff/black/pytest/mypy
├── Makefile                   # pipeline reproducible de 1 comando
├── .pre-commit-config.yaml
├── config/
│   └── config.yaml            # ÚNICA fuente de hiperparámetros, rutas, seed, mode
├── data/
│   ├── raw/                   # descargado, INMUTABLE (no editar a mano)
│   │   ├── SOURCES.md         # fuentes + decisión de API live + presupuesto de requests
│   │   └── results_*.parquet  # snapshots de resultados con timestamp (+ puntero latest)
│   ├── interim/               # limpio
│   └── processed/             # features listos para el modelo
├── src/worldcup/
│   ├── config.py              # carga/valida config.yaml con pydantic
│   ├── rng.py                 # RNG sembrado central (úsalo SIEMPRE)
│   ├── data/
│   │   ├── download.py        # histórico (martj42) + resultados LIVE; snapshots con ts
│   │   ├── live_results.py    # interfaz LiveResultsProvider + cliente football-data.org
│   │   ├── triggers.py        # RefreshTrigger: watch (ahora) -> cron / webhook (mejora)
│   │   ├── clean.py
│   │   └── schedule.py        # fixtures 2026: 12 grupos, mejores terceros
│   ├── features/
│   │   ├── elo.py             # Elo dinámico (k por importancia de partido)
│   │   └── build_features.py
│   ├── models/
│   │   ├── base.py            # interfaz MatchModel (predict_scoreline / probs)
│   │   ├── dixon_coles.py     # Poisson bivariado (PRIMARIO)
│   │   └── xgboost_model.py   # ML alternativo (misma interfaz)
│   ├── simulation/
│   │   ├── state.py           # estado real: finalizados, eliminados, bracket vigente
│   │   ├── match.py           # muestrea un partido
│   │   ├── group_stage.py     # tabla, desempates, mejores terceros (matching bipartito)
│   │   └── tournament.py      # Monte Carlo condicional (solo simula lo pendiente)
│   ├── evaluation/
│   │   ├── backtest.py        # accuracy/log-loss en torneos pasados
│   │   └── calibration.py     # Brier, reliability, Platt/isotónica
│   └── viz/
│       ├── theme.py           # paleta, tipografías, layout (1 sola fuente de marca)
│       ├── charts.py          # barras de probabilidad, heatmaps de marcador
│       ├── bracket.py         # render del bracket
│       └── export.py          # PNG 1080×1920 / 1920×1080, stitch a MP4
├── app/
│   └── dashboard.py           # Streamlit
├── scripts/
│   ├── run_pipeline.py        # download -> clean -> features -> fit -> simulate -> viz
│   └── predict_match.py       # CLI: python scripts/predict_match.py Brazil France
├── tests/
│   ├── test_elo.py
│   ├── test_dixon_coles.py
│   ├── test_simulation.py
│   └── test_calibration.py
├── outputs/
│   ├── figures/               # PNG generados (versionar solo los "finales")
│   └── videos/                # MP4/GIF de las animaciones
└── .github/workflows/ci.yml   # lint + tests en cada push
```

### Principios de arquitectura
- **Separación de responsabilidades:** `data` != `features` != `models` != `simulation`
  != `evaluation` != `viz`. No mezcles lógica de simulación dentro de viz, etc.
- **Dependencia en una sola dirección:** `viz` y `simulation` dependen de `models`;
  `models` no importa de `viz` ni de `app`.
- **Config sobre constantes mágicas:** ningún número "a mano" en el código de modelo;
  va en `config.yaml`.
- **Interfaz estable:** cualquier modelo nuevo implementa `models/base.MatchModel`.
- **Datos crudos inmutables:** nunca sobrescribas `data/raw/`.

---

## 5. Metodología del modelo

**Pipeline:** `Elo → tasas de gol esperadas → Dixon-Coles → Monte Carlo`.

1. **Elo dinámico** (`features/elo.py`): cada selección tiene un rating que se
   actualiza tras cada partido. El factor `k` pondera por importancia
   (Mundial > clasificatorio > amistoso) y el margen de goles. Localía/sede como bono.
2. **Tasas de gol** (`models/dixon_coles.py`): se mapea la diferencia de Elo a
   `λ_home`, `λ_away` (goles esperados). Dixon-Coles corrige la dependencia en
   marcadores bajos (0-0, 1-0, 0-1, 1-1) que el Poisson simple modela mal.
3. **Probabilidades de partido:** matriz de marcadores → P(victoria local), P(empate),
   P(visita). En fase eliminatoria, si hay empate → prórroga/penales (modelo aparte).
4. **Monte Carlo condicional** (`simulation/tournament.py`): se simula `N` veces
   (típico 50_000) **solo la parte no jugada** del torneo. Se agregan frecuencias →
   P(avanzar de grupo), P(llegar a la final), P(campeón) por selección.

### 5.1 Reconditioning live (núcleo del proyecto)

Antes de cada simulación, `simulation/tournament.py` aplica el estado real del torneo:

- **Partidos finalizados → bloqueados:** su marcador real es un hecho, no se muestrea.
  Probabilidad de esos resultados = 1.
- **Equipos eliminados → colapsan a 0%** en todas las rondas futuras.
- **Bracket real:** se usa el cruce eliminatorio efectivo, incluido el reparto de los
  **mejores terceros** (resolver con *matching* bipartito, como en el repo de Hicruben,
  para asignar terceros a llaves según la tabla FIFA).
- **Elo recondicionado:** los resultados ya jugados actualizan los ratings antes de
  simular lo que falta (no se usa el Elo "pre-torneo" para partidos posteriores).
- **Solo se simulan los partidos pendientes** dado ese estado. El resto es determinista.
- **Disparador de re-ejecución:** tras cada full-time, `download.py` detecta el resultado
  nuevo, crea un snapshot y se vuelve a correr el pipeline. **Empezar con `--watch`**
  (loop que sondea cada N min). **Camino de mejora (documentar, no quedarse en watch):**
  - `cron` / systemd timer / GitHub Actions schedule: re-corre el pipeline en intervalos
    fijos sin un proceso vivo; más robusto que `--watch` ante caídas.
  - **Webhook / push:** si el proveedor lo ofrece, dispara el refresh al instante en que
    termina un partido (latencia mínima, cero polling desperdiciado).
  El disparador vive detrás de una interfaz (`RefreshTrigger`) para cambiar
  watch → cron → webhook sin tocar el pipeline.

El feed de resultados live puede retrasarse o traer datos parciales;
`clean.py` debe validar (marcador, estado finished/in-play/scheduled) y, ante datos
sospechosos, **conservar el último snapshot válido** en vez de degradar la predicción.

**Por qué este modelo:** es transparente (cada probabilidad es trazable), está bien
calibrado, se explica en 60 segundos, y al condicionar sobre resultados reales mejora su
precisión conforme avanza el torneo. El ML (XGBoost) queda como alternativa enchufable
para comparar accuracy/log-loss en el backtest, no como default.

---

## 6. Fuentes de datos

- **Resultados internacionales:** `martj42/international_results` (GitHub/Kaggle, libre,
  ~45k+ partidos desde 1872). Base de Elo y backtest.
- **Calendario WC2026:** `openfootball/worldcup.json` (fixtures, 12 grupos, sedes).
- **Ranking FIFA:** opcional como prior de Elo inicial.
- **Resultados live (REQUERIDO, no opcional):** **football-data.org** (v4, free forever,
  cubre el WC2026 sin límite de temporada; auth `X-Auth-Token`): un solo GET a
  `/competitions/WC/matches` trae los 104 partidos con su estado.
  Schedule: **openfootball/worldcup.json**.
  La decisión completa, el presupuesto de requests y los puntos a verificar están en
  **`data/raw/SOURCES.md`**.

**Reglas de datos:**
- Toda fuente live se accede detrás de la interfaz `data/live_results.LiveResultsProvider`
  para poder cambiar de proveedor sin tocar el modelo.
- Cita la fuente y fecha de descarga en `data/raw/SOURCES.md`.
- `download.py` debe ser idempotente y cachear; cada corrida guarda un **snapshot con
  timestamp** (`results_YYYYMMDDtHHMM.parquet`) y deja un puntero `latest`. No machaques
  snapshots anteriores: son el registro reproducible de "qué sabía el modelo y cuándo".
- **Polling solo por ventanas** (mientras hay partidos en juego), cada 10–15 min; cachea
  todo lo que cambia lento (teams, standings). Nunca polling continuo.
- Valida el estado de cada partido (`scheduled` / `in_play` / `finished`) antes de
  bloquearlo. Ante datos parciales o sospechosos, conserva el último snapshot válido.

---

## 7. Visualización

Todo el branding vive en `viz/theme.py` (una sola fuente de verdad). No hardcodees
colores/fuentes en otros módulos.

Entregables visuales mínimos:
- **Barra de probabilidad** de un partido (local/empate/visita) con escudos/colores.
- **Ranking de campeón** (top-10 selecciones por P(título)) — la figura estrella.
  Incluye **delta vs. snapshot anterior** (↑/↓): resalta cómo cambian las
  probabilidades en vivo cuando entra un resultado nuevo.
- **Bracket** eliminatorio renderizado (marca llaves ya resueltas con el resultado real).
- **Tabla de grupo** con P(avance) y partidos ya jugados resaltados.
- **Heatmap de marcadores** más probables de un partido.
- **Sello de "última actualización"** (timestamp del snapshot) en toda figura live, para
  que quien la vea sepa que el número es de ese momento.

Reglas de export (`viz/export.py`):
- Vertical **1080×1920** y horizontal **1920×1080**.
- PNG a 150+ dpi, fondo y tipografía consistentes con `theme.py`.
- Opción de animar la simulación (frames PNG → MP4) para mostrar cómo se va definiendo
  el campeón.
- Cada figura se guarda en `outputs/figures/` con nombre descriptivo y determinista.

Estilo: limpio, alto contraste, legible en móvil (texto grande), sin sobrecargar.

---

## 8. Comandos

```bash
# Setup
make setup                 # crea venv e instala deps (pyproject.toml)

# Pipeline completo en modo live (descarga resultados -> recondiciona -> simula -> figuras)
make run                   # == python scripts/run_pipeline.py --config config/config.yaml

# Refrescar predicción con los últimos resultados (uso principal durante el torneo)
make refresh               # baja nuevo snapshot y re-simula solo lo pendiente
# Loop automático (re-corre cada N minutos mientras hay partidos)
python scripts/run_pipeline.py --watch --interval 600
# Mejora recomendada (no quedarse en watch): cron / systemd timer / GitHub Actions
#   schedule para robustez ante caídas; webhook del proveedor para refresh instantáneo
#   al terminar cada partido. Ver RefreshTrigger en src/worldcup/data/triggers.py.

# Reproducir EXACTAMENTE un estado (para que los números no cambien)
python scripts/run_pipeline.py --snapshot 20260616t1830 --runs 50000

# Predicciones puntuales (usan el último snapshot por defecto)
python scripts/predict_match.py "Brazil" "France"     # head-to-head
python scripts/predict_match.py "USA" "Mexico" --host "USA"

# Forecast congelado pre-torneo (baseline para el backtest, NO es el enfoque)
python scripts/run_pipeline.py --mode pre_tournament --simulate --runs 50000

# Dashboard interactivo (muestra timestamp de "última actualización")
streamlit run app/dashboard.py

# Calidad
make test                  # pytest
make lint                  # ruff + black --check + mypy
make fmt                   # black + ruff --fix
```

Determinismo: dado **el mismo snapshot de resultados + semilla**, la salida es idéntica.
En modo `live`, las figuras cambian cuando entra un resultado nuevo (es lo esperado); para
reproducir un estado exacto, fija `--snapshot <ts>`.

---

## 9. Convenciones de código

- **Estilo:** `black` (88 cols) + `ruff`. Type hints obligatorios; `mypy` debe pasar.
- **Docstrings** estilo NumPy en funciones públicas (incluye unidades y supuestos).
- **Nombres** en inglés para código; comentarios pueden ir en español.
- **Funciones puras** en `models`/`simulation` (sin I/O ni efectos colaterales);
  el I/O vive en `data`, `scripts` y `app`.
- **Sin notebooks como fuente de verdad:** la lógica vive en `src/`.
- **Tests primero para lógica de simulación** (desempates, mejores terceros, penales):
  son donde más se cuelan bugs.
- **Reproducibilidad:** toda aleatoriedad pasa por `worldcup.rng.get_rng()`.

## 10. Qué NO hacer

- No meter números mágicos en el código de modelo; van en `config.yaml`.
- No acoplar `viz`/`app` dentro de `models`/`simulation`.
- No usar `random`/`np.random` global sin el RNG sembrado.
- No sobrescribir `data/raw/`.
- No commitear la API key: va en variable de entorno (`.env`, ignorado en git) y se usa
  server-side. `.env.example` documenta las variables sin valores reales.
- No hacer polling continuo del feed live: solo por ventanas y con caché (ver `SOURCES.md`).

## 11. Checklist por cambio

1. Código tipado, formateado y con docstring.
2. Test que cubra el caso normal + un edge case.
3. `make lint && make test` en verde.
4. Si afecta el output: figura regenerada y revisada visualmente.
5. README/PROJECT.md actualizados si cambian comandos o estructura.

---

## 12. Repos de referencia

- `Hicruben/world-cup-2026-prediction-model` — Elo + Dixon-Coles + Monte Carlo,
  transparente y reproducible. **Referencia principal de metodología.**
- `Currybon30/fifa_wc_2026_datacamp` — ML + Elo + Monte Carlo, modelos `.pkl`.
- `EhteshamBahoo/Fifa-WorldCup-Data-Analysis-1930-2026` — scraping Wikipedia + RF + export CSV.
- `neelabhsinha/fifa-world-cup-prediction-ml` — clasificación + clustering de grupos.
- `rivu-intel45/FIFA-2026-Winner-Prediction`, `ysadre/FifaWC_Predictions`,
  `e7alves/world_cup_predictor` — variantes de pipeline/predicción.

Toma de cada uno lo que sirva (Elo, Dixon-Coles, simulación de bracket, export de
visuales), pero respeta esta arquitectura y las reglas de reproducibilidad.
