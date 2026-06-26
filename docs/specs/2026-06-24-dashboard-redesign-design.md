# Rediseño del dashboard + simulador de partido (Fase 8.1)

- **Fecha:** 2026-06-24
- **Estado:** diseño aprobado, pendiente de plan de implementación
- **Tipo:** herramienta personal (no sitio público) — backbone para correr la simulación

## Objetivo

Dejar el dashboard de Streamlit con un look limpio y minimalista (estética Mundial,
un solo acento), y agregar un **simulador de partido** que solo opere sobre los
fixtures reales del WC2026 (no enfrentamientos arbitrarios).

## No-objetivos (YAGNI)

- Nada de sitio público desplegado, stack distinto a Streamlit, ni CSS pesado.
- Sin colores por selección (un solo acento, `#185FA5`, consistente con los gráficos
  de los reels).
- Sin cambiar las salidas del pipeline: el dashboard solo **lee** artefactos existentes.
- Sin simular enfrentamientos arbitrarios: solo partidos del calendario WC2026.

## Enfoque

Aprobado el **enfoque A**: tema nativo de Streamlit + reusar los gráficos que `viz/`
ya renderiza + un panel de partido. Sin reescribir el modelo ni el pipeline.

## Diseño

### 1. Tema y shell

- Nuevo `.streamlit/config.toml`: tema claro, `primaryColor = "#185FA5"`, fondo blanco,
  fuente limpia. Theming nativo, sin inyección de CSS.
- `app/dashboard.py`: cabecera con marca (⚽ + "Predictor Mundial 2026" + "actualizado ·
  `<ts>`" + botón Re-simular), y luego todo bajo `st.tabs(["Torneo", "Partido"])`.

### 2. Pestaña "Torneo"

Reemplazar las tablas `st.dataframe` por las figuras que `viz/` ya produce, vía
`st.pyplot(fig)`:

- Ranking de campeón (`render_champion_ranking`).
- Avance por grupo (`render_group_table`).
- Detalle por selección (la escalera de rondas; `prepare_team_detail` + el mismo
  renderer de barras).

Mismos datos (`load_latest_run`), mejor presentación, reusando código ya probado.

### 3. Pestaña "Partido" (simulador)

- **Fuente de fixtures:** `load_latest_snapshot` (el último snapshot live). Lista los
  partidos del WC2026 con ambos equipos definidos (fase de grupos ya; eliminatoria a
  medida que se resuelven los cruces). Los slots sin resolver (`2A` vs `1E`) no aparecen.
  Si no hay snapshot todavía: mensaje "corré una simulación live primero".
- **Selector:** `st.selectbox` agrupado/legible por etapa, p.ej. "Grupo K · Colombia vs
  DR Congo · 24 jun", ordenado por fecha de kickoff.
- **Ventaja de sede:** si uno de los dos equipos es anfitrión (`config.simulation.hosts`),
  un toggle "aplicar ventaja de sede a `<equipo>`" (default ON) que pasa ese equipo como
  `host` a la predicción.
- **Salida:** barra 1X2 (`render_match_bar`) + heatmap de marcadores
  (`render_score_heatmap`). Si el partido ya está FINISHED, mostrar además el marcador
  real (una línea "ya jugado: X-Y"), junto a lo que el modelo daba.

### 4. Lógica pura nueva (testeable)

Un helper en `worldcup.pipeline` que, dado un fixture, devuelve **el 1X2 y la matriz de
marcadores** (la única lógica nueva; el dashboard solo lo llama y renderiza):

```python
def predict_fixture(
    home: str, away: str, history: list[HistoricalMatch], config: Config,
    *, host: str | None = None, reference_date: date | None = None,
) -> tuple[MatchOutcome, np.ndarray]:
    """1X2 + matriz de marcadores Dixon-Coles para un partido (puro)."""
```

- Ajusta Elo (`fit_elo`), resuelve ratings (default `initial_rating` si falta),
  aplica `host_advantage` igual que `predict_match` (host == home/away).
- Construye `DixonColesModel`, devuelve `(outcome_proba, score_matrix)`.
- `predict_match` se refactoriza para delegar en `predict_fixture` y devolver solo el
  `MatchOutcome` (DRY; sin duplicar la lógica de ratings/ventaja).

### 5. Touchpoints de código

- **Nuevo:** `.streamlit/config.toml`.
- **Nuevo:** `pipeline.predict_fixture` (+ refactor de `predict_match` para delegar).
- **Reescrito:** `app/dashboard.py` (sigue siendo glue fino: tabs, selectbox, `st.pyplot`).
- **Caché:** `@st.cache_data` para cargar el histórico y ajustar Elo una sola vez (que
  elegir fixtures sea ágil).

### 6. Pruebas

- `predict_fixture`: unit tests (1X2 suma 1; matriz suma ~1; la ventaja de sede mueve la
  probabilidad del anfitrión; equipo desconocido cae a `initial_rating`). `predict_match`
  sigue verde tras el refactor.
- `app/dashboard.py`: se mantiene la prueba de smoke-import (sigue la convención: el glue
  de Streamlit no se testea a fondo).

## Riesgos / notas

- El panel de partido depende de que exista un snapshot live; en modo solo-baseline no
  habrá fixtures con estado. Aceptable para una herramienta personal que se corre live.
- `render_*` por defecto usan specs PORTRAIT/LANDSCAPE pensados para reels; en el dashboard
  se renderizan a un tamaño cómodo para web (definir el spec al llamarlos; detalle del plan).
