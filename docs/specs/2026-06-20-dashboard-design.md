# Fase 8 — Dashboard interactivo (design)

> Fecha: 2026-06-20. Estado: aprobado. Implementación: TDD.

## Objetivo

Dashboard Streamlit que LEE los artefactos persistidos del pipeline
(`probabilities_<ts>.json` + `latest.json`) y los muestra de forma interactiva, con el sello
de "última actualización". Un botón "Re-simular" dispara el CLI del pipeline. No
re-implementa lógica: reusa los `prepare_*` ya testeados de `viz`.

## Decisiones (aprobadas)

- **Fuente de datos: AMBOS** — lee los artefactos por defecto + botón "Re-simular"
  (subprocess al CLI en modo live; reusa la única orquestación, sin duplicar fetch/sim).
- **Vistas:** ranking de campeón (P actual), tablas de P(avance) por grupo, detalle por
  selección (ronda a ronda), y el sello de última actualización. Sin predictor de partido
  (ya existe el CLI `predict_match`). Los deltas ↑/↓ viven en la **figura estática**
  (`render_outputs`): el dashboard es un lector puro y no tiene un puntero fiable a la
  corrida previa, así que no los muestra (evita una columna siempre vacía).
- **Extensión del artefacto:** el JSON incluye `groups` para que el dashboard dibuje las
  tablas de grupo siendo un lector puro (sin datos live).

## Componentes

### `pipeline.py` (extensión)
- `PipelineResult` += `groups: dict[str, list[str]]` (= `state.groups`).
- `write_probabilities` += `groups` opcional → payload `{timestamp, groups, probabilities}`.
  `load_latest_probabilities` se mantiene (probabilidades, para los deltas del pipeline).
- `RunArtifact(timestamp, groups, probabilities)` + `load_latest_run(outdir) ->
  RunArtifact | None` (lee puntero → archivo; `None` si no hay corrida).

### `viz/charts.py`
- `prepare_team_detail(probabilities, team) -> list[TeamRound(label, prob)]` en orden
  canónico de ronda (avance → octavos → … → campeón); falla si el equipo no está.

### `app/dashboard.py` (glue de Streamlit, sin tests unitarios)
- Header (título + sello + botón "Re-simular" vía subprocess al CLI), ranking
  (`prepare_champion_ranking`), tablas (`prepare_group_table`), detalle (`selectbox` +
  `prepare_team_detail`). Sin corridas → mensaje amable. `streamlit` se importa de forma
  perezosa dentro de `main`; `if __name__ == "__main__": main()` (Streamlit fija
  `__name__ == "__main__"`), así el módulo es importable sin ejecutar `st.*`.

## Manejo de errores

Sin puntero (aún no hay corrida) → `st.info` "corré el pipeline primero", no un crash.
Re-simular falla (returncode != 0) → `st.error` con el stderr.

## Testing (TDD)

- `load_latest_run`: round-trip (incluye `groups`); `None` si falta el puntero o el archivo.
- `prepare_team_detail`: orden canónico + equipo faltante lanza `ValueError`.
- `write_probabilities`/`load_latest_run` consistentes con el campo `groups`.
- `app/dashboard.py`: smoke-import (importlib ejecuta el cuerpo — imports + defs — sin
  llamar a `st.*`), para detectar regresiones de cableado en CI.

## Notas

`streamlit`/`pandas` son dependencias declaradas (en el venv). La lógica testeable vive en
`worldcup.*`; `app/dashboard.py` es glue fino (como los `scripts/`). Verificación con
`.venv/bin/*`.
