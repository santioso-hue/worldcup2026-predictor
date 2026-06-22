# Fase 7 — Orquestación del pipeline (design)

> Fecha: 2026-06-20. Estado: aprobado. Implementación: TDD (en una fase posterior; este
> documento solo fija el diseño).

## Objetivo

Encadenar los componentes de las Fases 1–6 en un pipeline end-to-end **reproducible**:
schedule + resultados live → reconcile → fit_elo → build_state → Monte Carlo → figuras +
artefacto de probabilidades. Cierra el hilo pendiente: `reconcile` alimenta `build_state`
(antes no estaba cableado).

## Decisiones (aprobadas)

- **Alcance: TODO el §8** — `run_pipeline` (modo `live` + `--snapshot` reproducible +
  `--mode pre_tournament` + `--watch`) y `predict_match`.
- **Arquitectura:** núcleo PURO y testeado en `src/worldcup/pipeline.py` + CLIs finos en
  `scripts/`. Toda la red/fs vive fuera del núcleo.
- **Salida:** `probabilities_<ts>.json` + `latest.json` (puntero, habilita deltas ↑/↓ vs la
  corrida previa) + figuras PNG/MP4.

## Componentes

### `src/worldcup/pipeline.py` (puro, testeado)
- `PipelineResult` (frozen): `probabilities` (`{team: {champion, advance, …}}`), `ratings`,
  `anomalies`.
- `run_pipeline(incoming, previous, history, config, annex_c, *, runs, seed,
  reference_date=None) -> tuple[PipelineResult, list[NormalizedMatch]]`: `reconcile` →
  `fit_elo` → `build_state` → `run_from_state`. Devuelve el resultado y los matches
  reconciliados (para snapshotear). Sin I/O.
- `predict_match(home, away, history, config, *, host=None) -> OutcomeProba`: `fit_elo` +
  `outcome_proba` (1X2 de un partido).
- I/O fino y testeable (tmp_path, sin red): `write_probabilities(probs, outdir, ts) -> Path`
  (`probabilities_<ts>.json` + `latest.json`), `load_latest_probabilities(outdir) ->
  dict | None` (para deltas), `render_outputs(result, previous, outdir, *, spec) ->
  list[Path]` (figuras con deltas vs previo).

### `scripts/run_pipeline.py` (typer, fino)
- Flags: `--config --snapshot <ts> --runs --seed --mode {live,pre_tournament} --watch
  --interval`.
- `live`: fetch schedule + resultados → `incoming`; `load_latest_snapshot` → `previous`;
  cargar `history`.
- `--snapshot <ts>`: carga ese snapshot como `incoming` (reproducible).
- `pre_tournament`: `incoming` = schedule sin resultados, `previous` vacío, y el Elo se
  ajusta solo con partidos ANTES del primer fixture (`history_cutoff`) — baseline limpio,
  sin contaminarse con resultados del torneo en curso que trae el histórico.
- Corre `run_pipeline` → `save_snapshot(reconciled)` + `write_probabilities` +
  `render_outputs`.
- `--watch`: bucle fino sobre un `run_once` testeado, usando `triggers.RefreshTrigger`;
  `sleep` inyectable para poder testearlo.

### `scripts/predict_match.py` (typer, fino)
- `home away --host` → cargar history → `predict_match` → imprimir el 1X2.

## Flujo, modos y errores

- Diagrama de flujo: ver brainstorm (inputs → I/O fino → núcleo puro → outputs).
- **Determinismo:** mismo snapshot + misma semilla → salida idéntica. En `live`, las figuras
  cambian cuando entra un resultado nuevo (esperado); para grabar, `--snapshot <ts>`.
- **Fail-loud:** las `anomalies` de `reconcile` se devuelven y se loggean, nunca se tragan;
  falta de API key (live) / de history / `--snapshot` inexistente → error claro.

## Testing (TDD)

- `run_pipeline`: fixtures de 12 grupos + history mínima → champion suma 1; determinista por
  semilla; **un KO finalizado en empate sospechoso en `incoming` lo descarta `reconcile`, de
  modo que `build_state` NO lo bloquea** (regresión del hilo pendiente); las anomalías se
  reportan.
- `predict_match`: 1X2 válido que suma 1.
- `write_probabilities`/`load_latest_probabilities`: round-trip + cálculo de deltas vs previo.
- `render_outputs`: escribe los PNG esperados (tmp_path).
- `watch`: `sleep` inyectado + tope de iteraciones → corre N veces y para cuando no hay
  partidos pendientes.

## Notas

Núcleo puro (sin red/fs) para testear con fixtures; el I/O vive en `scripts/` + helpers finos.
`pipeline` puede importar `data`/`features`/`models`/`simulation`/`evaluation`/`viz`/`config`;
no `app`.
