# Fase 6a — Visualización: fundamentos + gráficos principales (design)

> Fecha: 2026-06-19. Estado: aprobado. Implementación: TDD.

## Objetivo

Capa `viz/` que produzca las figuras sociales clave del proyecto a partir de la salida del
Monte Carlo (Fase 4), el modelo de partido (Fase 3) y la calibración (Fase 5). Marca única,
limpia, alto contraste, legible en móvil.

## Alcance (6a, aprobado) y diferido (6b)

- **6a (este spec):** `theme.py` + `export.py` + `charts.py` con: ranking de campeón
  (top-N por P(título), con delta ↑/↓ vs snapshot previo), barra 1X2 de un partido, heatmap
  de marcadores, y **diagrama de fiabilidad** (visualiza la calibración de Fase 5).
- **6b (diferido):** `bracket.py`, tabla de grupo con P(avance), animación PNG→MP4.

## Decisiones (aprobadas)

- **Dirección visual:** acento único (azul), alto contraste, números grandes, sello de
  "última actualización" en figuras live (según el mock aprobado). Sin colores por
  selección en 6a.
- **Arquitectura testeable:** cada figura se parte en `prepare_*` (puro: datos → estructura
  lista para plotear, testeado por valor) + `render_*` (matplotlib, fino, smoke-test). Nada
  de comparación de píxeles.
- **Marca única:** todo color/tipografía/tamaño vive en `theme.py`; ningún otro módulo
  hardcodea marca.
- **Export:** `export.py` guarda PNG determinista en `outputs/figures/`; tamaños vertical
  1080×1920 y horizontal 1920×1080, 150 dpi (de `theme.py`). Backend matplotlib **Agg**
  (headless, reproducible).

## Componentes

### `viz/theme.py`
- `ExportSpec(width_px, height_px, dpi=150)`; `PORTRAIT=1080×1920`, `LANDSCAPE=1920×1080`.
- `Theme` (frozen): paleta (accent, background, text_primary/muted, grid, up/down, draw/away,
  heat), tipografía (`font_family="DejaVu Sans"` — siempre disponible, determinista) y
  tamaños. Instancia `THEME`.

### `viz/charts.py`
Pares prepare/render por figura:
- `prepare_champion_ranking(probs, previous=None, *, top_n=10, min_delta=0.005)
  -> list[RankingRow(rank, team, prob, delta)]` — orden desc, desempate por nombre, delta
  `up/down/flat` (umbral `min_delta` para no marcar ruido). `render_champion_ranking(...)`.
- `prepare_match_bar(home, away, p_home, p_draw, p_away) -> list[BarSegment]` — falla si no
  suman 1. `render_match_bar(...)`.
- `prepare_score_heatmap(score_matrix, *, max_goals=5) -> HeatmapData(grid, mode)` — valida
  matriz 2D no negativa que suma 1; recorta a la región mostrada y halla el marcador modal.
  `render_score_heatmap(...)`.
- `prepare_reliability(bins) -> ReliabilityData(pred, observed, counts)` — de
  `calibration.reliability_bins`; falla si vacío. `render_reliability(...)` dibuja la
  diagonal de calibración perfecta.

### `viz/export.py`
- `save_figure(fig, name, spec, *, outdir=outputs/figures) -> Path` — fija tamaño en píxeles
  desde `spec` (width/height/dpi), guarda PNG con nombre determinista, devuelve la ruta.

## Testing (TDD)

- `theme`: PORTRAIT/LANDSCAPE exactos; todos los colores son hex `#rrggbb` válidos.
- `prepare_champion_ranking`: orden desc + tope top_n + desempate; delta up/down/flat con
  umbral; equipo ausente en `previous` → flat.
- `prepare_match_bar`: exige suma 1 (ValueError si no); segmentos y roles correctos.
- `prepare_score_heatmap`: valida 2D/no-negativo/suma 1; recorte a max_goals; marcador modal.
- `prepare_reliability`: passthrough + falla en vacío.
- `render_*` (con matplotlib): devuelven `Figure` con el nº esperado de artistas (p.ej. nº de
  barras == nº de filas) y título puesto; `save_figure` escribe un PNG no vacío del tamaño
  correcto. Backend Agg.

## Notas

`viz` puede importar `config`/`models`/`simulation`/`evaluation`; NO `app`. matplotlib es
dependencia declarada. Las funciones `prepare_*` son puras y no tocan matplotlib (testeables
sin la dependencia); `render_*`/`export` sí.
