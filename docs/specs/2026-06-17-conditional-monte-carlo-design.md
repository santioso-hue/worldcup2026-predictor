# Fase 4 — Conditional Monte Carlo (design)

> Fecha: 2026-06-17. Estado: aprobado. Implementación: TDD.

## Objetivo

Simular `N` veces SOLO la parte no jugada del torneo, condicionando sobre los resultados
reales (bloquea finalizados, equipos eliminados emergen a 0%, bracket real con mejores
terceros vía Annex C). Agrega frecuencias → P(avanzar de grupo / R16 / QF / SF / final /
campeón) por selección. Determinista dado snapshot + semilla.

## Datos verificados (de FWC2026 regulations PDF)

- **Annex C** (Art. 12.6 / Annexe C): 495 combinaciones → asignación de los 8 mejores
  terceros a los ganadores de grupo. Extraído y verificado (495 filas, cubre C(12,8),
  0 violaciones de elegibilidad, idéntico en dos copias del PDF). Se guarda como
  `data/raw/annex_c_2026.json` generado por `scripts/extract_annex_c.py` (provenance).
  Columnas (orden): `1A 1B 1D 1E 1G 1I 1K 1L`.
- **Bracket de eliminatoria** (Art. 12.6–12.11), VERIFICADO:
  - R32 (M73–M88): mezcla de 1X vs 3rd(set), 1X vs 2Y, 2X vs 2Y. Los 8 slots de tercero:
    M74 1E·ABCDF, M77 1I·CDFGH, M79 1A·CEFHI, M80 1L·EHIJK, M81 1D·BEFIJ, M82 1G·AEHIJ,
    M85 1B·EFGIJ, M87 1K·DEIJL. (Resto: M73 2A-2B, M75 1F-2C, M76 1C-2F, M78 2E-2I,
    M83 2K-2L, M84 1H-2J, M86 1J-2H, M88 2D-2G.)
  - R16 (M89–M96): 89=W74-W77, 90=W73-W75, 91=W76-W78, 92=W79-W80, 93=W83-W84,
    94=W81-W82, 95=W86-W88, 96=W85-W87.
  - QF: 97=W89-W90, 98=W93-W94, 99=W91-W92, 100=W95-W96.
  - SF: 101=W97-W98, 102=W99-W100. Final: 104=W101-W102. (3er puesto M103: perdedores SF.)

## Tie-breakers (Art. 13 — VERIFICADO, orden correcto)

**Ranking de grupo** (si empate en puntos):
1. Puntos (overall).
2. **Step 1 — head-to-head** entre los empatados: pts H2H → GD H2H → goles H2H.
3. **Step 2** — si Step 1 separa parcialmente, re-aplica H2H al subconjunto aún empatado
   (matches between remaining teams only, recursivo); luego GD overall → goles overall →
   conduct score (tarjetas). Step 2 NO reinicia tras pasar a GD/goles/conduct.
4. **Step 3** — FIFA/Coca-Cola Men's World Ranking.

**Ranking de terceros** (sin H2H, distintos grupos): puntos → GD → goles → conduct → FIFA.

**Qué simulamos:** puntos, H2H (recursivo), GD overall, goles overall. **Qué NO** (y cómo):
- *conduct score* (tarjetas): no modelamos tarjetas → criterio omitido (limitación documentada).
- *FIFA ranking* (paso final): no se ingiere; el **Elo** recondicionado (un ranking
  estilo-FIFA en vivo) es el proxy determinista de ese paso. Sorteo sembrado solo como
  último recurso teórico (Elos empatados ~nunca).

Cadena implementada: `puntos → H2H(pts/GD/GF, recursivo) → GD → goles → [conduct omitido]
→ Elo`. Igual para terceros, sin H2H.

## Módulos (orden de construcción)

1. `simulation/bracket.py` (nuevo, para mantener `tournament.py` enfocado): plantilla del
   bracket (R32–Final), loader de Annex C, `assign_best_thirds(qualifying_groups)`.
2. `simulation/group_stage.py`: `standings` (Art. 13, H2H recursivo) + `rank_thirds`.
3. `simulation/match.py`: `simulate_match` (reglamentario vía DixonColes; KO empate → ET
   Poisson ~0.8 total por reparto de λ → penales Elo-weighted).
4. `simulation/state.py`: `TournamentState` desde schedule + snapshot + ratings (locked /
   pending; eliminación emergente).
5. `simulation/tournament.py`: Monte Carlo condicional; agrega P por ronda sobre `N` runs
   vía `worldcup.rng.spawn_rngs`.

## Test invariants

- Annex C: loader da 495 entradas; combinación conocida → asignación correcta (ya
  verificado en extracción: 0 violaciones, cubre C(12,8)).
- Tiebreakers: H2H ANTES de GD; recursión Step-1 en casos de 3 empatados; Step-2 no
  reinicia; Elo como desempate final.
- Conditioning: partidos locked nunca se re-muestrean; equipo fuera por resultado locked → 0% campeón.
- Bracket: exactamente un campeón por run; P(campeón) suma 1 sobre equipos.
- Reproducibilidad: mismo snapshot + semilla → agregados idénticos.

## Reglas

Sin números mágicos (config). `simulation` puede importar `models`/`features`/`data`/
`config`, no `viz`/`app`. Aleatoriedad solo vía `worldcup.rng`.
