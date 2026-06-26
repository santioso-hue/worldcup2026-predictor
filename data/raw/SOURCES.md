# SOURCES.md — Fuentes de datos (ubicar en `data/raw/SOURCES.md`)

> Investigado: 16 jun 2026. El esquema de cada endpoint (cuotas y campos) puede cambiar;
> conviene confirmarlo contra la documentación oficial.

---

## Decisión

| Rol | Fuente | Por qué |
|-----|--------|---------|
| **Resultados live** | **football-data.org** (v4), tier free | "Free forever", cubre el WC2026 **sin límite de temporada** (verificado 22-jun-2026: los 104 partidos vía un solo GET `/competitions/WC/matches`, in-play + finished). Auth `X-Auth-Token`, 10 req/min. *Notas v4:* los KO sin resolver llegan con equipos `null` (se omiten; el bracket se reconstruye); los penales van en `score.winner` (no se desglosa la tanda). |
| **Calendario / backbone** | **openfootball/worldcup.json** (GitHub) | Versionado en git, sin API key, fixtures + 12 grupos + sedes. Fuente confiable del schedule y fallback offline. |
| **Histórico (Elo + backtest)** | **martj42/international_results** (GitHub/Kaggle) | ~47k+ partidos internacionales desde 1872. Base del Elo y del backtest. |
| **Conveniencia WC-específica (fallback)** | **rezarahiminia/worldcup2026** (`worldcup26.ir`) | REST gratis, sin key, específico del WC2026 (104 partidos, standings, bracket). *Riesgo:* proyecto de un solo mantenedor; uptime/longevidad sin verificar → solo fallback, o auto-hospedar (Node/Express/Mongo). |

**Regla de diseño:** todas las fuentes live se acceden detrás de la interfaz
`data/live_results.LiveResultsProvider`, para poder cambiar de proveedor
(football-data.org ↔ openfootball ↔ worldcup26) sin tocar el modelo.

---

## Presupuesto de requests (por qué el tier free alcanza)

Nuestro modelo recondiciona con **resultados finalizados**; el in-play es solo para el
dashboard. No necesitamos granularidad de segundos.

- football-data.org free = **10 req/min** (sin límite diario práctico para 1 competición).
- Cada corrida del pipeline hace **1 request** (`GET /competitions/WC/matches` trae los
  104 partidos con su estado). El cron de refresco corre cada **30 min** → ~48 req/día,
  muy holgado bajo el límite por minuto.
- Eliminatorias (1–4 partidos, ventanas cortas) → mismo costo (1 request/corrida). ✓

**Nunca** hagas polling continuo ni fuera de ventana de partido. Cachea todo lo que
cambia lento y nunca gastes una request en un refresh de página.

---

## Alternativas evaluadas (descartadas como primario)

- **Highlightly** — free 100 req/día, cubre los 104 partidos del WC2026, live cada 15 s.
  Cobertura equivalente; menor ecosistema/wrappers. **Suplente válido del primario.**
- **TheSportsDB** — gratis, crowd-sourced, amplio pero cobertura/consistencia variable.
- **Sportmonks / TheStatsAPI** — de pago (trial), fuera del requisito "gratis".
- **BSD (bzzoiro)** — "free, sin rate limit", pero foco en ligas de clubes europeas;
  cobertura de selecciones/WC sin verificar → no apto como primario.

---

## Puntos a confirmar en la doc oficial

1. **football-data.org:** si los resultados **FINALIZADOS** llegan en el tier free sin el
   add-on de livescore, y con qué latencia tras el pitazo final.
2. **openfootball:** formato del JSON de resultados (cuándo y cómo se llenan los marcadores).
3. **worldcup26.ir:** estabilidad/uptime; si conviene clonar y auto-hospedar el repo.
