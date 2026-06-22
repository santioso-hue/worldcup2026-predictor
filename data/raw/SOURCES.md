# SOURCES.md — Fuentes de datos (ubicar en `data/raw/SOURCES.md`)

> Investigado: 16 jun 2026. **Verifica el esquema real de cada endpoint en su
> documentación oficial antes de escribir el parser** (las cuotas y campos cambian).
> Nunca inventes columnas. Si un campo no existe, decláralo.

---

## Decisión

| Rol | Fuente | Por qué |
|-----|--------|---------|
| **Resultados live (primario)** | **football-data.org** (v4), tier free | "Free forever", cubre el WC2026 **sin límite de temporada** (verificado 22-jun-2026: los 104 partidos vía un solo GET `/competitions/WC/matches`, in-play + finished). Auth `X-Auth-Token`, 10 req/min. *Notas v4:* los KO sin resolver llegan con equipos `null` (se omiten; el bracket se reconstruye); los penales van en `score.winner` (no se desglosa la tanda). |
| **Resultados live (alternativo)** | **API-Football** (api-sports.io), tier free | `fixtures?live=all` devuelve todos los partidos concurrentes en una llamada. **Caveat (verificado 22-jun-2026):** el tier free NO da acceso a la temporada 2026 ("Free plans do not have access to this season, try from 2022 to 2024") → inservible para el WC2026 sin plan de pago. |
| **Calendario / backbone** | **openfootball/worldcup.json** (GitHub) | Versionado en git, sin API key, fixtures + 12 grupos + sedes. Fuente confiable del schedule y fallback offline. |
| **Histórico (Elo + backtest)** | **martj42/international_results** (GitHub/Kaggle) | ~47k+ partidos internacionales desde 1872. Base del Elo y del backtest. |
| **Conveniencia WC-específica (fallback)** | **rezarahiminia/worldcup2026** (`worldcup26.ir`) | REST gratis, sin key, específico del WC2026 (104 partidos, standings, bracket). *Riesgo:* proyecto de un solo mantenedor; uptime/longevidad sin verificar → solo fallback, o auto-hospedar (Node/Express/Mongo). |

**Regla de diseño:** todas las fuentes live se acceden detrás de la interfaz
`data/live_results.LiveResultsProvider`, para poder cambiar de proveedor
(API-Football ↔ football-data.org ↔ openfootball ↔ worldcup26) sin tocar el modelo.

---

## Presupuesto de requests (por qué el tier free alcanza)

Nuestro modelo recondiciona con **resultados finalizados**; el in-play es solo para el
dashboard. No necesitamos granularidad de 15 s.

- API-Football free = **100 req/día**. La cobertura de competición puede variar;
  los datos de live se actualizan cada ~15 s del lado del servidor.
- Estrategia de bajo consumo:
  1. **1×/día:** traer fixtures del día (1 llamada) + standings (1 llamada). Cachear.
  2. **Solo durante ventanas con partidos en juego:** `fixtures?live=all`
     (1 llamada devuelve TODOS los partidos concurrentes) cada **10–15 min**.
- Caso peor (fase de grupos, ~12 h de fútbol repartido por husos US/MX/CA):
  - cada 10 min → 12 h × 6 = **72** + ~5 mantenimiento ≈ **77/día** (cabe, ajustado).
  - cada 15 min → ~**48/día** (cómodo).
- Eliminatorias (1–4 partidos, ventanas cortas) → muy por debajo de 100. ✓

**Nunca** hagas polling continuo ni fuera de ventana de partido. Cachea todo lo que
cambia lento (teams, standings) y nunca gastes una request en un refresh de página.

---

## Alternativas evaluadas (descartadas como primario)

- **Highlightly** — free 100 req/día, cubre los 104 partidos del WC2026, live cada 15 s.
  Equivalente a API-Football; menor ecosistema/wrappers. **Suplente válido del primario.**
- **TheSportsDB** — gratis, crowd-sourced, amplio pero cobertura/consistencia variable.
- **Sportmonks / TheStatsAPI** — de pago (trial), fuera del requisito "gratis".
- **BSD (bzzoiro)** — "free, sin rate limit", pero foco en ligas de clubes europeas;
  cobertura de selecciones/WC sin verificar → no apto como primario.

---

## A verificar en la doc oficial (antes de codear `live_results.py`)

1. **API-Football:** `league` id y `season` del WC2026; forma exacta del status
   (`NS/1H/HT/2H/FT/AET/PEN`), penales y prórroga; campos de marcador final.
2. **football-data.org:** si los resultados **FINALIZADOS** llegan en el tier free sin el
   add-on de livescore, y con qué latencia tras el pitazo final.
3. **openfootball:** formato del JSON de resultados (cuándo y cómo se llenan los marcadores).
4. **worldcup26.ir:** estabilidad/uptime; si conviene clonar y auto-hospedar el repo.
