# worldcup2026-predictor

Predictor del Mundial 2026 que se actualiza en vivo (48 selecciones, 12 grupos). A medida
que se juegan los partidos, recalcula las probabilidades con los resultados reales y arma
los gráficos de las predicciones. El modelo combina **Elo dinámico → Dixon-Coles → Monte
Carlo condicional**.

## Para arrancar

```bash
make setup                        # crea el .venv e instala las dependencias
make run                          # corrida live completa (necesita FOOTBALL_DATA_TOKEN)
streamlit run app/dashboard.py    # tablero interactivo (lee la última corrida)
```

Si no hay clave de API, igual se puede simular el baseline previo al torneo, que baja el
calendario (openfootball) y el histórico (martj42):

```bash
python scripts/run_pipeline.py --mode pre_tournament --runs 50000
```

## Comandos

```bash
# Pronóstico de un partido suelto (1X2), con el histórico de martj42.
# Los nombres tienen que coincidir con martj42 (p. ej. "United States", no "USA").
python scripts/predict_match.py "Brazil" "France"
python scripts/predict_match.py "United States" "Mexico" --host "United States"

# Reproducir un estado exacto (los números no cambian)
python scripts/run_pipeline.py --snapshot 20260616t1830 --runs 50000

# Refrescar en bucle mientras hay partidos (sondea por ventanas; ver triggers.py)
python scripts/run_pipeline.py --watch --interval 600

# Calidad (todo corre dentro del .venv, vía make)
make test                         # pytest
make lint                         # ruff + black --check + mypy
make fmt                          # black + ruff --fix
```

Para algo más estable que `--watch`, el workflow
[`.github/workflows/refresh.yml`](.github/workflows/refresh.yml) opera el pipeline con
GitHub Actions en un horario fijo y sube las figuras y el JSON como artefactos.

## Resultados

- `data/processed/probabilities_<ts>.json` (y `latest.json`): por selección, la
  probabilidad de avanzar y de llegar a octavos, cuartos, semis, final y título, más los
  grupos. El puntero `latest.json` sirve para mostrar las flechas ↑/↓ contra el simulacro
  anterior.
- `outputs/figures/*.png`: ranking de campeón, barra 1X2, mapa de calor de marcadores,
  bracket, tabla de grupos y curva de fiabilidad (1080×1920 vertical o 1920×1080 horizontal).
- `outputs/videos/*.mp4`: la animación que va revelando al campeón.
- `data/raw/results_<ts>.parquet`: el snapshot inmutable, el registro de *qué sabía el
  modelo y cuándo*.

**Reproducibilidad:** con el mismo snapshot y la misma semilla, la salida es idéntica. En
modo `live` las figuras cambian cuando entra un resultado nuevo (es lo que se espera); para
fijar un estado reproducible, se usa `--snapshot <ts>`.

## Cómo funciona

El pipeline encadena cuatro piezas:

- **Datos en vivo:** `FootballDataProvider` baja los resultados; el calendario sale de
  openfootball y el histórico de martj42. Cada corrida guarda un snapshot con timestamp,
  valida y reconcilia.
- **Elo dinámico:** ajuste secuencial y determinista sobre el histórico, con multiplicador
  por margen de gol (eloratings.net) y peso por recencia.
- **Dixon-Coles:** convierte la diferencia de Elo en goles esperados, arma la matriz de
  Poisson y corrige los marcadores bajos con τ.
- **Monte Carlo condicional:** Anexo C, desempates del Art. 13 (enfrentamiento directo
  recursivo), prórroga y penales; simula solo lo que falta por jugar y da P(ronda/título).

Encima van el backtest (walk-forward con calibración de Platt), los gráficos (PNG/MP4) y
el tablero de Streamlit, todo sobre los artefactos que deja cada simulacro.

## Arquitectura

Las capas están separadas y las dependencias van en un solo sentido:
`data → features → models → simulation → evaluation → viz`. El I/O vive en `data`, `scripts`
y `app`; Toda la
aleatoriedad pasa por `worldcup.rng.get_rng()`, así que con la misma semilla siempre sale lo
mismo.
