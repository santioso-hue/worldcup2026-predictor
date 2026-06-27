"""Validación y reconciliación de resultados (funciones puras, sin I/O).

Dos responsabilidades:

1. **Validar** el estado y el marcador de cada partido (``scheduled``/``in_play``/
   ``finished``) y detectar datos sospechosos.
2. **Reconciliar** un snapshot nuevo contra el anterior: ante datos sospechosos o un
   intento de modificar un resultado YA bloqueado, **conservar el último snapshot
   válido** en vez de degradar la predicción.

La canonicalización de nombres entre proveedores vive en ``data/team_names.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .live_results import MatchStatus, NormalizedMatch, is_lockable


def validate_match(match: NormalizedMatch) -> list[str]:
    """Devuelve la lista de problemas detectados en un partido (vacía si está sano).

    Reglas:

    - Ningún marcador puede ser negativo.
    - Un ``FINISHED`` debe traer marcador de 90' (``ft_home``/``ft_away``).
    - Un ``SCHEDULED`` no debería traer marcador (dato contradictorio).
    - La prórroga debe estar completa (ambos lados o ninguno).
    - No puede haber penales sin prórroga (los penales van tras la prórroga).
    """
    issues: list[str] = []
    scores = (
        match.ht_home,
        match.ht_away,
        match.ft_home,
        match.ft_away,
        match.et_home,
        match.et_away,
        match.pen_home,
        match.pen_away,
    )
    if any(s is not None and s < 0 for s in scores):
        issues.append("marcador negativo")
    if match.status is MatchStatus.FINISHED and (
        match.ft_home is None or match.ft_away is None
    ):
        issues.append("finished sin marcador de 90'")
    if match.status is MatchStatus.SCHEDULED and (
        match.ft_home is not None or match.ft_away is not None
    ):
        issues.append("scheduled con marcador")
    if (match.et_home is None) != (match.et_away is None):
        issues.append("prórroga incompleta (un solo lado)")
    if match.pen_home is not None and match.et_home is None:
        issues.append("penales sin prórroga")
    # Una eliminatoria FINALIZADA no puede quedar en empate sin resolución: si llega así
    # es dato sospechoso (bloquearla dejaría una llave sin ganador en la simulación).
    is_group = match.stage.startswith("Group")
    if (
        match.status is MatchStatus.FINISHED
        and not is_group
        and match.ft_home is not None
        and match.ft_away is not None
        and match.ft_home == match.ft_away
    ):
        et_breaks = (
            match.et_home is not None
            and match.et_away is not None
            and match.et_home != match.et_away
        )
        pen_breaks = (
            match.pen_home is not None
            and match.pen_away is not None
            and match.pen_home != match.pen_away
        )
        if not (et_breaks or pen_breaks):
            issues.append("eliminatoria finalizada en empate sin resolución")
    return issues


def is_suspicious(match: NormalizedMatch) -> bool:
    """``True`` si el partido tiene algún problema de validación."""
    return bool(validate_match(match))


@dataclass(frozen=True)
class ReconcileResult:
    """Resultado de :func:`reconcile`: partidos elegidos + anomalías registradas."""

    matches: list[NormalizedMatch]
    anomalies: list[tuple[str, str]]  # (match_id, motivo)


def _choose(
    previous: NormalizedMatch | None, incoming: NormalizedMatch
) -> tuple[NormalizedMatch | None, str | None]:
    """Elige entre el partido previo y el entrante; devuelve (elegido, anomalía)."""
    if is_suspicious(incoming):
        reason = "; ".join(validate_match(incoming))
        if previous is not None:
            return previous, f"entrante sospechoso ({reason}); se conserva previo"
        return None, f"entrante sospechoso sin previo ({reason}); descartado"

    if previous is not None and is_lockable(previous):
        # Resultado bloqueado: inmutable, incluida su resolución (prórroga/penales).
        # El entrante solo se acepta si conserva el marcador de 90' Y no CAMBIA ninguna
        # fase ya conocida del previo (puede rellenar fases que antes eran None, p. ej.
        # añadir el detalle de penales sobre un mismo 1-1).
        same_score = (
            incoming.status is MatchStatus.FINISHED
            and incoming.ft_home == previous.ft_home
            and incoming.ft_away == previous.ft_away
        )
        changes_resolution = (
            (previous.et_home is not None and incoming.et_home != previous.et_home)
            or (previous.et_away is not None and incoming.et_away != previous.et_away)
            or (
                previous.pen_home is not None and incoming.pen_home != previous.pen_home
            )
            or (
                previous.pen_away is not None and incoming.pen_away != previous.pen_away
            )
        )
        if same_score and not changes_resolution:
            return incoming, None
        return (
            previous,
            "intento de modificar un resultado ya bloqueado; se conserva previo",
        )

    return incoming, None


def reconcile(
    previous: list[NormalizedMatch], incoming: list[NormalizedMatch]
) -> ReconcileResult:
    """Funde un snapshot entrante con el anterior conservando lo válido.

    - Un partido sospechoso en ``incoming`` se descarta a favor del previo (o se omite
      si no había previo).
    - Un resultado ya bloqueado (finished con marcador) no puede mutar: si el entrante
      lo contradice, se conserva el previo y se registra la anomalía.
    - Un partido que estaba en ``previous`` y no aparece en ``incoming`` se conserva
      (un feed parcial no debe borrar lo ya conocido).

    Parameters
    ----------
    previous:
        Último snapshot válido (puede estar vacío en el primer arranque).
    incoming:
        Snapshot recién descargado.

    Returns
    -------
    ReconcileResult
        ``matches`` reconciliados (orden: los de ``incoming`` y luego los exclusivos de
        ``previous``) y la lista de ``anomalies``.
    """
    prev_by = {m.match_id: m for m in previous}
    anomalies: list[tuple[str, str]] = []
    result: list[NormalizedMatch] = []
    seen: set[str] = set()

    for inc in incoming:
        seen.add(inc.match_id)
        chosen, anomaly = _choose(prev_by.get(inc.match_id), inc)
        if anomaly is not None:
            anomalies.append((inc.match_id, anomaly))
        if chosen is not None:
            result.append(chosen)

    for prev in previous:
        if prev.match_id not in seen:
            result.append(prev)

    return ReconcileResult(matches=result, anomalies=anomalies)
