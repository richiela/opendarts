"""Argus's decision: per-camera features in, one board call out.

Pure Python, no image work -- `opendarts.engines.argus.features` has
already reduced each camera to a handful of numbers. Three layers, each
only stepping in where the one below it is unsure:

1. VISUAL VOTE (`visual_decision`). Every camera gets a confidence from
   how well it saw the needle: entry visibility (taper + elongation of
   the shaft blob, penalised when the entry sits on an earlier dart of
   the visit or on a prior dart's line), steepness (its mm-per-pixel
   along the needle against the best camera's), clarity of the tip in
   the difference image, distance from the nearest wire, how decisively
   the board paint under the tip matched one bed, and agreement with
   the other cameras' board points. A camera whose needle entry is
   visible calls the bed by PAINT (the colour under the tip), otherwise
   by the calibration MAP at its tip. One clearly-best camera with
   doubtful peers is definitive; otherwise a confidence-weighted vote.

2. POSITION FUSION (`fuse`). Each camera's tip, projected onto the
   board, is precise ACROSS its needle and uncertain ALONG it (where the
   point really ends). So each camera contributes an elongated
   uncertainty ellipse (sig_u px along, sig_v px across, mapped to board
   mm by the local Jacobian); the fused point is the information-
   weighted mean. A camera inconsistent with the fused point
   (Mahalanobis^2 > chi2_max) is dropped and the rest refit. A camera
   whose needle entry is hidden (on an earlier dart, on the prior dart's
   line, or barrel-wide) has its along-needle sigma inflated.

3. HYBRID (`decide`). The beds the fused point's k-sigma ellipse
   touches are the candidates. The visual call stands if it is one of
   them; otherwise the best-weighted candidate in the per-camera tally;
   failing that, the bed under the fused point. No consistent pair of
   cameras -> the visual call alone.

Board map. The two INNER ring boundaries (treble and double) are moved
inward by `INNER_RING_SCORING_OFFSET_MM`, OD's measured scoring
convention (a needle on the inner wire belongs to the ring); outer
boundaries and the bull keep regulation radii. Per-camera map tokens,
wire margins and every label here use that map. A paint read that only
echoed the regulation map token follows the moved one.

Tokens are "DB", "OB", "OUT", "S<n>", "T<n>", "D<n>"; the engine maps
them to OD's (sector, ring) vocabulary.

Parameters (`PARAMS`) were chosen by cross-validated search on the
4,077-dart truth corpus and are frozen here; the decision must stay
call-for-call identical to that search's reference output.
"""
from __future__ import annotations

import math

from opendarts.engines.apollo.tip_detection import MIN_ELONGATION_RATIO
from opendarts.geometry.board import (
    BULL_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    INNER_RING_SCORING_OFFSET_MM,
    OUTER_BULL_RADIUS_MM,
    SECTOR_NUMBERS_CLOCKWISE,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
)

PARAMS: dict[str, float] = {
    # visual vote: per-term weights of a camera's confidence
    "w_entry": 1.0, "w_steep": 1.0, "w_clarity": 1.0,
    "w_margin": 1.0, "w_paint": 1.0, "w_agree": 1.0,
    # term scales
    "margin_sigma_px": 5.0,     # wire distance (in px) that counts as fully clear
    "taper_weight": 0.75,       # entry = 0.75 taper + 0.25 elongation
    "elong_scale": 3.5,
    "clarity_scale": 4.0,
    "paint_scale": 0.6,
    "agree_scale_mm": 8.0,
    "entry_visible_min": 0.15,  # below this the camera calls by map, not paint
    "contam_penalty": 0.3,      # entry on the prior dart's line
    "occl_entry_penalty": 0.05,  # entry on an earlier dart of the visit
    # definitive single camera
    "definitive_abs": 0.45,
    "definitive_margin": 0.05,
    "other_ambiguous_max": 0.75,
    # position fusion
    "sig_u": 2.0,               # px along the needle
    "sig_v": 2.5,               # px across it
    "chi2_max": 25.0,
    "infl": 10.0,               # along-needle inflation for a hidden entry
    "barrel_mm": 6.0,           # needle wider than this near the tip = barrel, entry hidden
    "k_sigma": 1.0,             # candidate ellipse
    # board map
    "inner_off": INNER_RING_SCORING_OFFSET_MM,
}


# --------------------------------------------------------------------------- #
# Board map with the inner-wire scoring offset
# --------------------------------------------------------------------------- #

def label(x: float, y: float, off: float) -> str:
    """Token of the bed at board point (x, y) mm. Kept in this exact
    arithmetic form (round-to-nearest sector index, `<=` boundaries) so
    calls reproduce the tuned reference bit for bit."""
    r = math.hypot(x, y)
    th = math.degrees(math.atan2(x, y)) % 360
    n = SECTOR_NUMBERS_CLOCKWISE[int(round(th / 18.0)) % 20]
    if r <= BULL_RADIUS_MM:
        return "DB"
    if r <= OUTER_BULL_RADIUS_MM:
        return "OB"
    if r <= TREBLE_INNER_RADIUS_MM - off:
        return f"S{n}"
    if r <= TREBLE_OUTER_RADIUS_MM:
        return f"T{n}"
    if r <= DOUBLE_INNER_RADIUS_MM - off:
        return f"S{n}"
    if r <= DOUBLE_OUTER_RADIUS_MM:
        return f"D{n}"
    return "OUT"


def margin_mm(x: float, y: float, off: float) -> float:
    """Distance (mm) from (x, y) to the nearest ring boundary or wire."""
    r = math.hypot(x, y)
    th = math.degrees(math.atan2(x, y)) % 360
    ring = min(abs(r - rb) for rb in (
        BULL_RADIUS_MM, OUTER_BULL_RADIUS_MM, TREBLE_INNER_RADIUS_MM - off,
        TREBLE_OUTER_RADIUS_MM, DOUBLE_INNER_RADIUS_MM - off, DOUBLE_OUTER_RADIUS_MM))
    k = int(round(th / 18.0)) % 20

    def ad(a, b):
        d = (a - b) % 360
        return min(d, 360 - d)

    wdeg = min(ad(th, (18 * k - 9) % 360), ad(th, (18 * k + 9) % 360))
    return min(ring, math.radians(wdeg) * max(r, 20.0))


def apply_board_offset(cams: dict, off: float) -> dict:
    """Copy of `cams` with each ok camera's map token and wire margin
    recomputed on the offset map. A paint token equal to the regulation
    map token was only echoing the map, so it follows the new token."""
    out = {}
    for c, x in cams.items():
        x = dict(x)
        if x.get("ok") and x.get("board_xy") is not None:
            bx, by = x["board_xy"]
            old = x["token"]
            new = label(bx, by, off)
            if x.get("paint_token") == old:
                x["paint_token"] = new
            x["token"] = new
            x["margin_mm"] = margin_mm(bx, by, off)
        out[c] = x
    return out


# --------------------------------------------------------------------------- #
# 1. Visual vote
# --------------------------------------------------------------------------- #

def _clip01(x):
    return max(0.0, min(1.0, x))


def camera_confidence_parts(feat: dict, params: dict) -> dict:
    """One ok camera's features -> its confidence terms."""
    width_ratio = feat.get("width_ratio")
    elong = feat.get("elongation_ratio")
    taper_q = _clip01(1.0 - width_ratio) if width_ratio is not None else 0.5
    elong_q = (_clip01((elong - MIN_ELONGATION_RATIO) / params["elong_scale"])
               if elong is not None else 0.5)
    entry_conf = _clip01(params["taper_weight"] * taper_q + (1.0 - params["taper_weight"]) * elong_q)
    if feat.get("texture_tiebreak_used"):
        entry_conf *= 0.7

    # Paint is only trusted when the entry looked visible on the
    # unpenalised score (as tuned); the penalties below then decide
    # whether the camera actually calls by paint.
    paint_trusted = entry_conf >= params["entry_visible_min"]

    clarity_conf = _clip01(feat["clarity_z"] / params["clarity_scale"])

    mm_per_px = feat.get("mm_per_px")
    margin_conf = (_clip01((feat["margin_mm"] / mm_per_px) / params["margin_sigma_px"])
                   if mm_per_px and mm_per_px > 1e-6 else 0.5)

    paint_ratio = feat.get("paint_margin_ratio")
    paint_conf = (_clip01(paint_ratio / params["paint_scale"])
                  if (paint_trusted and paint_ratio is not None) else None)

    agree_mm = feat.get("agree_mm")
    agree_conf = _clip01(1.0 - agree_mm / params["agree_scale_mm"]) if agree_mm is not None else None

    if feat.get("contam_suspected"):
        entry_conf *= params["contam_penalty"]
    if feat.get("entry_occluded"):
        entry_conf *= params["occl_entry_penalty"]
        entry_visible = False  # taper/elongation say nothing when the entry itself is hidden
    else:
        entry_visible = entry_conf >= params["entry_visible_min"]

    return dict(entry_conf=entry_conf, entry_visible=entry_visible, clarity_conf=clarity_conf,
                margin_conf=margin_conf, paint_conf=paint_conf, agree_conf=agree_conf)


def visual_decision(cams: dict, params: dict) -> dict:
    """Per-camera confidence + call, then definitive or weighted vote.
    Returns dict(call, method, cams={cam: {confidence, call_token, ...}})."""
    ok_cams = [c for c, r in cams.items() if r.get("ok")]

    mmpp = {c: cams[c]["mm_per_px"] for c in ok_cams if cams[c].get("mm_per_px")}
    best_mmpp = min(mmpp.values()) if mmpp else None

    per_cam = {}
    for c in ok_cams:
        feat = cams[c]
        parts = camera_confidence_parts(feat, params)
        if best_mmpp and feat.get("mm_per_px"):
            steep_conf = min(1.0, best_mmpp / feat["mm_per_px"])
        else:
            steep_conf = 0.5
        parts["steep_conf"] = steep_conf

        terms = [(params["w_entry"], parts["entry_conf"]),
                 (params["w_steep"], steep_conf),
                 (params["w_clarity"], parts["clarity_conf"]),
                 (params["w_margin"], parts["margin_conf"])]
        if parts["paint_conf"] is not None:
            terms.append((params["w_paint"], parts["paint_conf"]))
        if parts["agree_conf"] is not None:
            terms.append((params["w_agree"], parts["agree_conf"]))
        wsum = sum(w for w, _ in terms)
        confidence = sum(w * v for w, v in terms) / wsum if wsum > 0 else 0.0

        call_token = feat["paint_token"] if parts["entry_visible"] else feat["token"]
        per_cam[c] = dict(confidence=confidence, call_token=call_token,
                          entry_visible=parts["entry_visible"], parts=parts)

    if not per_cam:
        return dict(call=None, method="none", cams=per_cam)

    ranked = sorted(per_cam.items(), key=lambda kv: -kv[1]["confidence"])
    top_cam, top = ranked[0]
    if len(ranked) == 1:
        return dict(call=top["call_token"], method="solo_only_cam", cams=per_cam)

    second_conf = ranked[1][1]["confidence"]
    others_ambiguous = all(info["confidence"] < params["other_ambiguous_max"] for _, info in ranked[1:])
    if (top["confidence"] >= params["definitive_abs"]
            and (top["confidence"] - second_conf) >= params["definitive_margin"]
            and others_ambiguous):
        return dict(call=top["call_token"], method="definitive", cams=per_cam)

    tally: dict = {}
    for _, info in per_cam.items():
        tally[info["call_token"]] = tally.get(info["call_token"], 0.0) + info["confidence"]
    return dict(call=max(tally, key=tally.get), method="weighted_vote", cams=per_cam)


# --------------------------------------------------------------------------- #
# 2. Position fusion
# --------------------------------------------------------------------------- #

def _cov(x: dict, params: dict):
    ju, jv = x["jac_u"], x["jac_v"]
    su = params["sig_u"]
    hidden = x.get("entry_occluded") or x.get("contam_suspected") or (
        x.get("needle_w_mm") is not None and x["needle_w_mm"] >= params.get("barrel_mm", 99))
    if hidden:
        su *= params["infl"]
    sv = params["sig_v"]
    a = su * su * ju[0] * ju[0] + sv * sv * jv[0] * jv[0]
    b = su * su * ju[0] * ju[1] + sv * sv * jv[0] * jv[1]
    d = su * su * ju[1] * ju[1] + sv * sv * jv[1] * jv[1]
    return a, b, d


def _inv(a, b, d):
    det = a * d - b * b
    if det <= 1e-12:
        return None
    return d / det, -b / det, a / det


def fuse(cams: dict, params: dict):
    """(fused (x, y) or None, {cam: (board_xy, inverse covariance)} of the
    cameras kept)."""
    pts = {}
    for c, x in cams.items():
        if not x.get("ok") or x.get("jac_u") is None or x.get("board_xy") is None:
            continue
        inv = _inv(*_cov(x, params))
        if inv is None:
            continue
        pts[c] = (x["board_xy"], inv)
    while len(pts) >= 2:
        A = B = D = 0.0
        bx = by = 0.0
        for (p, (ia, ib, idd)) in pts.values():
            A += ia
            B += ib
            D += idd
            bx += ia * p[0] + ib * p[1]
            by += ib * p[0] + idd * p[1]
        det = A * D - B * B
        if det <= 1e-12:
            return None, pts
        X = (D * bx - B * by) / det
        Y = (-B * bx + A * by) / det
        worst, wr = None, -1
        for c, (p, (ia, ib, idd)) in pts.items():
            dx, dy = p[0] - X, p[1] - Y
            r2 = ia * dx * dx + 2 * ib * dx * dy + idd * dy * dy
            if r2 > wr:
                worst, wr = c, r2
        if wr <= params["chi2_max"]:
            return (X, Y), pts
        del pts[worst]
    return None, pts


def _fused_cov(pts: dict):
    A = B = D = 0.0
    for (_, (ia, ib, idd)) in pts.values():
        A += ia
        B += ib
        D += idd
    det = A * D - B * B
    return D / det, -B / det, A / det


def candidates(xy, cov, k: float, off: float) -> set[str]:
    """Every bed touched by the k-sigma ellipse around the fused point
    (16 directions, at half and full radius) plus the bed under it."""
    a, b, d = cov
    tr, dt = a + d, a * d - b * b
    disc = math.sqrt(max(tr * tr / 4 - dt, 0.0))
    l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 0.0)
    if abs(b) > 1e-12:
        e1 = (l1 - d, b)
    else:
        e1 = (1.0, 0.0) if a >= d else (0.0, 1.0)
    n = math.hypot(*e1)
    e1 = (e1[0] / n, e1[1] / n)
    e2 = (-e1[1], e1[0])
    s1, s2 = k * math.sqrt(l1), k * math.sqrt(l2)
    out = {label(xy[0], xy[1], off)}
    for i in range(16):
        t = 2 * math.pi * i / 16
        for f in (0.5, 1.0):
            x = xy[0] + f * (s1 * math.cos(t) * e1[0] + s2 * math.sin(t) * e2[0])
            y = xy[1] + f * (s1 * math.cos(t) * e1[1] + s2 * math.sin(t) * e2[1])
            out.add(label(x, y, off))
    return out


# --------------------------------------------------------------------------- #
# 3. Hybrid
# --------------------------------------------------------------------------- #

def decide(cams: dict, params: dict = PARAMS) -> dict:
    """Features of every camera -> dict(
        call: token or None,
        method: how it was reached,
        fused_xy: fused board point (or None), fused_cams: cams kept in it,
        candidates: sorted candidate beds (or None),
        cams: {cam: {confidence, call_token, entry_visible, parts}})."""
    off = params.get("inner_off", 0.0)
    if off:
        cams = apply_board_offset(cams, off)
    vis = visual_decision(cams, params)
    out = dict(call=vis["call"], method=vis["method"], cams=vis["cams"],
               fused_xy=None, fused_cams=[], candidates=None)
    xy, used = fuse(cams, params)
    if xy is None:
        out["method"] = "nofuse_" + vis["method"]
        return out
    out["fused_xy"] = (float(xy[0]), float(xy[1]))
    out["fused_cams"] = sorted(used)
    cand = candidates(xy, _fused_cov(used), params["k_sigma"], off)
    out["candidates"] = sorted(cand)
    if vis["call"] in cand:
        out["method"] = "agree_" + vis["method"]
        return out
    tally: dict = {}
    for info in vis["cams"].values():
        tally[info["call_token"]] = tally.get(info["call_token"], 0.0) + info["confidence"]
    inside = {t: w for t, w in tally.items() if t in cand}
    if inside:
        out["call"] = max(inside, key=inside.get)
        out["method"] = "restricted_vote"
    else:
        out["call"] = label(xy[0], xy[1], off)
        out["method"] = "fused_override"
    return out
