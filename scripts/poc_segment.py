#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, cv2, numpy as np

# kernels chicos (rápidos en 84x84)
K3 = np.ones((3,3), np.uint8)
K5 = np.ones((5,5), np.uint8)

# ---------- utils ----------
def _circ_hue_dist(h, h0):
    d = np.abs(h - h0)
    return np.minimum(d, 180 - d)

def _inside_and_center_score(ball_xy, poly):
    if poly is None or ball_xy is None:
        return False, 0.0
    cx, cy = poly[:,0].mean(), poly[:,1].mean()
    rmax = float(np.sqrt(((poly[:,0]-cx)**2 + (poly[:,1]-cy)**2).max()) + 1e-8)
    inside = cv2.pointPolygonTest(poly, (float(ball_xy[0]), float(ball_xy[1])), False) >= 0
    d = float(np.hypot(ball_xy[0]-cx, ball_xy[1]-cy))
    return bool(inside), float(1.0 - min(d/rmax, 1.0))

# ---------- TOP: por HUE dominante (permite sombras/brillos) ----------
def segment_top_by_hue(hsv, s_min=50, bandwidth=12, min_area_frac=0.05):
    """
    1) toma pixeles con S >= s_min (color "confiable")
    2) calcula HUE dominante (modo del histograma)
    3) máscara: |H - H_mode|_circular <= bandwidth
    4) morfología y quedarse con el componente más grande
    Devuelve: (poly_top Nx2 float32, mask_top uint8)
    """
    H, S, V = cv2.split(hsv)
    # máscara de "pixeles con color"
    color_mask = (S >= s_min).astype(np.uint8)
    if cv2.countNonZero(color_mask) == 0:
        return None, np.zeros_like(H, np.uint8)

    # histograma de H con peso por saturación (más saturado = más peso)
    h_vals = H[color_mask > 0].ravel()
    s_vals = S[color_mask > 0].ravel().astype(np.float32)
    # normalizar pesos ~ [0..1] para estabilidad
    weights = (s_vals / (s_vals.max() + 1e-6))
    hist = np.bincount(h_vals, weights=weights, minlength=180)
    h_mode = int(np.argmax(hist))

    # banda alrededor del H dominante (distancia circular)
    band = (_circ_hue_dist(H, h_mode) <= bandwidth).astype(np.uint8) * 255
    mask = cv2.bitwise_and(band, color_mask * 255)

    # cerrar huecos y limpiar ruido
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K5, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  K3, 1)

    if cv2.countNonZero(mask) == 0:
        return None, mask

    # componente más grande por área (descarta letras/ruido)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    Hh, Ww = H.shape
    area_img = Hh * Ww
    area_min = int(min_area_frac * area_img)

    best_i, best_a = -1, -1
    for i in range(1, num):  # 0 = fondo
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < area_min: 
            continue
        if a > best_a:
            best_a, best_i = a, i
    if best_i == -1:
        return None, mask

    top_mask = (labels == best_i).astype(np.uint8) * 255

    # contorno -> convexo -> polilínea
    cnts, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    poly = cv2.approxPolyDP(hull, 2.0, True).reshape(-1, 2).astype(np.float32)
    return poly, top_mask

# ---------- BALL: S baja + V alta ADAPTATIVOS (permite sombras/oclusión) ----------
def segment_ball_adaptive(hsv, p_s=35, p_v=80,
                          min_area_px=40, max_area_frac=0.03,
                          restrict_poly=None, margin_px=0,
                          close_iters=1):
    """
    Bola robusta y rápida:
      - S <= perc(S,p_s)  ∧  V >= perc(V,p_v)  (adaptativo por frame)
      - closing para unir fragmentos (oclusión) y open para ruido
      - tomar SOLO el componente más grande en [min_area_px, max_area_frac * área]
    Devuelve: (center(x,y), r_eq, mask_clean, present(bool), confidence[0..1])
    """
    H, W = hsv.shape[:2]
    S = hsv[...,1].astype(np.uint8)
    V = hsv[...,2].astype(np.uint8)

    s_th = int(np.percentile(S, p_s))
    v_th = int(np.percentile(V, p_v))

    m_s = (S <= s_th).astype(np.uint8) * 255
    m_v = (V >= v_th).astype(np.uint8) * 255
    mask = cv2.bitwise_and(m_s, m_v)

    if restrict_poly is not None and len(restrict_poly) >= 3:
        pm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(pm, [restrict_poly.astype(np.int32)], 255)
        if margin_px > 0:
            pm = cv2.dilate(pm, np.ones((margin_px, margin_px), np.uint8), 1)
        mask = cv2.bitwise_and(mask, pm)

    # cierra huecos (oclusión leve) y quita speckles
    if close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K5, close_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K3, 1)

    if cv2.countNonZero(mask) == 0:
        return None, None, mask, False, 0.0

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area_img = H * W
    area_max = int(max_area_frac * area_img)

    best_i, best_a = -1, -1
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area_px or a > area_max:
            continue
        if a > best_a:
            best_a, best_i = a, i

    if best_i == -1:
        return None, None, mask, False, 0.0

    cx, cy = map(float, cents[best_i])
    r_eq = float((best_a / np.pi) ** 0.5)

    # confianza = mezcla área normalizada y circularidad (barato)
    comp = (labels == best_i).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = 0.0
    if cnts:
        A = float(cv2.contourArea(cnts[0])); P = float(cv2.arcLength(cnts[0], True)) + 1e-6
        circ = float(4.0 * np.pi * A / (P * P))  # 1.0 = círculo perfecto
    area_norm = min(1.0, max(0.0, (best_a - min_area_px) / max(1, area_max - min_area_px)))
    confidence = float(0.5 * area_norm + 0.5 * min(1.0, circ))

    clean = comp
    return np.array([cx, cy], np.float32), r_eq, clean, True, confidence

# ---------- pipeline por frame ----------
def process_observation(frame_bgr,
                        restrict_ball_to_top=False,
                        top_s_min=50, top_bandwidth=12,
                        ball_p_s=35, ball_p_v=80,
                        ball_min_area_px=40, ball_max_area_frac=0.03,
                        close_iters=1):
    """
    Entrada: BGR (np.ndarray). Salida: dict con
      poly_top, ball_xy, ball_r, ball_present, ball_confidence,
      inside, center_score, top_mask, ball_mask
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # TOP
    poly, top_mask = segment_top_by_hue(
        hsv, s_min=top_s_min, bandwidth=top_bandwidth, min_area_frac=0.05
    )

    # BALL
    restrict_poly = poly if (restrict_ball_to_top and poly is not None) else None
    ball_xy, ball_r, ball_mask, present, conf = segment_ball_adaptive(
        hsv, p_s=ball_p_s, p_v=ball_p_v,
        min_area_px=ball_min_area_px, max_area_frac=ball_max_area_frac,
        restrict_poly=restrict_poly, margin_px=2, close_iters=close_iters
    )

    inside, cscore = (False, 0.0)
    if present and poly is not None:
        inside, cscore = _inside_and_center_score(ball_xy, poly)

    return {
        "poly_top": poly,
        "ball_xy": ball_xy,
        "ball_r": ball_r,
        "ball_present": bool(present),
        "ball_confidence": conf,
        "inside": inside,
        "center_score": cscore,
        "top_mask": top_mask,
        "ball_mask": ball_mask,
    }

# ---------- overlay (para debug, no usar en training) ----------
def draw_overlay(bgr, out):
    img = bgr.copy()
    if out["poly_top"] is not None:
        cv2.polylines(img, [out["poly_top"].astype(np.int32)], True, (0,255,0), 2, cv2.LINE_AA)
    if out["ball_xy"] is not None and out["ball_r"] is not None and out["ball_present"]:
        c = (int(out["ball_xy"][0]), int(out["ball_xy"][1]))
        cv2.circle(img, c, max(2, int(out["ball_r"])), (0,0,255), 2, cv2.LINE_AA)
        cv2.circle(img, c, 3, (0,0,255), -1, cv2.LINE_AA)
    tag = f"in={out['inside']}  c={out['center_score']:.3f}  present={out['ball_present']}({out['ball_confidence']:.2f})"
    cv2.rectangle(img, (10,10), (10+8*len(tag), 40), (0,0,0), -1)
    cv2.putText(img, tag, (14,34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return img

# ---------- CLI de prueba ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="ruta a la imagen de observación")
    ap.add_argument("--restrict", action="store_true", help="restringe la bola al área del top")
    ap.add_argument("--show", action="store_true", help="muestra ventanas OpenCV")
    ap.add_argument("--save-prefix", type=str, default=None, help="prefijo para PNGs")
    # knobs (ajusta si hace falta, defaults funcionan bien en 84x84 sim)
    ap.add_argument("--top-s-min", type=int, default=50)
    ap.add_argument("--top-bandwidth", type=int, default=12)
    ap.add_argument("--ball-p-s", type=int, default=35)
    ap.add_argument("--ball-p-v", type=int, default=80)
    ap.add_argument("--ball-min-area", type=int, default=40)
    ap.add_argument("--ball-max-area-frac", type=float, default=0.03)
    ap.add_argument("--close-iters", type=int, default=1)
    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(args.image)
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)

    out = process_observation(
        bgr,
        restrict_ball_to_top=args.restrict,
        top_s_min=args.top_s_min,
        top_bandwidth=args.top_bandwidth,
        ball_p_s=args.ball_p_s,
        ball_p_v=args.ball_p_v,
        ball_min_area_px=args.ball_min_area,
        ball_max_area_frac=args.ball_max_area_frac,
        close_iters=args.close_iters
    )

    overlay = draw_overlay(bgr, out)
    print(f"inside={out['inside']}  center_score={out['center_score']:.3f}  "
          f"present={out['ball_present']}  conf={out['ball_confidence']:.2f}")

    if args.save_prefix:
        cv2.imwrite(args.save_prefix + "_overlay.png",   overlay)
        cv2.imwrite(args.save_prefix + "_top_mask.png",  out["top_mask"])
        cv2.imwrite(args.save_prefix + "_ball_mask.png", out["ball_mask"])

    if args.show:
        cv2.imshow("RGB", bgr)
        cv2.imshow("Overlay", overlay)
        cv2.imshow("Top mask", out["top_mask"])
        cv2.imshow("Ball mask", out["ball_mask"])
        cv2.waitKey(0)