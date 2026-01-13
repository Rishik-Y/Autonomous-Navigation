from . import config as C

def scale_px_per_m(zoom: float) -> float:
    return C.BASE_SCALE * zoom

def world_to_screen(xm, ym, cam, win_w, win_h):
    S = scale_px_per_m(cam.zoom)
    xs = win_w * 0.5 + (xm - cam.cx) * S
    ys = win_h * 0.5 + (cam.cy - ym) * S
    return int(xs), int(ys)

def screen_to_world(xs, ys, cam, win_w, win_h):
    S = scale_px_per_m(cam.zoom)
    xm = cam.cx + (xs - win_w * 0.5) / S
    ym = cam.cy - (ys - win_h * 0.5) / S
    return xm, ym

def zoom_at_cursor(cam, factor, mouse_pos, win_w, win_h):
    if factor <= 0.0:
        return
    mx, my = mouse_pos
    wx_before, wy_before = screen_to_world(mx, my, cam, win_w, win_h)
    cam.zoom *= factor
    cam.zoom = max(0.05, min(20.0, cam.zoom))
    wx_after, wy_after = screen_to_world(mx, my, cam, win_w, win_h)
    cam.cx += (wx_before - wx_after)
    cam.cy += (wy_before - wy_after)
