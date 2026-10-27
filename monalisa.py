import math
import os 
import numpy as np
import pygame

pygame.init()
w, h = 540, 960
screen = pygame.display.set_mode((w, h))
pygame.display.set_caption("mona lisa, painted by code")
clock = pygame.time.Clock()
fps = 60
rng = np.random.default_rng(5)
 
bg = (9, 6, 26)
frame_col = (150, 120, 80)
 
here = os.path.dirname(os.path.abspath(__file__))
ref_small = pygame.image.load(os.path.join(here, "mona_lisa.png")).convert()
cw = w
ch = int(round(cw * ref_small.get_height() / ref_small.get_width()))
top = (h - ch) // 2
ref = pygame.transform.smoothscale(ref_small, (cw, ch))
 
 
def blur(surf, sigma):
    f = max(1.0, sigma)
    small = pygame.transform.smoothscale(surf, (max(2, int(cw / f)), max(2, int(ch / f))))
    return pygame.transform.smoothscale(small, (cw, ch))

# a little unsharp mask so the enlarged picture keeps some crispness

ref_a = pygame.surfarray.array3d(ref).astype(np.float32)
soft_a = pygame.surfarray.array3d(blur(ref, 2.5)).astype(np.float32)
sharp_a = np.clip(ref_a + 0.7 * (ref_a - soft_a), 0, 255)
ref = pygame.surfarray.make_surface(sharp_a.astype(np.uint8))
 
xx, yy = np.meshgrid(np.arange(cw), np.arange(ch), indexing="ij")
flat_theta = 0.45 * np.sin(xx / 80.0) + 0.35 * np.cos(yy / 95.0) + 0.5
 
# where the eye should look: face, smile and hands get the finest strokes

face = np.exp(-(((xx / cw - 0.47) / 0.13) ** 2 + ((yy / ch - 0.27) / 0.11) ** 2))
hands = np.exp(-(((xx / cw - 0.45) / 0.22) ** 2 + ((yy / ch - 0.84) / 0.08) ** 2))
fine_lum = pygame.surfarray.array3d(blur(ref, 1.5)).astype(np.float32).mean(axis=2)
fgx, fgy = np.gradient(fine_lum)
fmag = np.hypot(fgx, fgy)
saliency = 0.12 + 0.9 * np.clip(fmag / (np.percentile(fmag, 92) + 1e-6), 0, 1) + 1.1 * face + 0.7 * hands

# layers: brush radius, how many passes, the time window in seconds

radii = [34, 20, 12, 7, 4.0, 2.6]
passes = [2, 2, 2, 1, 1, 1]
t0s = [0.0, 2.5, 6.0, 10.5, 16.0, 22.0]
t1s = [2.5, 6.0, 10.5, 16.0, 22.0, 28.0]
 
S_x, S_y, S_th, S_len, S_r, S_col, S_layer = [], [], [], [], [], [], []
layer_counts = []
for li, r in enumerate(radii):
    sigma = max(1.0, r * 0.5)
    bl = blur(ref, sigma)
    col_a = pygame.surfarray.array3d(bl).astype(np.float32)
    lum = col_a.mean(axis=2)
    gx, gy = np.gradient(lum)
    mag = np.hypot(gx, gy)
    mag_n = np.clip(mag / (np.percentile(mag, 90) + 1e-6), 0, 1)
    edge_theta = np.arctan2(gy, gx) + math.pi / 2
    vx = mag_n * np.cos(2 * edge_theta) + (1 - mag_n) * np.cos(2 * flat_theta)
    vy = mag_n * np.sin(2 * edge_theta) + (1 - mag_n) * np.sin(2 * flat_theta)
    theta = 0.5 * np.arctan2(vy, vx)
 
    cell = r * 1.15
    gxs = np.arange(cell / 2, cw, cell)
    gys = np.arange(cell / 2, ch, cell)
    gX, gY = np.meshgrid(gxs, gys)
    pts = []
    for _ in range(passes[li]):
        px = gX.ravel() + rng.normal(0, cell * 0.35, gX.size)
        py = gY.ravel() + rng.normal(0, cell * 0.35, gY.size)
        pts.append(np.stack([px, py], axis=1))
    pts = np.concatenate(pts)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < cw - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < ch - 1)]
    ix, iy = pts[:, 0].astype(int), pts[:, 1].astype(int)
    if li >= 3:
        keep = rng.random(len(pts)) < np.clip(saliency[ix, iy] / (0.9 if li == 3 else 1.4 if li == 4 else 1.9), 0, 1)
        pts, ix, iy = pts[keep], ix[keep], iy[keep]
    order = rng.permutation(len(pts))
    pts, ix, iy = pts[order], ix[order], iy[order]
    n = len(pts)
    length = r * rng.uniform(1.8, 3.0, n) / (1 + 1.6 * mag_n[ix, iy])
    length = np.maximum(length, r * 1.1)
    col = np.clip(col_a[ix, iy] + rng.normal(0, 6, (n, 3)), 0, 255)
    S_x.append(pts[:, 0]); S_y.append(pts[:, 1]); S_th.append(theta[ix, iy]); S_len.append(length)
    S_r.append(np.full(n, r)); S_col.append(col); S_layer.append(np.full(n, li))
    layer_counts.append(n)
 
S_x, S_y, S_th, S_len = map(np.concatenate, (S_x, S_y, S_th, S_len))
S_r, S_col, S_layer = map(np.concatenate, (S_r, S_col, S_layer))
S_jit = rng.normal(0, 9, (len(S_x), 7))
layer_start = np.concatenate([[0], np.cumsum(layer_counts)])
 
canvas = pygame.Surface((cw, ch))
canvas.fill((26, 19, 14))
drawn = 0
 
 
def paint_stroke(i):
    r = S_r[i]
    th = S_th[i]
    dx, dy = math.cos(th), math.sin(th)
    nx, ny = -dy, dx
    half = S_len[i] / 2
    hw = r * 0.55
    nb = int(min(7, max(2, r / 2.5)))
    wd = max(1, int(2 * hw / nb * 1.3))
    base = S_col[i]
    for j in range(nb):
        o = (j / (nb - 1) - 0.5) * 2 * hw if nb > 1 else 0.0
        c = base + S_jit[i, j]
        color = (int(min(255, max(0, c[0]))), int(min(255, max(0, c[1]))), int(min(255, max(0, c[2]))))
        cx0 = S_x[i] + nx * o
        cy0 = S_y[i] + ny * o
        pygame.draw.line(canvas, color, (cx0 - dx * half, cy0 - dy * half), (cx0 + dx * half, cy0 + dy * half), wd)
 
 
def paint_until(t):
    global drawn
    # strokes are stored layer by layer, so the count per layer decides which ones are visible
    target = []
    for li in range(len(radii)):
        f = min(1.0, max(0.0, (t - t0s[li]) / (t1s[li] - t0s[li])))
        f = f * f * (3 - 2 * f) * 0.5 + f * 0.5
        target.append(int(layer_counts[li] * f))
    total = sum(target)
    if total < drawn:
        canvas.fill((26, 19, 14))
        drawn = 0
    # draw in layer order up to the total (earlier layers finish before later ones start)
    k = 0
    for li in range(len(radii)):
        for s in range(layer_start[li], layer_start[li] + target[li]):
            if k >= drawn:
                paint_stroke(s)
                drawn += 1
            k += 1
 
 
def draw(t):
    paint_until(min(t, t1s[-1]))
    screen.fill(bg)
    screen.blit(canvas, (0, top))
    pygame.draw.rect(screen, frame_col, (0, top - 2, cw, ch + 4), 2)
 
 
def run():
    running = True
    start = pygame.time.get_ticks() / 1000.0
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        draw(pygame.time.get_ticks() / 1000.0 - start)
        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()

if __name__ == "__main__":
    run()
