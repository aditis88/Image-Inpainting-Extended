# inpainting.py

import cv2
import numpy as np
import heapq

INF = 1e1
EPS = 1e-1
KNOWN = 0
BAND = 1
INSIDE = 2

def create_mask(filename, threshold=20, kernel_dim=(5, 5)):
    damaged_img = cv2.imread(filename=filename)
    gray_img = cv2.cvtColor(damaged_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones(kernel_dim, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def _pixel_gradient(y, x, height, width, vals, flags):
    val = vals[y, x]
    prev_y, next_y = y - 1, y + 1
    if prev_y < 0 or next_y >= height:
        grad_y = INF
    else:
        grad_y = compute_gradient_component(vals, flags, y, x, axis='y')

    prev_x, next_x = x - 1, x + 1
    if prev_x < 0 or next_x >= width:
        grad_x = INF
    else:
        grad_x = compute_gradient_component(vals, flags, y, x, axis='x')

    return grad_y, grad_x

def compute_gradient_component(vals, flags, y, x, axis='x'):
    if axis == 'y':
        prev, next_ = y - 1, y + 1
        val = vals[y, x]
        flag_prev, flag_next = flags[prev, x], flags[next_, x]
        return gradient_calc(val, vals[prev, x], vals[next_, x], flag_prev, flag_next)
    else:
        prev, next_ = x - 1, x + 1
        val = vals[y, x]
        flag_prev, flag_next = flags[y, prev], flags[y, next_]
        return gradient_calc(val, vals[y, prev], vals[y, next_], flag_prev, flag_next)

def gradient_calc(val, prev_val, next_val, flag_prev, flag_next):
    if flag_prev != INSIDE and flag_next != INSIDE:
        return (next_val - prev_val) / 2.0
    elif flag_prev != INSIDE:
        return val - prev_val
    elif flag_next != INSIDE:
        return next_val - val
    return 0.0

def inpaint_pixel(img, dists, flags, y, x, radius, height, width):
    dist = dists[y, x]
    dist_grad_y, dist_grad_x = _pixel_gradient(y, x, height, width, dists, flags)
    pixel_sum = np.zeros((3), dtype=float)
    weight_sum = 0.0

    for nb_y in range(y - radius, y + radius + 1):
        if nb_y < 0 or nb_y >= height:
            continue
        for nb_x in range(x - radius, x + radius + 1):
            if nb_x < 0 or nb_x >= width:
                continue
            if flags[nb_y, nb_x] == INSIDE:
                continue

            dir_y, dir_x = y - nb_y, x - nb_x
            dir_length = np.sqrt(dir_y ** 2 + dir_x ** 2)
            if dir_length > radius:
                continue

            dir_factor = abs(dir_y * dist_grad_y + dir_x * dist_grad_x) or EPS
            nb_dist = dists[nb_y, nb_x]
            level_factor = 1.0 / (1.0 + abs(nb_dist - dist))
            dist_factor = 1.0 / (dir_length * (dir_y**2 + dir_x**2))
            weight = abs(dir_factor * dist_factor * level_factor)

            pixel_sum += weight * img[nb_y, nb_x]
            weight_sum += weight

    return pixel_sum / weight_sum

def get_T_flags_narrow_band(image, mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    T = np.zeros_like(gray, dtype=float)
    T[mask == 255] = INF
    f = np.where(mask == 255, INSIDE, KNOWN)
    narrow_band = []

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if f[i, j] == INSIDE:
                for k, l in [(i-1, j), (i, j-1), (i+1, j), (i, j+1), (i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]:
                    if f[k, l] == KNOWN:
                        f[k, l] = BAND
                        T[i, j] = 0
                        heapq.heappush(narrow_band, (T[i, j], (i, j)))
                        break
    return T, f, narrow_band

def inpaint_image(image, mask, radius=3):
    T, f, narrow_band = get_T_flags_narrow_band(image, mask)
    height, width = image.shape[:2]

    while narrow_band:
        T_val, (i, j) = heapq.heappop(narrow_band)
        f[i, j] = KNOWN

        for k, l in [(i-1, j), (i, j-1), (i+1, j), (i, j+1)]:
            if k < 1 or l < 1 or k >= mask.shape[0] - 1 or l >= mask.shape[1] - 1:
                continue
            if f[k, l] != KNOWN:
                if f[k, l] == INSIDE:
                    T[k, l] = min(solve(T, f, k-1, l, k, l-1),
                                  solve(T, f, k+1, l, k, l-1),
                                  solve(T, f, k-1, l, k, l+1),
                                  solve(T, f, k+1, l, k, l+1))
                    pixel_vals = inpaint_pixel(image, T, f, k, l, radius, height, width)
                    image[k, l] = pixel_vals
                    heapq.heappush(narrow_band, (T[k, l], (k, l)))
                    f[k, l] = BAND

    return image

def solve(T, f, i1, j1, i2, j2):
    if not (0 <= i1 < T.shape[0] and 0 <= j1 < T.shape[1] and
            0 <= i2 < T.shape[0] and 0 <= j2 < T.shape[1]):
        return INF

    flag1, flag2 = f[i1, j1], f[i2, j2]
    if flag1 == KNOWN and flag2 == KNOWN:
        dist1, dist2 = T[i1, j1], T[i2, j2]
        d = 2.0 - (dist1 - dist2) ** 2
        if d > 0.0:
            r = np.sqrt(d)
            s = (dist1 + dist2 - r) / 2.0
            if s >= dist1 and s >= dist2:
                return s
            s += r
            return s if s >= dist1 and s >= dist2 else INF
    elif flag1 == KNOWN:
        return 1.0 + T[i1, j1]
    elif flag2 == KNOWN:
        return 1.0 + T[i2, j2]
    return INF
