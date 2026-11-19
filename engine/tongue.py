"""The «tongue twister» look (references/README.md, референс 4): drawing for styles with "look": "tongue".

Two colours swap on every card like the classic look, but each line of the lyrics gets one of several
presentations ("modes", markup «[mode]» at the start of a line, else the style's "modes" cycle):

  big     the word in a narrow condensed grotesque, about half the frame high, rough cut edges
  push    the same, but the new card slides in from below on its own panel in 2-3 jumps, ragged edge
  line    the sung line in small bold lowercase, growing word by word (and syllable by syllable)
  small   the current word only, small and bold; on dark cards as an outline
  script  the word big and muted (filled on the light cards, outlined on the dark ones) with the same
          word in a handwritten script over it
  ghost   the word big and muted with the growing line in small type over it
  strip   the big word a little higher, the growing line on a band of the other colour along the bottom

Everything flickers between two shades of its colour 30 times a second ("accent2", "dark2") and a
big word slowly widens or narrows (±6 %/s, max 4 %) while it is on screen, as in the reference.
"""
import hashlib, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ('big', 'push', 'line', 'small', 'script', 'ghost', 'strip')
SETTLED = 0.6          # dt used for a still / a contact sheet: the word as it looks after the first moment


def mix(a, b, k):
    return tuple(int(round(x + (y - x) * k)) for x, y in zip(a, b))


def seed_of(*parts):
    return int(hashlib.md5('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def noise2(seed, w, h, cell):
    """Smooth 2D noise (unit std) of w×h with features about «cell» px apart."""
    cell = max(2, int(cell))
    gw, gh = w // cell + 3, h // cell + 3
    g = np.random.default_rng(seed).standard_normal((gh, gw)).astype(np.float32)
    a = np.asarray(Image.fromarray(g, 'F').resize((gw * cell, gh * cell), Image.BICUBIC))[cell:cell + h, cell:cell + w]
    return a / max(float(a.std()), 1e-6)


def noise1(seed, w, cell):
    rng = np.random.default_rng(seed)
    xs = np.arange(0, w + cell, cell)
    return np.interp(np.arange(w), xs, rng.standard_normal(len(xs)))


def warp(m, seed, H):
    """Rough, hand-cut letters: the mask pushed around by smooth noise (slanted stems, uneven edges)."""
    a = np.asarray(m, np.float32)
    h, w = a.shape
    dx = noise2(seed, w, h, 0.45 * H) * 0.006 * H + noise2(seed + 1, w, h, 0.06 * H) * 0.0006 * H
    dy = noise2(seed + 2, w, h, 0.45 * H) * 0.003 * H + noise2(seed + 3, w, h, 0.06 * H) * 0.0006 * H
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = np.clip(xx + dx, 0, w - 1.001)
    y = np.clip(yy + dy, 0, h - 1.001)
    x0, y0 = x.astype(np.int32), y.astype(np.int32)
    fx, fy = x - x0, y - y0
    v = (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
         + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)
    return Image.fromarray(v.clip(0, 255).astype(np.uint8))


def face(path, px, weight=None, index=0):
    f = ImageFont.truetype(path if os.path.isabs(path) else os.path.join(ROOT, path), px, index=index)
    if weight:
        f.set_variation_by_axes([weight])
    return f


class TongueType:
    def __init__(self, st, W, H):
        self.W, self.H, self.st = W, H, st
        probe = face(st['big_font'], 400)
        cap400 = -probe.getbbox('Н', anchor='ls')[1]
        self.big = face(st['big_font'], max(8, round(400 * st['big_cap'] * H / cap400)))
        self.cap = -self.big.getbbox('Н', anchor='ls')[1]
        l, _, r, _ = self.big.getbbox(st.get('big_width_word', 'WANT'), anchor='ls')
        self.sq = st['big_width'] * self.cap / (r - l)
        self.maxw = round(st.get('big_maxw', 0.9) * W)
        sw = st.get('small_weight')
        self.line_font = face(st['small_font'], round(st['line_size'] * H), sw)
        self.word_font = face(st['small_font'], round(st['word_size'] * H), sw)
        self.script_px = round(st['script_size'] * H)
        self.cache = {}

    # ------------------------------------------------------------ masks
    def big_mask(self, text):
        """Letters as an L image of the frame height, centred, squeezed to the reference width, rough."""
        m = self.cache.get(('big', text))
        if m is None:
            l, _, r, _ = self.big.getbbox(text, anchor='ls')
            pad = round(0.03 * self.H)
            im = Image.new('L', (r - l + 2 * pad, self.H), 0)
            ImageDraw.Draw(im).text((pad - l, round((self.H + self.cap) / 2)), text, font=self.big, fill=255, anchor='ls')
            w = round(im.width * self.sq)
            hs = 1.0
            if w > self.maxw:                  # too long: squeeze a little more, then make it lower
                hs = min(1.0, self.maxw / (w * self.st.get('big_squeeze', 0.85)))
                w = self.maxw
            m = im.resize((w, round(self.H * hs)), Image.LANCZOS)
            if hs < 1:
                full = Image.new('L', (w, self.H), 0)
                full.paste(m, (0, (self.H - m.height) // 2))
                m = full
            m = warp(m, seed_of('big', text), self.H)
            self.cache[('big', text)] = m
        return m

    def squeezed(self, text):
        """True when a long word had to be made much lower (under 70 %) to fit: worth splitting the card."""
        l, _, r, _ = self.big.getbbox(text, anchor='ls')
        w = (r - l + 0.06 * self.H) * self.sq
        return w > self.maxw and self.maxw / (w * self.st.get('big_squeeze', 0.85)) < 0.7

    def outline(self, m):
        k = ('outline', id(m))
        o = self.cache.get(k)
        if o is None:
            a = np.asarray(m) > 127
            r = max(1, round(0.0022 * self.H))
            grow = a.copy()
            shrink = a.copy()
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    s = np.roll(np.roll(a, dy, 0), dx, 1)
                    grow |= s
                    shrink &= s
            o = self.cache[k] = Image.fromarray(((grow & ~shrink) * 255).astype(np.uint8))
        return o

    def script_font(self, text):
        f = face(self.st['script_font'], self.script_px, index=self.st.get('script_index', 0))
        w = f.getlength(text)
        if w > 0.88 * self.W:
            f = face(self.st['script_font'], round(self.script_px * 0.88 * self.W / w), index=self.st.get('script_index', 0))
        return f

    # ------------------------------------------------------------ pieces
    def paste_big(self, im, text, colour, dt, k, dy=0, outline=False):
        m = self.big_mask(text)
        if outline:
            m = self.outline(m)
        sx = 1 + (1 if k % 2 == 0 else -1) * min(0.06 * dt, 0.04)
        if abs(sx - 1) > 1e-3:
            m = m.resize((max(1, round(m.width * sx)), m.height), Image.BILINEAR)
        im.paste(colour, ((self.W - m.width) // 2, dy), m)

    def fit(self, text, maxw):
        """The line font, made smaller (down to 60 %) until text fits on one line of maxw."""
        f = self.line_font
        w = f.getlength(text)
        if w <= maxw:
            return f
        px = max(round(f.size * 0.6), int(f.size * maxw / w))
        key = ('fit', px)
        if key not in self.cache:
            self.cache[key] = face(self.st['small_font'], px, self.st.get('small_weight'))
        return self.cache[key]

    def lines(self, text, font, maxw):
        out, cur = [], ''
        for w in text.split():
            t = (cur + ' ' + w).strip()
            if cur and font.getlength(t) > maxw:
                out.append(cur)
                cur = w
            else:
                cur = t
        return out + [cur] if cur else out

    def small_text(self, im, text, font, colour, cy, maxw=None, stroke=None, bg=None):
        d = ImageDraw.Draw(im)
        ls = self.lines(text, font, maxw or 0.8 * self.W)
        lh = font.size * 1.22
        y = cy - lh * (len(ls) - 1) / 2
        for ln in ls:
            if stroke:
                d.text((self.W / 2, y), ln, font=font, fill=bg, anchor='mm', stroke_width=stroke, stroke_fill=colour)
            else:
                d.text((self.W / 2, y), ln, font=font, fill=colour, anchor='mm')
            y += lh

    def ragged(self, top, seed):
        """Rows ≥ a ragged line around «top» (px) as an L mask of the frame: torn-paper panel edge."""
        W, H = self.W, self.H
        edge = top + noise1(seed, W, round(0.05 * H)) * 0.004 * H + noise1(seed + 1, W, round(0.25 * H)) * 0.006 * H
        yy = np.arange(H)[:, None]
        return Image.fromarray(((yy >= edge[None, :]) * 255).astype(np.uint8))

    # ------------------------------------------------------------ one card
    def card(self, c, bg, fg, dt):
        """c = (mode, text, small, script, k, on_acc, full line): the card at dt s after it appeared."""
        mode, text, small, script, k, on_acc, full = c
        W, H = self.W, self.H
        im = Image.new('RGB', (W, H), bg)
        if not text and not small:
            return im
        if mode in ('big', 'push'):
            self.paste_big(im, text, fg, dt, k)
        elif mode == 'strip':
            f = self.fit(full or small, 0.92 * W)
            n = len(self.lines(full or small, f, 0.92 * W))
            band = max(round(0.12 * H), round(n * f.size * 1.22 + 0.05 * H))
            self.paste_big(im, text, fg, dt, k, dy=-round(0.06 * H) - (band - round(0.12 * H)) // 2)
            im.paste(fg, (0, 0), self.ragged(H - band, seed_of('strip', k)))
            self.small_text(im, small, f, bg, H - band / 2 + 0.004 * H, maxw=0.92 * W)
        elif mode == 'line':
            self.small_text(im, small, self.line_font, fg, H / 2)
        elif mode == 'small':
            word = script or small
            if on_acc:
                self.small_text(im, word, self.word_font, fg, H / 2, maxw=0.9 * W)
            else:
                self.small_text(im, word, self.word_font, fg, H / 2, maxw=0.9 * W,
                                stroke=max(1, round(0.0035 * H)), bg=bg)
        elif mode == 'script':
            if on_acc:
                self.paste_big(im, text, mix(bg, fg, self.st.get('ghost', 0.22)), dt, k)
            else:
                self.paste_big(im, text, mix(bg, fg, 0.55), dt, k, outline=True)
            f = self.script_font(script)
            ImageDraw.Draw(im).text((W / 2, H / 2 + 0.03 * H), script, font=f, fill=fg, anchor='mm')
        elif mode == 'ghost':
            self.paste_big(im, text, mix(bg, fg, self.st.get('ghost', 0.22) * (1 if on_acc else 0.8)), dt, k)
            self.small_text(im, small, self.line_font, fg, H / 2)
        return im


def second(st):
    """The flicker shades: "accent2"/"dark2", else as in the reference: the light colour 27 % of the way to
    white, the dark one 27 % brighter."""
    acc, dark = tuple(st['accent']), tuple(st['dark'])
    a2 = tuple(st['accent2']) if 'accent2' in st else tuple(int(round(c + 0.27 * (255 - c))) for c in acc)
    d2 = tuple(st['dark2']) if 'dark2' in st else tuple(min(255, int(round(c * 1.27 + 2))) for c in dark)
    return {acc: a2, dark: d2}


def colours(st, bg, fg, shade):
    """The second shade of both colours on odd flicker phases ("flicker": false turns it off)."""
    if not shade or st.get('flicker') is False:
        return bg, fg
    sw = second(st)
    return sw.get(bg, bg), sw.get(fg, fg)


def push_top(st, dt):
    """Where the new panel's top edge is (fraction of the frame) dt s after the card appears: 0 = done."""
    for t, f in reversed(st.get('push_steps', [[0, 0.5], [0.05, 0.22], [0.1, 0]])):
        if dt >= t:
            return f
    return 0


def compose(T, state, t=None, shade=0):
    """state = (text, bg, fg, '', (mode, small, script, k, t_card, on_acc, prev, full line)); t = video s or None."""
    text, bg, fg, _, (mode, small, script, k, t_card, on_acc, prev, full) = state
    dt = SETTLED if t is None else max(0.0, t - t_card)
    bg, fg = colours(T.st, bg, fg, shade)
    im = T.card((mode, text, small, script, k, on_acc, full), bg, fg, dt)
    if mode == 'push' and prev and t is not None:
        f = push_top(T.st, dt)
        if f > 0:
            ptext, pbg, pfg, pmode, psmall, pscript, pk, pon, pfull = prev
            pbg, pfg = colours(T.st, pbg, pfg, shade)
            old = T.card((pmode, ptext, psmall, pscript, pk, pon, pfull), pbg, pfg, SETTLED + dt)
            top = round(f * T.H)
            frame = Image.new('RGB', (T.W, T.H), pbg)
            frame.paste(old, (0, -round((T.H - top) * 0.5)))
            new = Image.new('RGB', (T.W, T.H), bg)
            new.paste(im, (0, top))
            frame.paste(new, (0, 0), T.ragged(top, seed_of('push', k)))
            im = frame
    return im
