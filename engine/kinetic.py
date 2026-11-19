"""Kinetic-typography videos in the look of references/ (TikTok «слово на весь экран»): the render
core of dynamic_snippets.

One word (or a short group) at a time in a heavy grotesque stretched to the full frame height,
revealed syllable by syllable on the vocal (НАП → НАПЛЕ → НАПЛЕВАТЬ); every new card flips the
background between the accent colour (black letters) and black (accent letters). Options: a small
white italic line in the middle that accumulates the sung line (the «киса подай мне знак» look),
colour flips on the beat while a card is held (strobe), title cards at fixed times.

Inputs (see README.md): videos/videos.json (one spec per video), videos/lyrics/<vid>.txt (lyrics with
markup), videos/align/<vid>.json (MMS_FA letter times, ./snip align), styles/styles.json,
tracks/<track>/{track.json, source.wav, stems/}. Normally driven through ./snip; directly:

  python3 -I engine/kinetic.py plan dora1
  python3 -I engine/kinetic.py still dora1 --t=8
  python3 -I engine/kinetic.py video dora1
"""
import hashlib, json, os, re, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tongue  # noqa: E402  the «tongue twister» look (styles with "look": "tongue")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS_DIR = os.path.join(ROOT, 'tracks')
VIDEOS_DIR = os.path.join(ROOT, 'videos')
OUT = os.path.join(ROOT, 'output')
CACHE = os.path.join(ROOT, 'cache')
SR = 44100
SIZES = {'16x9': (1920, 1080), '9x16': (1080, 1920), '1x1': (1080, 1080), '4x5': (1080, 1350), '4x3': (1440, 1080)}

TRACKS = {}   # id → track.json
VIDEOS = {}   # vid → spec from videos/videos.json
STYLES = {}   # name → style from styles/styles.json ("_base" holds the shared typography)

LEAD = 0.02           # a step shows this much before its syllable starts
MIN_STEP = 0.07       # a reveal step shorter than this is folded into the step before it
MIN_CARD = 0.12       # a card shorter than this joins the next one («и ничего» → «И НИЧЕГО»)
SNAP = (0.08, 0.04)   # a syllable moves to a vocal onset up to 80 ms before / 40 ms after its aligned time
PREP = {'в', 'с', 'к', 'у', 'о'}          # one-letter prepositions are sung with the next word
VOWELS = set('аеёиоуыэюяaeiouy')


def load():
    for tid in sorted(os.listdir(TRACKS_DIR)):
        p = os.path.join(TRACKS_DIR, tid, 'track.json')
        if os.path.exists(p):
            TRACKS[tid] = json.load(open(p, encoding='utf-8'))
    p = os.path.join(VIDEOS_DIR, 'videos.json')
    if os.path.exists(p):
        VIDEOS.update(json.load(open(p, encoding='utf-8')))
    STYLES.update(json.load(open(os.path.join(ROOT, 'styles', 'styles.json'), encoding='utf-8')))


def track_file(track, *parts):
    return os.path.join(TRACKS_DIR, track, *parts)


def lyrics_file(vid):
    return os.path.join(VIDEOS_DIR, 'lyrics', f'{vid}.txt')


def align_file(vid):
    return os.path.join(VIDEOS_DIR, 'align', f'{vid}.json')


def style(vid):
    """styles.json "_base", then the video's style, then the spec's own "tweak" keys."""
    v = VIDEOS[vid]
    name = v.get('style', 'navy')
    if name not in STYLES:
        raise SystemExit(f'style «{name}» is not in styles/styles.json')
    st = dict(STYLES['_base'])
    st.update(STYLES[name])
    st.update(v.get('tweak', {}))
    return st


def size(vid):
    return SIZES[VIDEOS[vid].get('size', '16x9')]


# ---------------------------------------------------------------- lyrics
def syllables(text):
    """Pieces of a word for the reveal, as the references cut them: НАП|ЛЕ|ВАТЬ, ПОД|РУ|ГИ, РОК|СТАР,
    ЗА|Е|БА|ЛИ. One consonant between two vowels opens the next piece, of two or more the first stays
    behind; ь/ъ stay with their consonant, hyphens and punctuation with the piece before them. Latin
    words: adjacent vowels are one, a final consonant + e is silent."""
    low = text.lower()
    letters = [i for i, c in enumerate(low) if c.isalpha()]
    latin = all(low[i].isascii() for i in letters)
    vow = [i for i in letters if low[i] in VOWELS]
    if latin:
        vow = [i for i in vow if not (i - 1 in vow)]                      # «ea», «ou»: one sound
        if len(vow) > 1 and low.rstrip('!?.…').endswith('e') and vow[-1] == len(low.rstrip('!?.…')) - 1:
            vow = vow[:-1]
    if len(vow) < 2:
        return [text]
    cuts = []
    for a, b in zip(vow, vow[1:]):
        if latin:
            a = max(i for i in letters if i < b and (i == a or low[i] in VOWELS))
        cons = [i for i in letters if a < i < b and low[i] not in 'ьъ']
        cuts.append(a + 1 if not cons else cons[0] if len(cons) == 1 else cons[1])
    # what is not a letter right before a cut (a hyphen) stays with the left piece
    cuts = [c + len(text[c:]) - len(text[c:].lstrip('-')) for c in cuts]
    return [p for p in (text[i:j] for i, j in zip([0] + cuts, cuts + [len(text)])) if p]


def on_screen(tok):
    """A lyric token as it is printed: capitals, no commas/quotes, «-» for any dash."""
    t = re.sub(r'\*$', '', re.sub(r'\{.*?\}', '', tok)).replace('|', '').replace('_', ' ')   # «*бать»: a * inside stays
    t = re.sub(r'[,;:«»"“”„]', '', t).replace('—', '-').replace('–', '-')
    return t.upper()


def in_sub(tok):
    """A lyric token as the italic line shows it: the user's text and punctuation, no markup."""
    return re.sub(r'\*$', '', re.sub(r'\{.*?\}', '', tok)).replace('|', '').replace('_', ' ').replace('—', '-')


def word(raw, sung, line, sub):
    say = re.search(r'\{(.*?)\}', raw)
    text = on_screen(raw)
    if say:                                       # 7{семь}: shown as written, aligned as said, no reveal
        pieces = [text]
    elif '|' in raw:                              # за|еба|ли: the user's own steps; рокстар| = whole at once
        pieces = [on_screen(p) for p in re.sub(r'\{.*?\}', '', raw).split('|') if on_screen(p)]
    else:
        pieces = syllables(text)
    src = say.group(1) if say else re.sub(r'\{.*?\}', '', raw)
    return dict(raw=raw, text=text, pieces=pieces, say=''.join(c for c in src.lower() if c.isalpha()),
                fixed=bool(say), sung=sung, line=line, sub=sub)


def parse(vid):
    """Cards in the order of videos/lyrics/<vid>.txt: [{words, at, strobe, line}]; at = song s of a title
    card (a «@s» line), None for sung ones. A word: {raw, text, pieces, say, fixed, sung, line, sub}."""
    cards, sung, nline = [], 0, 0
    for li, raw in enumerate(open(lyrics_file(vid), encoding='utf-8').read().splitlines()):
        raw = re.sub(r'\(.*?\)', ' ', raw).strip()            # (ad-libs) are not shown
        if not raw or raw.startswith('#'):
            continue
        mode = None
        m = re.match(r'\[(\w+)\]\s*', raw)                        # [script] …: the line's presentation (look tongue)
        if m:
            mode = m.group(1)
            if mode not in tongue.MODES:
                raise SystemExit(f'{vid}: line {li + 1}: no mode [{mode}], there are ' + ', '.join(tongue.MODES))
            raw = raw[m.end():]
        if raw.startswith('@'):
            for m in re.finditer(r'@([\d.]+)\s+(\S+)', raw):
                ws = [word(p, None, li, '') for p in m.group(2).split('_')]
                for w in ws:
                    w.update(orig=in_sub(w['raw']), before='')
                cards.append(dict(words=ws, at=float(m.group(1)), strobe=m.group(2).endswith('*'), line=li,
                                  mode=mode, nline=nline))
            continue
        line_cards, shown = [], []
        for tok in raw.split():
            ws = []
            for p in tok.split('_'):
                shown.append(in_sub(p))
                if not re.search(r'\w', on_screen(p)):          # a lone dash
                    continue
                ws.append(word(p, sung, li, ' '.join(shown)))
                ws[-1].update(orig=in_sub(p), before=' '.join(shown[:-1]))
                sung += 1
            if ws:
                line_cards.append(dict(words=ws, at=None, strobe=tok.endswith('*'), line=li, mode=mode, nline=nline))
        k = 0
        while k < len(line_cards) - 1:                         # «в дребезги», «с этим»: one card
            c = line_cards[k]
            if len(c['words']) == 1 and c['words'][0]['say'] in PREP and not c['strobe']:
                line_cards[k + 1]['words'][:0] = c['words']
                del line_cards[k]
            else:
                k += 1
        cards += line_cards
        nline += 1
    return cards


def sung_words(cards):
    return [w for c in cards if c['at'] is None for w in c['words']]


def text_hash(cards):
    return hashlib.md5('\n'.join(w['say'] for w in sung_words(cards)).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- timing
def read_mono(path, t0, dur, sr):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{max(t0, 0):.4f}', '-t', f'{dur:.4f}', '-i', path,
                        '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32)


_ON = {}


def vocal_onsets(track, t0, t1):
    """Syllable onsets in the vocal stem between t0 and t1: peaks of the spectral flux at 300-4000 Hz,
    [(song s, strength 0…1)]."""
    key = (track, round(t0, 3), round(t1, 3))
    if key not in _ON:
        sr, n, hop = 22050, 1024, 110
        t0 = max(t0, 0)
        y = read_mono(track_file(track, 'stems', 'vocals.wav'), t0, t1 - t0, sr)
        fr = np.lib.stride_tricks.sliding_window_view(y, n)[::hop] * np.hanning(n).astype(np.float32)
        f = np.fft.rfftfreq(n, 1 / sr)
        mag = np.abs(np.fft.rfft(fr, axis=1))[:, (f > 300) & (f < 4000)]
        lm = np.log1p(40 * mag)
        fl = np.r_[0, np.maximum(np.diff(lm, axis=0), 0).sum(1)]
        fl /= max(np.percentile(fl, 99.5), 1e-9)
        gap = int(0.07 * sr / hop)
        pk = [i for i in range(1, len(fl) - 1) if fl[i] > 0.18 and fl[i] == fl[max(0, i - gap):i + gap + 1].max()]
        _ON[key] = [(t0 + (i * hop + n / 2) / sr, float(min(fl[i], 1))) for i in pk]
    return _ON[key]


def load_align(vid, cards):
    p = align_file(vid)
    if not os.path.exists(p):
        raise SystemExit(f'{vid}: no videos/align/{vid}.json - run ./snip align {vid}')
    al = json.load(open(p, encoding='utf-8'))
    if al.get('text') != text_hash(cards):
        raise SystemExit(f'{vid}: the lyrics changed after ./snip align - run it again')
    return al['words']


def word_times(vid, cards):
    """{sung index: [song s of each piece]}: MMS_FA letter times (first letter of the piece), moved to the
    nearest vocal onset within SNAP, then the spec's "onsets" ("12" = word 12, "12.1" = its 2nd piece)."""
    v = VIDEOS[vid]
    words = sung_words(cards)
    al = load_align(vid, cards)
    peaks = [p for p in vocal_onsets(v['track'], v['start'] - 0.5, v['end'] + 0.5) if p[1] >= 0.2]
    pt = np.array([p[0] for p in peaks])
    starts = [a['s'] for a in al]
    known = [i for i, s in enumerate(starts) if s is not None]
    res = {}
    for i, (w, a) in enumerate(zip(words, al)):
        n = len(w['pieces'])
        if a['s'] is None:                                     # no letters to align: between the neighbours
            lo = max([k for k in known if k < i], default=None)
            hi = min([k for k in known if k > i], default=None)
            s = (starts[lo] + starts[hi]) / 2 if lo is not None and hi is not None else \
                (starts[lo] + 0.3 if lo is not None else starts[hi] - 0.3)
            res[w['sung']] = [s + 0.12 * j for j in range(n)]
            continue
        ts, k = [], 0
        lt = a['letters']
        for p in w['pieces']:
            m = sum(1 for c in p.lower() if c.isalpha())
            got = [s for _, s in lt[k:k + m] if s is not None] if not w['fixed'] else []
            ts.append(got[0] if got else None)
            k += m
        ts[0] = a['s'] if ts[0] is None else ts[0]
        for j in range(1, n):                                   # pieces with no aligned letter: spread evenly
            if ts[j] is None:
                nxt = next((ts[q] for q in range(j + 1, n) if ts[q] is not None), a['e'])
                ts[j] = ts[j - 1] + (nxt - ts[j - 1]) / 2
        out = []
        for t in ts:
            near = pt[(pt >= t - SNAP[0]) & (pt <= t + SNAP[1])] if len(pt) else []
            if len(near):
                t = float(near[np.argmin(np.abs(near - t))])
            out.append(t)
        res[w['sung']] = out
    for key, t in v.get('onsets', {}).items():
        wi, _, pi = str(key).partition('.')
        if int(wi) in res:
            res[int(wi)][int(pi or 0)] = float(t)
    order = [w['sung'] for w in words]
    last = -1e9
    for wi in order:                                            # never backwards
        ts = res[wi]
        for j in range(len(ts)):
            ts[j] = last = max(ts[j], last + 0.03)
    return res


def cards_timed(vid):
    """Cards with their reveal steps in song time, in screen order: [{..., steps: [(t, text, sub)]}]."""
    cards = parse(vid)
    times = word_times(vid, cards) if sung_words(cards) else {}
    out = []
    for c in cards:
        if c['at'] is not None:
            out.append(dict(c, steps=[(c['at'], ' '.join(w['text'] for w in c['words']), '')]))
            continue
        steps, shown = [], ''
        for w in c['words']:
            pre, acc = (shown + ' ' if shown else ''), ''
            for p, t in zip(w['pieces'], times[w['sung']]):
                acc += p
                steps.append((t - LEAD, pre + acc, w['sub']))
            shown = pre + acc
        out.append(dict(c, steps=steps))
    out.sort(key=lambda c: c['steps'][0][0])
    k = 0
    while k < len(out) - 1:                                     # too short to read: joins the next card
        a, b = out[k], out[k + 1]
        if a['at'] is None and b['at'] is None and b['steps'][0][0] - a['steps'][0][0] < MIN_CARD:
            last = a['steps'][-1][1]
            a['steps'] = a['steps'] + [(t, last + ' ' + s, sub) for t, s, sub in b['steps']]
            a['words'], a['strobe'] = a['words'] + b['words'], b['strobe']
            del out[k + 1]
        else:
            k += 1
    for k, c in enumerate(out):
        end = out[k + 1]['steps'][0][0] if k + 1 < len(out) else 1e9
        st = [s for s in c['steps'] if s[0] < end - 0.01] or c['steps'][:1]
        if len(st) < len(c['steps']):                           # the word ran into the next card: show it whole
            st[-1] = (st[-1][0],) + c['steps'][-1][1:]
        i = 0
        while i < len(st) - 1:                                  # a step too short to see: show the next one early
            if st[i + 1][0] - st[i][0] < MIN_STEP:
                st[i] = (st[i][0],) + st[i + 1][1:]
                del st[i + 1]
            else:
                i += 1
        if len(st) > 1 and end - st[-1][0] < MIN_STEP:
            st[-2] = (st[-2][0],) + st[-1][1:]
            del st[-1]
        c['steps'] = st
    return out


def beats(track, t0, t1):
    """Beat times in [t0, t1] from track.json "bpm" + "downbeat" (song s of a bar start), else every 0.25 s."""
    m = TRACKS[track]
    if m.get('bpm'):
        b, d = 60 / m['bpm'], m.get('downbeat', 0.0)
        return [d + b * n for n in range(int(np.ceil((t0 - d) / b)), int(np.floor((t1 - d) / b)) + 1)]
    return list(np.arange(t0, t1, 0.25))


def cut_letters(text, n):
    """text up to its n-th letter (whole, with the punctuation after it, when n covers every letter)."""
    if n >= sum(c.isalnum() for c in text):
        return text
    k = 0
    for i, c in enumerate(text):
        k += c.isalnum()
        if k == n:
            return text[:i + 1]
    return text


def segments(vid):
    """The video as gapless [[a, b, (text, bg, fg, sub)]] in video time, 0 … duration; the look «tongue»
    adds a fifth element (mode, small line, script word, card no, card start, on accent, previous card)."""
    v = VIDEOS[vid]
    st = style(vid)
    if st.get('look') == 'tongue':
        return segments_tongue(vid, v, st)
    t0, dur = v['start'], v['end'] - v['start']
    cs = cards_timed(vid)
    acc, dark = tuple(st['accent']), tuple(st['dark'])
    first_acc = st.get('first', 'accent') == 'accent'
    ev = []
    for k, c in enumerate(cs):
        on_acc = (k % 2 == 0) == first_acc
        end = cs[k + 1]['steps'][0][0] if k + 1 < len(cs) else t0 + dur
        ev += [(t, text, on_acc, sub) for t, text, sub in c['steps']]
        if c['strobe']:
            t_last, text, sub = c['steps'][-1]
            for j, b in enumerate(beats(v['track'], t_last + 0.1, end - 0.1)):
                ev.append((b - LEAD, text, on_acc == (j % 2 == 1), sub))
    ev.sort(key=lambda e: e[0])
    segs = []
    for i, (t, text, on_acc, sub) in enumerate(ev):
        a = max(t - t0, 0.0)
        b = min(ev[i + 1][0] - t0, dur) if i + 1 < len(ev) else dur
        if b <= a:
            continue
        bg, fg = (acc, dark) if on_acc else (dark, 'grad' if st.get('grad') else acc)
        state = (text, bg, fg, sub if st.get('subs') else '')
        if segs and segs[-1][2] == state:
            segs[-1][1] = b
        else:
            segs.append([a, b, state])
    if not segs:
        return [[0.0, dur, ('', dark, acc, '')]]
    if segs[0][0] > 0:
        segs.insert(0, [0.0, segs[0][0], ('', segs[0][2][1], segs[0][2][2], '')])
    return segs


def segments_tongue(vid, v, st):
    t0, dur = v['start'], v['end'] - v['start']
    cs = cards_timed(vid)
    acc, dark = tuple(st['accent']), tuple(st['dark'])
    first_acc = st.get('first', 'accent') == 'accent'
    modes = st.get('modes', ['big'])
    ev, prev = [], None
    for k, c in enumerate(cs):
        on_acc = (k % 2 == 0) == first_acc
        mode = c.get('mode') or modes[c.get('nline', 0) % len(modes)]
        orig = ' '.join(w['orig'] for w in c['words'])
        before = c['words'][0]['before']
        t_card = c['steps'][0][0] - t0
        last = None
        for t, text, _ in c['steps']:
            n = sum(ch.isalnum() for ch in text)
            part = cut_letters(orig, n)
            small = (before + ' ' + part).strip()
            word_only = re.sub(r'[,;:«»"“”„]', '', part).lower().strip()
            ev.append((t, text, on_acc, (mode, small, word_only, k, t_card, on_acc, prev), c['line']))
            last = (text, on_acc, mode, small, word_only)
        if c['strobe']:
            end = cs[k + 1]['steps'][0][0] if k + 1 < len(cs) else t0 + dur
            for j, b in enumerate(beats(v['track'], c['steps'][-1][0] + 0.1, end - 0.1)):
                ev.append((b - LEAD, last[0], on_acc == (j % 2 == 1), ev[-1][3][:5] + (on_acc == (j % 2 == 1), None),
                           c['line']))
        text, oa, mode, small, word_only = last
        prev = (text, acc if oa else dark, dark if oa else acc, mode, small, word_only, k, oa, c['line'])
    full = {}                                                   # the whole sung line: sizes the strip text once
    for e in sorted(ev, key=lambda e: e[0]):
        full[e[4]] = e[3][1]
    ev = [(t, text, on_acc, x[:6] + ((x[6][:8] + (full.get(x[6][8], ''),)) if x[6] else None, full.get(li, '')))
          for t, text, on_acc, x, li in ev]
    ev.sort(key=lambda e: e[0])
    segs = []
    for i, (t, text, on_acc, x) in enumerate(ev):
        a = max(t - t0, 0.0)
        b = min(ev[i + 1][0] - t0, dur) if i + 1 < len(ev) else dur
        if b <= a:
            continue
        bg, fg = (acc, dark) if on_acc else (dark, acc)
        state = (text, bg, fg, '', x)
        if segs and segs[-1][2] == state:
            segs[-1][1] = b
        else:
            segs.append([a, b, state])
    if not segs:
        return [[0.0, dur, ('', dark, acc, '', ('big', '', '', 0, 0.0, False, None, ''))]]
    if segs[0][0] > 0:
        s = segs[0][2]
        segs.insert(0, [0.0, segs[0][0], ('', s[1], s[2], '', ('big', '', '', -1, 0.0, s[4][5], None, ''))])
    return segs


def make_type(vid, W=None, H=None):
    if W is None:
        W, H = size(vid)
    st = style(vid)
    return tongue.TongueType(st, W, H) if st.get('look') == 'tongue' else Type(st, W, H)


# ---------------------------------------------------------------- drawing
class Type:
    """The stretched grotesque: cap height = cap × frame height, centred; a word as wide as
    "width" × cap for the reference word («НАПЛЕВАТЬ» in references/r1); no wider than maxw × frame."""

    def __init__(self, st, W, H):
        self.W, self.H = W, H
        path = os.path.join(ROOT, st['font'])

        def face(px):
            f = ImageFont.truetype(path, px)
            if st.get('weight'):
                f.set_variation_by_axes([st['weight']])
            return f

        probe = face(200)
        cap200 = -probe.getbbox('Н', anchor='ls')[1]
        self.font = face(max(8, round(200 * st['cap'] * H / cap200)))
        cap = -self.font.getbbox('Н', anchor='ls')[1]
        self.base = round((H + cap) / 2)
        self.sq = st['width'] * cap / self.font.getlength(st.get('width_word', 'НАПЛЕВАТЬ'))
        self.maxw = round(st.get('maxw', 0.94) * W)
        self.grad = st.get('grad')
        self.cache = {}
        self.sub_font = ImageFont.truetype(st['sub_font'], round(st.get('sub_size', 0.06) * min(W, H)),
                                           index=st.get('sub_index', 0)) if st.get('sub_font') else None

    def mask(self, text):
        """Letters as an L image of the frame height; drawn at full height, then squeezed sideways."""
        m = self.cache.get(text)
        if m is None:
            l, _, r, _ = self.font.getbbox(text, anchor='ls')
            im = Image.new('L', (r - l + 16, self.H), 0)
            ImageDraw.Draw(im).text((8 - l, self.base), text, font=self.font, fill=255, anchor='ls')
            m = self.cache[text] = im.resize((min(self.maxw, max(1, round(im.width * self.sq))), self.H), Image.BOX)
        return m

    def squeezed(self, text):
        """True when the text had to be squeezed more than the style to fit the frame."""
        l, _, r, _ = self.font.getbbox(text, anchor='ls')
        return (r - l + 16) * self.sq > self.maxw

    def sub(self, im, text):
        f, d = self.sub_font, ImageDraw.Draw(im)
        lines, cur = [], ''
        for w in text.split():
            t = (cur + ' ' + w).strip()
            if cur and f.getlength(t) > 0.52 * self.W:
                lines.append(cur)
                cur = w
            else:
                cur = t
        lines.append(cur)
        lh = f.size * 1.15
        y = self.H / 2 - lh * (len(lines) - 1) / 2
        for ln in lines:
            d.text((self.W / 2, y), ln, font=f, fill=(255, 255, 255), anchor='mm')
            y += lh


def compose(T, state, t=None, shade=0):
    if len(state) > 4:
        return tongue.compose(T, state, t, shade)
    text, bg, fg, sub = state
    im = Image.new('RGB', (T.W, T.H), bg)
    if text:
        m = T.mask(text)
        if fg == 'grad':                          # letters on the dark cards: colour stops left to right
            st = np.array(T.grad, np.float32)
            x = np.linspace(0, len(st) - 1, m.width)
            i = np.minimum(x.astype(int), len(st) - 2)
            row = st[i] + (st[i + 1] - st[i]) * (x - i)[:, None]
            fill = Image.fromarray(np.repeat(row[None].round().astype(np.uint8), T.H, 0))
            im.paste(fill, ((T.W - m.width) // 2, 0), m)
        else:
            im.paste(fg, ((T.W - m.width) // 2, 0), m)
    if sub and T.sub_font:
        T.sub(im, sub)
    return im


def state_at(segs, t):
    for a, b, s in segs:
        if a <= t < b:
            return s
    return segs[-1][2]


def still(vid, t):
    return compose(make_type(vid), state_at(segments(vid), t), t)


def sheet(vid, cols=6, tw=320):
    """Every screen state in order with its video time: the contact sheet of the whole video."""
    W, H = size(vid)
    T = make_type(vid)
    segs = segments(vid)
    th = round(tw * H / W)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    rows = (len(segs) + cols - 1) // cols
    out = Image.new('RGB', (cols * (tw + 4), rows * (th + 4)), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for i, (a, b, s) in enumerate(segs):
        x, y = (i % cols) * (tw + 4), (i // cols) * (th + 4)
        out.paste(compose(T, s).resize((tw, th), Image.BILINEAR), (x, y))
        d.rectangle((x, y, x + 92, y + 18), fill=(255, 255, 255))
        d.text((x + 3, y + 1), f'{a:6.2f} {b - a:.2f}', fill=(0, 0, 0), font=font)
    return out


# ---------------------------------------------------------------- sound + video
STEM = {'mix': 'source.wav', 'inst': 'stems/no_vocals.wav', 'voc': 'stems/vocals.wav'}


def read_pcm(path, t0, n):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.4f}', '-t', f'{n / SR + 0.1:.4f}', '-i', path,
                        '-ar', str(SR), '-ac', '2', '-f', 'f32le', '-'], capture_output=True, check=True)
    a = np.frombuffer(r.stdout, np.float32).reshape(-1, 2)
    out = np.zeros((n, 2), np.float32)
    out[:min(n, len(a))] = a[:n]
    return out


def premix(vid, dur):
    """cache/premix/<vid>.wav, exactly dur s: the song from "start", switching between the mix and the
    Demucs stems at the times in "audio" (30 ms crossfades; stems are matched to the mix level on this
    stretch), "fade" s fade-out at the end (short by default: the video loops)."""
    spec = VIDEOS[vid]
    track, t0 = spec['track'], spec['start']
    pieces = spec.get('audio') or [[t0, 'mix']]
    n = int(round(dur * SR))
    need = {p[1] for p in pieces}
    if need - {'mix'}:
        need |= {'mix', 'inst', 'voc'}
    pcm = {k: read_pcm(track_file(track, STEM[k]), t0, n) for k in need}
    if 'inst' in pcm:
        s = pcm['inst'] + pcm['voc']
        g = float((pcm['mix'] * s).sum() / max(float((s * s).sum()), 1e-9))
        pcm['inst'] *= g
        pcm['voc'] *= g
    out = np.zeros((n, 2), np.float32)
    xf = int(0.03 * SR)
    for k, (ts, name) in enumerate(pieces):
        a = int(round((ts - t0) * SR))
        b = int(round((pieces[k + 1][0] - t0) * SR)) if k + 1 < len(pieces) else n
        env = np.zeros(n, np.float32)
        env[a:b] = 1
        if k:
            env[a:a + xf] = np.linspace(0, 1, len(env[a:a + xf]))
        if k + 1 < len(pieces):
            env[b:b + xf] = np.linspace(1, 0, len(env[b:b + xf]))
        out += pcm[name] * env[:, None]
    i0 = int(0.005 * SR)
    out[:i0] *= np.linspace(0, 1, i0)[:, None]
    f = int(float(spec.get('fade', 0.03)) * SR)
    out[n - f:] *= np.linspace(1, 0, f)[:, None]
    peak = float(np.abs(out).max())
    if peak > 0.98:
        out *= 0.98 / peak
    os.makedirs(os.path.join(CACHE, 'premix'), exist_ok=True)
    wav = os.path.join(CACHE, 'premix', f'{vid}.wav')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', '-',
                    '-c:a', 'pcm_f32le', wav], input=out.tobytes(), check=True)
    return wav


def out_name(vid, ext):
    """output/<title>/<title> - dynamic <n> <ext>; n is the vid's trailing number."""
    title = TRACKS[VIDEOS[vid]['track']]['title']
    n = re.search(r'(\d+)$', vid).group(1)
    os.makedirs(os.path.join(OUT, title), exist_ok=True)
    return os.path.join(OUT, title, f'{title} - dynamic {n} {ext}')


def video(vid, draft=False):
    """The video with sound: output/<title>/… (or cache/drafts/<vid>.mp4 at half size, 30 fps)."""
    v = VIDEOS[vid]
    W, H = size(vid)
    fps = int(v.get('fps', 60))
    if draft:
        W, H, fps = W // 2, H // 2, 30
        os.makedirs(os.path.join(CACHE, 'drafts'), exist_ok=True)
        out = os.path.join(CACHE, 'drafts', f'{vid}.mp4')
    else:
        out = out_name(vid, f"{v.get('size', '16x9')}.mp4")
    segs = segments(vid)
    n = int(round((v['end'] - v['start']) * fps))
    T = make_type(vid, W, H)
    live = isinstance(T, tongue.TongueType)                     # flicker and motion: every frame is drawn
    flick = max(1, round(fps / float(style(vid).get('flicker_hz', 30))))
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                            '-r', str(fps), '-i', '-', '-i', premix(vid, n / fps), '-map', '0:v', '-map', '1:a',
                            '-vf', 'scale=out_color_matrix=bt709:out_range=tv,format=yuv420p,'
                                   'setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv',
                            '-c:v', 'libx264', '-preset', 'veryfast' if draft else 'slow', '-crf', '16',
                            '-tune', 'animation', '-c:a', 'aac', '-b:a', '256k', '-movflags', '+faststart', out],
                           stdin=subprocess.PIPE)
    k, cur, buf = 0, None, None
    for i in range(n):
        t = i / fps
        while k + 1 < len(segs) and t >= segs[k][1]:
            k += 1
        if live:
            buf = compose(T, segs[k][2], t, (i // flick) % 2).tobytes()
        elif segs[k][2] is not cur:
            cur = segs[k][2]
            buf = compose(T, cur).tobytes()
        enc.stdin.write(buf)
    enc.stdin.close()
    if enc.wait():
        raise SystemExit('ffmpeg failed')
    return out


if __name__ == '__main__':
    load()
    mode, vid = sys.argv[1], sys.argv[2]
    opt = dict(a[2:].split('=', 1) for a in sys.argv[3:] if a.startswith('--') and '=' in a)
    if mode == 'still':
        os.makedirs(os.path.join(CACHE, 'stills'), exist_ok=True)
        p = opt.get('out', os.path.join(CACHE, 'stills', f"{vid}_{float(opt.get('t', 8)):.2f}.png"))
        still(vid, float(opt.get('t', 8))).save(p)
        print(p)
    elif mode == 'plan':
        for a, b, (text, bg, fg, sub, *_) in segments(vid):
            print(f'{a:7.3f} {b - a:5.2f}  {"█" if bg != (0, 0, 0) else "·"}  {text}' + (f'   | {sub}' if sub else ''))
    else:
        print(video(vid, draft='--draft' in sys.argv))
