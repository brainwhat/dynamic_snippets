"""dynamic_snippets command line. Run through ./snip (uv with numpy + pillow); see README.md.

  ./snip list                                   videos with track, part, times, style, status
  ./snip tracks                                 tracks, sections, which are used
  ./snip styles                                 colour styles from styles/styles.json
  ./snip ref <file in references/>              measure a reference video and print a style for styles.json
  ./snip import-track <id>                      take a track from ../lyrics_snippets/tracks/<id> (no Demucs, no Whisper)
  ./snip add-track <audio> <id> --title=… [--artist=…] [--lang=ru]   new track: wav, Demucs stems, GigaAM words (Whisper if not ru)
  ./snip map <track> [--from=s] [--to=s]        words.json lines + vocal level every 0.5 s + pauses
  ./snip edge <track> <s>                       vocal level every 0.1 s around s, to place a cut
  ./snip bars <track> [--from=s] [--to=s]       bar starts from "bpm" + "downbeat" in track.json
  ./snip new <vid> <track> <s0> <s1> [--style=navy] [--size=16x9] [--part=…]   (or --bars=0-16 instead of s0 s1)
  ./snip align <vid>                            MMS_FA letter times for the lyrics → videos/align/<vid>.json
  ./snip plan <vid>                             every screen state: time, length, colour, text
  ./snip check <vid>                            files, lyrics, alignment, short cards, squeezed words
  ./snip still <vid> --t=<s>                    one frame (video time) → cache/stills/
  ./snip sheet <vid>                            all screen states on one picture → cache/stills/<vid>_sheet.jpg
  ./snip render <vid>… [--draft]                video + cover + phone copy into output/<title>/ (--draft: cache/drafts, half size)
  ./snip index                                  rewrite INDEX.md (tracks, videos, styles) after changes
"""
import contextlib, io, json, os, re, shutil, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kinetic as K

ROOT = K.ROOT
LYRICS_SNIPPETS = os.path.join(os.path.dirname(ROOT), 'lyrics_snippets')
WHISPER_PY = '/Users/vasilijfrolov/Desktop/Реп/Конструктор снипеттов/.venv/bin/python'
# Russian words.json: GigaAM-v3 e2e CTC on MLX (more accurate than Whisper on rap and ~20x faster, see README)
GIGAAM = ['uv', 'run', '--quiet', '--python', '3.12', '--with',
          'gigaam-mlx @ git+https://github.com/aystream/gigaam-mlx@20276ddd6173d636b37c6c6e13b4ee8f7b94d1ac',
          'python', '-I', os.path.join(ROOT, 'engine', 'transcribe_gigaam.py')]
DEMUCS = ['uvx', '--python', '3.11', '--from', 'demucs==4.1.0', '--with', 'torchaudio<2.9', '--with', 'soundfile',
          'demucs', '-n', 'htdemucs', '--two-stems=vocals', '-d', 'mps']
ALIGN = ['uv', 'run', '--quiet', '--python', '3.11', '--with', 'torch==2.8.0', '--with', 'torchaudio==2.8.0', '--with', 'numpy',
         'python', '-I', os.path.join(ROOT, 'engine', 'align.py')]
SPECS = os.path.join(K.VIDEOS_DIR, 'videos.json')
MAX_LEN = 20.0                     # the user's rule (2026-11-04): no video longer than 20 s, 2-3+ videos per track


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def duration(path):
    return float(sh(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                    capture_output=True, text=True).stdout)


def load_json(p, default=None):
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else default


def save_json(p, d):
    json.dump(d, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


_DB = {}


def vocal_db(track):
    """Vocal-stem level in dB, 100 values per second of the song."""
    if track not in _DB:
        v = K.read_mono(K.track_file(track, 'stems', 'vocals.wav'), 0, 1e5, 8000)
        v = v[:len(v) // 80 * 80].reshape(-1, 80)
        _DB[track] = 10 * np.log10((v ** 2).mean(1) + 1e-12)
    return _DB[track]


def level(track, step):
    db = vocal_db(track)
    k = int(round(step * 100))
    n = len(db) // k * k
    return 10 * np.log10((10 ** (db[:n].reshape(-1, k) / 10)).mean(1))


def bar_of(track, t):
    m = K.TRACKS[track]
    if not m.get('bpm'):
        return ''
    b = (t - m.get('downbeat', 0)) / (240 / m['bpm'])
    return f'такт {int(np.floor(b))}, доля {(b - np.floor(b)) * 4 + 1:.1f}'


# ---------------------------------------------------------------- tracks
def cmd_import_track(tid):
    src = os.path.join(LYRICS_SNIPPETS, 'tracks', tid)
    if not os.path.isdir(src):
        sys.exit(f'no {src}')
    dst = K.track_file(tid)
    os.makedirs(os.path.join(dst, 'stems'), exist_ok=True)
    for f in ('source.wav', 'stems/vocals.wav', 'stems/no_vocals.wav', 'words.json'):
        if os.path.exists(os.path.join(src, f)):
            sh(['cp', '-c', os.path.join(src, f), os.path.join(dst, f)])     # APFS clone: no extra disk space
    m = load_json(os.path.join(src, 'track.json'))
    m['source'] = f'lyrics_snippets/tracks/{tid} ({m.get("source", "")})'
    m['videos'] = []
    m.setdefault('bpm', None)
    m.setdefault('downbeat', None)
    save_json(K.track_file(tid, 'track.json'), m)
    print(f'tracks/{tid}: source.wav, stems/, words.json, track.json - check "sections", "bpm", "downbeat"')


def cmd_add_track(audio, tid, title, artist='', lang='ru'):
    d = K.track_file(tid)
    os.makedirs(os.path.join(d, 'stems'), exist_ok=True)
    src = os.path.join(d, 'source.wav')
    sh(['ffmpeg', '-v', 'error', '-y', '-i', audio, '-ar', '44100', '-c:a', 'pcm_s16le', src])
    tmp = os.path.join(K.CACHE, 'demucs')
    sh(DEMUCS + ['-o', tmp, src])
    for stem in ('vocals.wav', 'no_vocals.wav'):
        os.replace(os.path.join(tmp, 'htdemucs', 'source', stem), os.path.join(d, 'stems', stem))
    vocals, words = os.path.join(d, 'stems', 'vocals.wav'), os.path.join(d, 'words.json')
    if lang == 'ru':
        sh(GIGAAM + [vocals, words])
    else:                                          # GigaAM knows only Russian
        sh([WHISPER_PY, os.path.join(ROOT, 'engine', 'transcribe.py'), vocals, lang, words])
    save_json(os.path.join(d, 'track.json'),
              dict(title=title, artist=artist, lang=lang, bpm=None, downbeat=None, source=audio,
                   duration=round(duration(src), 2), sections=[], notes='', videos=[]))
    print(f'tracks/{tid}: source.wav, stems/, words.json, track.json - now fill "sections" from ./snip map {tid}')


def cmd_tracks():
    for tid in sorted(K.TRACKS):
        m = K.TRACKS[tid]
        used = [v for v, s in K.VIDEOS.items() if s['track'] == tid]
        print(f"\n{tid}  «{m['title']}»  {m.get('artist', '')}  {m.get('duration', 0):.0f} с"
              + (f"  {m['bpm']} BPM, такт 0 = {m.get('downbeat')} с" if m.get('bpm') else ''))
        for s in m.get('sections', []):
            hit = [v for v in used if K.VIDEOS[v]['start'] < s['end'] - 1 and K.VIDEOS[v]['end'] > s['start'] + 1]
            print(f"   {s['start']:7.2f}-{s['end']:7.2f}  {s['name']:<32} {', '.join(hit) if hit else '- свободно'}")
        if m.get('notes'):
            print('   ' + m['notes'])


def cmd_map(track, frm=None, to=None):
    frm = float(frm) if frm is not None else 0
    to = float(to) if to is not None else 1e9
    print('words.json (vocal stem):')
    for seg in load_json(K.track_file(track, 'words.json'), []):
        if frm <= seg['s'] <= to:
            print(f"  {seg['s']:7.2f}-{seg['e']:7.2f}  {seg['text']}")
    lv = level(track, 0.5)
    print('\nvocal level dB every 0.5 s (below -45 = silence):')
    rows = [f'{i * 0.5:6.1f}:{v:4.0f}' for i, v in enumerate(lv) if frm <= i * 0.5 <= to]
    for k in range(0, len(rows), 12):
        print('  ' + '  '.join(rows[k:k + 12]))
    quiet = vocal_db(track) < -45
    print('\npauses ≥ 0.3 s:')
    i = 0
    while i < len(quiet):
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        if j - i >= 30 and frm <= i / 100 <= to:
            print(f'  {i / 100:7.2f}-{j / 100:7.2f}  {bar_of(track, i / 100)}')
        i = max(j, i + 1)


def cmd_edge(track, t):
    t = float(t)
    lv = level(track, 0.1)
    rows = [f'{i / 10:7.2f}:{lv[i]:4.0f}' for i in range(max(0, int((t - 1.2) * 10)), min(len(lv), int((t + 1.2) * 10)))]
    for k in range(0, len(rows), 12):
        print('  '.join(rows[k:k + 12]))


def grid(track):
    m = K.TRACKS[track]
    if not m.get('bpm') or m.get('downbeat') is None:
        sys.exit(f'tracks/{track}/track.json has no "bpm" + "downbeat"')
    return m['downbeat'], 240 / m['bpm']


def cmd_bars(track, frm=0, to=1e9):
    d, bar = grid(track)
    n0 = int(np.ceil((float(frm) - d) / bar))
    n1 = int(np.floor((min(float(to), K.TRACKS[track].get('duration', 600)) - d) / bar))
    for n in range(n0, n1 + 1):
        print(f'такт {n:3d}  {d + n * bar:8.3f}')


def cmd_retime(track, s0, s1, txt):
    sh([WHISPER_PY, os.path.join(ROOT, 'engine', 'window_words.py'), track, s0, s1, txt])


# ---------------------------------------------------------------- videos
def cmd_list():
    for vid, v in K.VIDEOS.items():
        print(f"{vid:<12} {v['track']:<11} {v.get('part', ''):<24} {v['start']:7.2f}-{v['end']:7.2f} "
              f"{v['end'] - v['start']:5.1f} с  {v.get('style', 'navy'):<7} {v.get('size', '16x9'):<5} {v['status']}")


def cmd_styles():
    for name, s in K.STYLES.items():
        if not name.startswith('_'):
            print(f"{name:<8} {str(s.get('accent')):<16} первая: {s.get('first', 'accent'):<7} "
                  f"{'подписи' if s.get('subs') else '':<8} {s.get('note', '')}")


def cmd_ref(path):
    """Measure a reference video: size, colours, letter height, cards per second, italic line → a style."""
    if not os.path.exists(path):
        hits = [f for f in os.listdir(os.path.join(ROOT, 'references')) if path in f and f.endswith('.mp4')]
        if len(hits) != 1:
            sys.exit(f'no single file in references/ matches «{path}»: {hits}')
        path = os.path.join(ROOT, 'references', hits[0])
    pr = sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate',
             '-of', 'csv=p=0', path], capture_output=True, text=True).stdout.strip().split(',')
    w, h = 320, 180
    raw = sh(['ffmpeg', '-v', 'error', '-i', path, '-vf', f'fps=30,scale={w}:{h}:flags=area,format=rgb24',
              '-f', 'rawvideo', '-'], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).astype(int)
    bg = fr[:, 1, 1]
    dark = bg.sum(1) < 60
    acc = np.median(bg[~dark], 0).astype(int).tolist() if (~dark).any() else None
    tops, bots, white = [], [], 0
    for f, b in zip(fr[::3], bg[::3]):
        m = np.abs(f - b).sum(2) > 90
        rows = np.where(m.mean(1) > 0.01)[0]
        if len(rows):
            tops.append(rows[0])
            bots.append(rows[-1])
        wt = f.min(2) > 200
        white += wt[int(h * 0.4):int(h * 0.6)].mean() > 0.003 and not wt[:int(h * 0.3)].any()
    n = len(fr)
    changes = sum(1 for i in range(1, n) if np.abs(fr[i] - fr[i - 1]).mean() > 2)
    flips = sum(1 for i in range(1, n) if dark[i] != dark[i - 1])
    cap = (np.median(bots) - np.median(tops) + 1) / h if tops else 0
    subs = white > 0.3 * len(fr[::3])
    print(f'{os.path.basename(path)}\n  {pr[0]}×{pr[1]}, {pr[2]} fps, {n / 30:.1f} s'
          f'\n  фон: {acc} и чёрный, первый кадр {"чёрный" if dark[0] else "цветной"}; смен фона {flips / (n / 30):.1f}/с, '
          f'смен картинки {changes / (n / 30):.1f}/с\n  буквы занимают {cap:.2f} высоты кадра'
          f'{", белая строка курсивом по центру" if subs else ""}')
    print('\nстиль для styles/styles.json:\n' + json.dumps(
        {'<имя>': dict(accent=acc, first='dark' if dark[0] else 'accent', subs=bool(subs),
                       note=f'референс {os.path.basename(path)[:40]}')}, ensure_ascii=False))


def cmd_new(vid, track, s0=None, s1=None, style='navy', size='16x9', part='', bars=None):
    specs = load_json(SPECS, {})
    if vid in specs:
        sys.exit(f'{vid} already exists in videos/videos.json')
    if not re.search(r'\d+$', vid):
        sys.exit('vid must end in a number: it becomes «dynamic N» in the file name')
    if style not in K.STYLES:
        sys.exit(f'no style {style}: ./snip styles')
    if bars:
        d, bar = grid(track)
        b0, b1 = (int(x) for x in bars.split('-'))
        s0, s1 = round(d + b0 * bar, 4), round(d + b1 * bar, 4)
    s0, s1 = float(s0), float(s1)
    specs[vid] = dict(track=track, part=part, start=s0, end=s1, style=style, size=size, fps=60,
                      audio=[[s0, 'mix']], fade=0.03, onsets={}, status='draft')
    save_json(SPECS, specs)
    m = K.TRACKS[track]
    m['videos'] = sorted(set(m.get('videos', [])) | {vid})
    save_json(K.track_file(track, 'track.json'), m)
    lf = K.lyrics_file(vid)
    if not os.path.exists(lf):
        open(lf, 'w').close()
    print(f'videos/videos.json: {vid} added ({s0}-{s1}); write the lyrics into videos/lyrics/{vid}.txt, then ./snip align {vid}')
    if s1 - s0 > MAX_LEN + 0.01:
        print(f'WARNING: {s1 - s0:.1f} s, the user wants {MAX_LEN:.0f} s or shorter: cut it into several videos')


def cmd_align(vid):
    v = K.VIDEOS[vid]
    cards = K.parse(vid)
    words = K.sung_words(cards)
    os.makedirs(os.path.join(K.CACHE, 'align'), exist_ok=True)
    txt = os.path.join(K.CACHE, 'align', f'{vid}.words.txt')
    tmp = os.path.join(K.CACHE, 'align', f'{vid}.json')
    open(txt, 'w', encoding='utf-8').write('\n'.join(w['say'] for w in words) + '\n')
    sh(ALIGN + [K.track_file(v['track'], 'stems', 'vocals.wav'), str(max(0, v['start'] - 0.3)), str(v['end'] + 0.3),
                txt, tmp])
    al = load_json(tmp)
    os.makedirs(os.path.dirname(K.align_file(vid)), exist_ok=True)
    save_json(K.align_file(vid), dict(text=K.text_hash(cards), words=al))
    # MMS_FA has no useful confidence on sung vocals; a word far from Whisper's same word is worth a look
    wh = [(re.sub(r'[^a-zа-яё]', '', w['w'].lower()), w['s']) for seg in load_json(K.track_file(v['track'], 'words.json'), [])
          for w in seg['words']]
    odd = []
    for a in al:
        near = [s for t, s in wh if t == a['w'] and a['s'] is not None and abs(s - a['s']) < 1.5]
        if near and min(abs(s - a['s']) for s in near) > 0.35:
            odd.append(f"{a['w']} {a['s']:.2f} (Whisper {min(near, key=lambda s: abs(s - a['s'])):.2f})")
    print(f'videos/align/{vid}.json: {len(al)} words'
          + (f'\nfar from Whisper, check with ./snip plan and pin in "onsets" if wrong: ' + '; '.join(odd) if odd else ''))


def cmd_plan(vid):
    t0 = K.VIDEOS[vid]['start']
    acc = tuple(K.style(vid)['accent'])
    for a, b, (text, bg, fg, sub, *x) in K.segments(vid):
        if x:                                       # look tongue: [mode] and the small line
            sub = f'[{x[0][0]}] {x[0][1]}'
        print(f'{a:7.3f}  {t0 + a:7.3f}  {b - a:5.2f}  {"█" if bg == acc else "·"}  {text}'
              + (f'    | {sub}' if sub else ''))


def cmd_check(vid):
    v = K.VIDEOS[vid]
    bad = []
    for f in ('source.wav', 'stems/vocals.wav', 'stems/no_vocals.wav'):
        if not os.path.exists(K.track_file(v['track'], f)):
            bad.append(f'tracks/{v["track"]}/{f} missing')
    lf = K.lyrics_file(vid)
    text = open(lf, encoding='utf-8').read() if os.path.exists(lf) else ''
    if not text.strip():
        bad.append(f'videos/lyrics/{vid}.txt is empty')
    if '—' in text or '–' in text:
        bad.append('lyrics contain «—»: the user wants «-» (the engine prints «-» anyway)')
    a = v.get('audio') or []
    if not a or abs(a[0][0] - v['start']) > 0.01:
        bad.append('audio[0] must start at "start"')
    if v['end'] - v['start'] > MAX_LEN + 0.01:
        bad.append(f'{v["end"] - v["start"]:.1f} s long: the user wants every video {MAX_LEN:.0f} s or shorter (README, «Правила пользователя»)')
    if text.strip() and not bad:
        try:
            segs = K.segments(vid)
        except SystemExit as e:
            bad.append(str(e))
            segs = []
        W, H = K.size(vid)
        T = K.make_type(vid, W, H)
        for x, y, (t, *_rest) in segs:
            if y - x < 0.06 and x > 0:
                bad.append(f'{x:6.2f}: «{t}» on screen only {y - x:.3f} s')
        wide = sorted({s[2][0] for s in segs if s[2][0] and T.squeezed(s[2][0])})
        if wide:
            bad.append('squeezed to fit the frame: ' + ', '.join(wide))
        cards = K.cards_timed(vid)
        for c in cards:
            if c['steps'][0][0] < v['start'] - 0.01:
                bad.append(f"«{c['steps'][-1][1]}» starts before the video ({c['steps'][0][0]:.2f})")
            if c['steps'][0][0] > v['end']:
                bad.append(f"«{c['steps'][-1][1]}» starts after the video ({c['steps'][0][0]:.2f})")
    print('\n'.join(bad) if bad else f'{vid}: ok')


def cmd_still(vid, t):
    os.makedirs(os.path.join(K.CACHE, 'stills'), exist_ok=True)
    out = os.path.join(K.CACHE, 'stills', f'{vid}_{float(t):.2f}.png')
    K.still(vid, float(t)).save(out)
    print(out)


def cmd_sheet(vid):
    os.makedirs(os.path.join(K.CACHE, 'stills'), exist_ok=True)
    out = os.path.join(K.CACHE, 'stills', f'{vid}_sheet.jpg')
    K.sheet(vid).save(out, quality=88)
    print(out)


def cover(vid):
    """The title card if the video has one, else the state at "cover" (video s) or at a third."""
    v = K.VIDEOS[vid]
    segs = K.segments(vid)
    title = K.on_screen(K.TRACKS[v['track']]['title'])
    hit = [s for s in segs if s[2][0] == title]
    t = v.get('cover', (hit[0][0] + 0.01) if hit else (v['end'] - v['start']) / 3)
    name = K.out_name(vid, 'обложка.jpg')
    K.still(vid, t).save(name, quality=92)
    return name


def phone(vid):
    v = K.VIDEOS[vid]
    src = K.out_name(vid, f"{v.get('size', '16x9')}.mp4")
    d = os.path.join(os.path.dirname(src), 'phone')
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f'{vid}_phone.mp4')
    sh(['ffmpeg', '-v', 'error', '-y', '-i', src, '-c:v', 'libx264', '-preset', 'slow', '-crf', '20',
        '-maxrate', '8M', '-bufsize', '16M', '-tune', 'animation', '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart', dst])
    return dst


def cmd_render(*vids, draft=None):
    specs = load_json(SPECS)
    for vid in vids:
        print(K.video(vid, draft=bool(draft)))
        if draft:
            continue
        print(cover(vid))
        p = phone(vid)
        print(p, f'{os.path.getsize(p) / 1e6:.1f} MB')
        if specs[vid]['status'] == 'draft':
            specs[vid]['status'] = 'rendered'
    save_json(SPECS, specs)


def cmd_index():
    parts = []
    for title, fn in (('Треки (секции, что уже снято)', cmd_tracks), ('Видео', cmd_list), ('Стили', cmd_styles)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        parts.append(f'## {title}\n\n```\n{buf.getvalue().strip()}\n```\n')
    open(os.path.join(ROOT, 'INDEX.md'), 'w', encoding='utf-8').write(
        '# dynamic_snippets: что есть сейчас\n\nСгенерировано `./snip index`, руками не править.\n\n' + '\n'.join(parts))
    print('INDEX.md')


# ---------------------------------------------------------------- main
def main(argv):
    if not argv or argv[0] in ('-h', '--help', 'help'):
        print(__doc__)
        return
    K.load()
    cmd, pos = argv[0], [a for a in argv[1:] if not a.startswith('--')]
    opt = dict(a[2:].split('=', 1) if '=' in a else (a[2:], '1') for a in argv[1:] if a.startswith('--'))
    opt = {k.replace('-', '_'): v for k, v in opt.items()}
    if 'from' in opt:
        opt['frm'] = opt.pop('from')
    if cmd == 'still':
        return cmd_still(pos[0], opt['t'])
    fn = globals().get('cmd_' + cmd.replace('-', '_'))
    if not fn:
        sys.exit(f'unknown command {cmd}; ./snip --help')
    fn(*pos, **opt)


if __name__ == '__main__':
    main(sys.argv[1:])
