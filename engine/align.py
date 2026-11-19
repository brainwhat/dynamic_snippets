"""Forced alignment of a video's sung words to the vocal stem: torchaudio MMS_FA on romanised text.
python -I engine/align.py <vocals.wav> <t0> <t1> <words.txt> <out.json>

words.txt: one sung word per line, letters only (what is pronounced). out.json: one entry per word,
{w, s, e, p, letters: [[letter, s], ...]} in song seconds; a letter's time is where its first
romanised character starts (ь and ъ have none, so null). A word with no letters gets s = null.
Run through ./snip align <vid> (uv with torch / torchaudio 2.8, the model is cached in ~/.cache/torch).
"""
import json, subprocess, sys
import numpy as np
import torch, torchaudio
import torchaudio.functional as F

TR = dict(а='a', б='b', в='v', г='g', д='d', е='e', ё='yo', ж='zh', з='z', и='i', й='y', к='k', л='l', м='m', н='n',
          о='o', п='p', р='r', с='s', т='t', у='u', ф='f', х='h', ц='ts', ч='ch', ш='sh', щ='shch', ъ='', ы='y', ь='',
          э='e', ю='yu', я='ya')


def rom(c):
    return TR.get(c, c if c.isascii() and c.isalpha() else '')


wav, t0, t1, txt, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
words = [w.strip().lower() for w in open(txt, encoding='utf-8').read().splitlines()]
bundle = torchaudio.pipelines.MMS_FA
SR = bundle.sample_rate
raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t0:.3f}', '-t', f'{t1 - t0:.3f}', '-i', wav, '-ac', '1',
                      '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
seg = torch.from_numpy(np.frombuffer(raw, np.float32).copy())[None]
model = bundle.get_model(with_star=False)
with torch.inference_mode():
    em, _ = model(seg)
dic = bundle.get_dict(star=None)
toks = [[(i, dic[ch]) for i, c in enumerate(w) for ch in rom(c) if ch in dic] for w in words]
flat = [t for w in toks for _, t in w]
ali, scores = F.forced_align(em, torch.tensor([flat], dtype=torch.int32), blank=0)
spans = F.merge_tokens(ali[0], scores[0].exp())
ratio = seg.size(1) / em.size(1) / SR
res, k = [], 0
for w, tk in zip(words, toks):
    sp = spans[k:k + len(tk)]
    k += len(tk)
    if not sp:
        res.append(dict(w=w, s=None, e=None, p=0, letters=[]))
        continue
    first = {}
    for (li, _), s in zip(tk, sp):
        first.setdefault(li, round(t0 + s.start * ratio, 3))
    res.append(dict(w=w, s=round(t0 + sp[0].start * ratio, 3), e=round(t0 + sp[-1].end * ratio, 3),
                    p=round(float(sum(s.score for s in sp)) / len(sp), 2),
                    letters=[[c, first.get(i)] for i, c in enumerate(w)]))
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
print(f'{len(res)} words aligned in {t0:.2f}-{t1:.2f}')
