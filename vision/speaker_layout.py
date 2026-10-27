"""Learn a name-box candidate from existing OCR, with explicit confirmation."""
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

from desktop_pet.vision.ocr import OCRLine, _result_from_lines, normalize_speaker_name, strip_ui_chrome


PROFILE_PATH = Path(__file__).resolve().parent.parent / 'data/speaker_layouts.json'


def load_profiles(path=PROFILE_PATH):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_profile(details, *, remove=False, path=PROFILE_PATH):
    profiles = load_profiles(path)
    key = details['key']
    if remove:
        profiles.pop(key, None)
    else:
        profiles[key] = {k: details[k] for k in ('title', 'process', 'box', 'aspect')}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _close(left, right):
    return (min(abs(left[0] - right[0]), abs((left[0] + left[2] - right[0] - right[2]) / 2)) < .025
            and abs((left[1] + left[3] - right[1] - right[3]) / 2) < .025)


def _preview(frame, box):
    width, height = frame.size
    # Include surrounding artwork so the user can check the candidate's location.
    left = max(0, round((box[0] - .04) * width))
    top = max(0, round((box[1] - .04) * height))
    right = min(width, round((box[2] + .12) * width))
    bottom = min(height, round((box[3] + .08) * height))
    patch = frame.crop((left, top, right, bottom)).convert('RGB')
    draw = ImageDraw.Draw(patch)
    draw.rectangle((round(box[0]*width)-left, round(box[1]*height)-top,
                    round(box[2]*width)-left, round(box[3]*height)-top), outline='red', width=2)
    patch.thumbnail((440, 180), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    patch.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


class SpeakerLayout:
    def __init__(self, path=PROFILE_PATH):
        self.path = path
        self.identity = None
        self.samples = []

    def observe(self, result, frame, crop_box, *, title, process):
        if frame is None or not crop_box or not title or not process:
            return result, {}
        # Exclude UI words before either learning positions or reading a saved box.
        lines = []
        for line in result.lines:
            text = strip_ui_chrome(line.text)
            if text:
                lines.append(line if text == line.text else OCRLine(text, line.confidence, line.box))
        if tuple(lines) != result.lines:
            result = _result_from_lines(lines)
        key = hashlib.sha256((process.casefold() + '\n' + title).encode()).hexdigest()[:24]
        aspect = frame.width / frame.height
        identity = (key, tuple(crop_box), round(aspect, 2))
        if identity != self.identity:
            self.samples = []
            self.identity = identity
        info = dict(key=key, title=title, process=process, aspect=aspect,
                    status='learning', speaker='', examples=[], box=None, preview='')
        cx1, cy1, cx2, cy2 = crop_box
        located = []
        for line in result.lines:
            if line.box is None:
                continue
            x1, y1, x2, y2 = line.box
            box = (cx1+x1*(cx2-cx1), cy1+y1*(cy2-cy1),
                   cx1+x2*(cx2-cx1), cy1+y2*(cy2-cy1))
            located.append((line, box))
        bodies = [(line, box) for line, box in located
                  if len(line.text.strip()) >= 4 and box[2]-box[0] >= .08
                  and (len(line.text.strip()) >= 8 or any(c in line.text for c in '「」『』，。！？…'))]
        names = [(line, box) for line, box in located
                 if line.confidence >= .40 and box[2]-box[0] <= .22 and box[3]-box[1] <= .08
                 and normalize_speaker_name(line.text)]
        candidates = [(line, box) for line, box in names if any(
            (-.02 <= body[1]-box[3] <= .18 and abs(box[0]-body[0]) <= .16)
            or (0 <= body[0]-box[2] <= .16
                and abs((body[1]+body[3]-box[1]-box[3])/2) <= .03)
            for _, body in bodies)]
        signature = '\n'.join(line.text for line, _ in bodies)
        if signature and (not self.samples or self.samples[-1][0] != signature):
            self.samples.append((signature, [(normalize_speaker_name(line.text), box) for line, box in candidates]))
            self.samples = self.samples[-12:]
        learned = None
        for line, box in candidates:
            seen = {text for _, items in self.samples for text, old in items if _close(old, box)}
            count = len({sig for sig, items in self.samples if any(_close(old, box) for _, old in items)})
            if count >= 3 and len(seen) >= 2:
                boxes = [old for _, items in self.samples for _, old in items if _close(old, box)]
                region = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                          max(b[2] for b in boxes), max(b[3] for b in boxes)]
                learned = dict(box=region, examples=sorted(seen)[:6], observations=count)
                break
        saved = load_profiles(self.path).get(key)
        valid = (isinstance(saved, dict) and isinstance(saved.get('box'), list)
                 and len(saved['box']) == 4 and abs(aspect / saved.get('aspect', aspect) - 1) < .02)
        if valid and learned and not _close(saved['box'], learned['box']):
            valid = False
        if valid:
            info.update(status='confirmed', box=saved['box'])
            # The confirmed location is sufficient even when OCR misses the dialogue.
            matches = [(line, box) for line, box in names if _close(box, saved['box'])]
            if len(matches) == 1:
                name, _ = matches[0]
                info['speaker'] = normalize_speaker_name(name.text)
                # Reorder only the current observed name; never carry a previous speaker.
                result = _result_from_lines([name, *[line for line in result.lines if line is not name]])
        elif learned:
            info.update(learned, status='candidate')
        elif saved:
            info['status'] = 'layout_changed'
        if info['box']:
            info['preview'] = _preview(frame, info['box'])
        return result, info


def observe_speaker_layout(result, image, frame, rect, cache):
    if not image.info.get('vn_ocr_box'):
        return result, {}
    learner = cache.setdefault('speaker_layout_learner', SpeakerLayout())
    return learner.observe(result, frame, image.info['vn_ocr_box'],
                           title=str(getattr(rect, 'title', '') or ''),
                           process=str(getattr(rect, 'process_name', '') or ''))
