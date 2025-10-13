import os, re, xml.etree.ElementTree as ET
import shutil
from dataclasses import dataclass
from typing import Optional
from svgpathtools import parse_path


@dataclass
class InputData:
    input_path: str
    output_dir: str
    target_size: int = 128
    ndigits: int = 2

def fmt_number(num: float, ndigits: int = 2) -> str:
    val = round(float(num), ndigits)
    return str(int(val)) if val.is_integer() else f"{val:.{ndigits}f}".rstrip("0").rstrip(".")

_unit_re = re.compile(r"[a-zA-Z%]+$")

def parse_length(val: Optional[str], ref: Optional[float] = None) -> float:
    if val is None or not str(val).strip():
        return 0.0
    s = str(val).strip()
    if s.endswith('%'):
        if ref is None:
            raise ValueError(f"Cannot resolve percentage '{s}' without reference")
        return float(s[:-1]) * ref / 100.0
    return float(_unit_re.sub('', s))

_CMD_RE = re.compile(r'([MmZzLlHhVvCcSsQqTtAa])')
_SIGN_RE = re.compile(r'(?<![eE])([-+])')

def _sanitize_path(d: str) -> str:
    d = d.replace(',', ' ')
    d = _CMD_RE.sub(r' \1 ', d)
    d = _SIGN_RE.sub(r' \1', d)
    return re.sub(r'\s+', ' ', d).strip()

def _safe_parse_path(d: str):
    try:
        return parse_path(d)
    except Exception:
        return parse_path(_sanitize_path(d))

def _round_path_numbers(d_str: str, ndigits: int) -> str:
    return re.sub(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                  lambda m: fmt_number(m.group(), ndigits), d_str)

def _transform_path(d: str, x_min: float, y_min: float, s: float, ndigits: int) -> str:
    p = _safe_parse_path(d).translated(complex(-x_min, -y_min)).scaled(s, s)
    return _round_path_numbers(p.d(), ndigits)

_TRANSFORM_RE = re.compile(r'([a-zA-Z]+)\(([^)]*)\)')

def _rescale_transform(t_str: str, x_min: float, y_min: float,
                       s: float, ndigits: int = 2) -> str:
    out, pos = [], 0
    for m in _TRANSFORM_RE.finditer(t_str):
        out.append(t_str[pos:m.start()])
        pos = m.end()
        name, args = m.group(1), m.group(2)
        try:
            nums = [float(v) for v in re.split(r'[ ,]+', args.strip()) if v]
        except Exception:
            out.append(m.group(0))
            continue

        low = name.lower()

        if low == 'translate':
            if len(nums) == 1:
                nums[0] = nums[0] * s
            elif len(nums) >= 2:
                nums[0] = nums[0] * s
                nums[1] = nums[1] * s

        elif low == 'scale':
            pass

        elif low == 'rotate':
            if len(nums) == 3:
                nums[1] = (nums[1] - x_min) * s
                nums[2] = (nums[2] - y_min) * s

        elif low in {'skewx', 'skewy'}:
            pass

        elif low == 'matrix' and len(nums) == 6:
            a, b, c, d, e, f = nums
            e = s * (e + (a - 1) * x_min + c * y_min)
            f = s * (f + b * x_min + (d - 1) * y_min)
            a_str = fmt_number(a, 6)
            b_str = fmt_number(b, 6)
            c_str = fmt_number(c, 6)
            d_str = fmt_number(d, 6)
            e_str = fmt_number(e, ndigits)
            f_str = fmt_number(f, ndigits)
            out.append(f"matrix({a_str} {b_str} {c_str} {d_str} {e_str} {f_str})")
            continue
        else:
            out.append(m.group(0))
            continue

        nums_str = " ".join(fmt_number(v, ndigits) for v in nums)
        out.append(f"{name}({nums_str})")

    out.append(t_str[pos:])
    return "".join(out)

def _scale_style_lengths(style_str: str, s: float, ndigits: int) -> str:
    if not style_str:
        return style_str
    items = [kv for kv in style_str.split(';') if kv.strip()]
    out = []
    for kv in items:
        if ':' not in kv:
            out.append(kv); continue
        k, v = kv.split(':', 1)
        kk, vv = k.strip(), v.strip()
        if kk in ('stroke-width',):
            try:
                vv_num = parse_length(vv, None) * s
                vv = fmt_number(vv_num, ndigits)
            except Exception:
                pass
        elif kk == 'stroke-dasharray':
            parts = [p.strip() for p in vv.replace(',', ' ').split()]
            scaled = []
            ok = True
            for p in parts:
                try:
                    scaled.append(fmt_number(parse_length(p, None) * s, ndigits))
                except Exception:
                    ok = False
                    break
            if ok and scaled:
                vv = ','.join(scaled)
        out.append(f"{kk}:{vv}")
    return ';'.join(out)

_val_num_re = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

def _scale_filters(root: ET.Element, s: float, ndigits: int):
    for fe in root.iter():
        tag = fe.tag.split('}')[-1]
        if tag == 'feGaussianBlur':
            std = fe.get('stdDeviation')
            if std:
                parts = [p for p in re.split(r'[ ,]+', std.strip()) if p]
                try:
                    nums = [float(x) * s for x in parts]
                    fe.set('stdDeviation', ' '.join(fmt_number(v, ndigits) for v in nums))
                except Exception:
                    pass

def _scale_animatetransform(root: ET.Element, x_min: float, y_min: float,
                            s: float, ndigits: int):
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        if tag != 'animateTransform':
            continue
        typ = (el.get('type') or el.get('typeName') or '').lower()
        vals = el.get('values')
        if not vals:
            continue

        if typ == 'translate':
            nums = [fmt_number(float(n) * s, ndigits) for n in _val_num_re.findall(vals)]
            i = 0
            def repl(_):
                nonlocal i
                i += 1
                return nums[i-1]
            el.set('values', _val_num_re.sub(repl, vals))

        elif typ == 'rotate':
            chunks = vals.split(';')
            out_chunks = []
            for ch in chunks:
                ns = [float(n) for n in _val_num_re.findall(ch)]
                if len(ns) >= 3:
                    ns[1] = (ns[1] - x_min) * s
                    ns[2] = (ns[2] - y_min) * s
                    out_chunks.append(' '.join(fmt_number(v, ndigits) for v in ns[:3]))
                else:
                    out_chunks.append(ch)
            el.set('values', ';'.join(out_chunks))

        else:
            pass

def _scale_gradients_user_space(root: ET.Element, x_min: float, y_min: float,
                                s: float, ndigits: int):
    for g in root.iter():
        tag = g.tag.split('}')[-1]
        if tag not in ('linearGradient', 'radialGradient'):
            continue
        units = (g.get('gradientUnits') or '').lower()
        if units != 'userspaceonuse':
            continue

        def trans(v, is_x):
            return fmt_number((parse_length(v) - (x_min if is_x else y_min)) * s, ndigits)
        def only(v):
            return fmt_number(parse_length(v) * s, ndigits)

        if tag == 'linearGradient':
            if g.get('x1') is not None: g.set('x1', trans(g.get('x1'), True))
            if g.get('y1') is not None: g.set('y1', trans(g.get('y1'), False))
            if g.get('x2') is not None: g.set('x2', trans(g.get('x2'), True))
            if g.get('y2') is not None: g.set('y2', trans(g.get('y2'), False))
        else:  # radialGradient
            if g.get('cx') is not None: g.set('cx', trans(g.get('cx'), True))
            if g.get('cy') is not None: g.set('cy', trans(g.get('cy'), False))
            if g.get('fx') is not None: g.set('fx', trans(g.get('fx'), True))
            if g.get('fy') is not None: g.set('fy', trans(g.get('fy'), False))
            if g.get('r')  is not None: g.set('r',  only(g.get('r')))

_num_re = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?')

def _scale_number_list_attr(attrib: dict, key: str, s: float, ndigits: int):
    if key not in attrib:
        return
    raw = attrib[key]
    nums = [fmt_number(float(n) * s, ndigits) for n in _num_re.findall(raw)]
    i = 0
    def repl(_):
        nonlocal i
        v = nums[i]; i += 1
        return v
    attrib[key] = _num_re.sub(repl, raw)

def _scale_animate_numeric(root: ET.Element, attr_names: set, s: float, ndigits: int):
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        if tag != 'animate':
            continue
        target = (el.get('attributeName') or '').strip().lower()
        if target not in attr_names:
            continue
        for key in ('values', 'from', 'to', 'by'):
            val = el.get(key)
            if not val:
                continue
            nums = [fmt_number(float(n) * s, ndigits) for n in _num_re.findall(val)]
            i = 0
            def repl(_):
                nonlocal i
                v = nums[i]; i += 1
                return v
            el.set(key, _num_re.sub(repl, val))
            
def _scale_animate_path_d(root: ET.Element, x_min: float, y_min: float,
                          s: float, ndigits: int):
    for el in root.iter():
        tag = el.tag.split('}')[-1]
        if tag != 'animate':
            continue
        if (el.get('attributeName') or '').strip().lower() != 'd':
            continue
        for key in ('values', 'from', 'to', 'by'):
            val = el.get(key)
            if not val:
                continue
            parts = val.split(';')
            out = []
            for p in parts:
                p = p.strip()
                if not p:
                    out.append(p)
                    continue
                try:
                    out.append(_transform_path(p, x_min, y_min, s, ndigits))
                except Exception:
                    out.append(p)
            el.set(key, ';'.join(out))

def parse_viewbox(vb: str):
    if not vb:
        raise ValueError("empty viewBox")
    # 先统一替换逗号为空格
    vb = vb.replace(',', ' ')
    parts = vb.split()
    if len(parts) != 4:
        raise ValueError(f"bad viewBox: {vb!r}")
    x_min, y_min, vb_w, vb_h = map(float, parts)
    return x_min, y_min, vb_w, vb_h


# --------------------------- Main Entry ---------------------------
def normalize_svg(in_svg: str, out_dir: str,
                  target: int = 128, ndigits: int = 2) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(in_svg))

    try:
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
        ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
        root = ET.parse(in_svg).getroot()
    except Exception as e:
        print(f"⚠️  SVG parse failed ({in_svg}) → {e}");  return

    try:
        vb = root.get('viewBox')
        if vb:
            x_min, y_min, vb_w, vb_h = parse_viewbox(vb)
        else:
            x_min = y_min = 0.0
            vb_w = parse_length(root.get('width'), target) or target
            vb_h = parse_length(root.get('height'), target) or target
        if vb_w == 0 or vb_h == 0:
            raise ValueError("viewBox/width/height all zero")
    except Exception as e:
        print(f"⚠️  Size parse failed ({in_svg}) → {e}");  return

    s = min(target / vb_w, target / vb_h)
    
    if s == 1:
        shutil.copy(in_svg, out_path)
        return

    def ref_len(is_x): return vb_w if is_x else vb_h
    def trans_scaled(v, is_x):
        return fmt_number((parse_length(v, ref_len(is_x)) - (x_min if is_x else y_min)) * s, ndigits)
    def only_scaled(v, is_x):
        return fmt_number(parse_length(v, ref_len(is_x)) * s, ndigits)

    for elem in root.iter():
        tag, at = elem.tag.split('}')[-1], elem.attrib
        try:
            if tag == 'svg' and elem is not root:
                for k in ('x', 'y'):
                    if k in at:
                        at[k] = trans_scaled(at[k], k == 'x')
                for k in ('width', 'height'):
                    if k in at:
                        at[k] = only_scaled(at[k], k == 'width')
                if 'viewBox' in at:
                    minx, miny, w, h = map(float, at['viewBox'].split())
                    new_minx = fmt_number((minx - x_min) * s, ndigits)
                    new_miny = fmt_number((miny - y_min) * s, ndigits)
                    new_w    = fmt_number(w * s, ndigits)
                    new_h    = fmt_number(h * s, ndigits)
                    at['viewBox'] = f"{new_minx} {new_miny} {new_w} {new_h}"

            if tag == 'path' and 'd' in at:
                at['d'] = _transform_path(at['d'], x_min, y_min, s, ndigits)

            elif tag == 'line':
                for k in ('x1', 'y1', 'x2', 'y2'):
                    if k in at: at[k] = trans_scaled(at[k], k.startswith('x'))

            elif tag == 'rect':
                if 'x' in at: at['x'] = trans_scaled(at['x'], True)
                if 'y' in at: at['y'] = trans_scaled(at['y'], False)
                for k in ('width', 'height'):
                    if k in at: at[k] = only_scaled(at[k], k == 'width')
                for k in ('rx', 'ry'):
                    if k in at: at[k] = only_scaled(at[k], k == 'rx')

            elif tag == 'circle':
                if 'cx' in at: at['cx'] = trans_scaled(at['cx'], True)
                if 'cy' in at: at['cy'] = trans_scaled(at['cy'], False)
                if 'r'  in at: at['r']  = only_scaled(at['r'], True)

            elif tag == 'ellipse':
                if 'cx' in at: at['cx'] = trans_scaled(at['cx'], True)
                if 'cy' in at: at['cy'] = trans_scaled(at['cy'], False)
                if 'rx' in at: at['rx'] = only_scaled(at['rx'], True)
                if 'ry' in at: at['ry'] = only_scaled(at['ry'], False)

            elif tag in ('polygon', 'polyline') and 'points' in at:
                pts = re.split(r'[ ,]+', at['points'].strip())
                at['points'] = ' '.join(
                    trans_scaled(n, i % 2 == 0) for i, n in enumerate(pts) if n
                )

            elif tag == 'text':
                if 'x' in at: at['x'] = trans_scaled(at['x'], True)
                if 'y' in at: at['y'] = trans_scaled(at['y'], False)
                if 'font-size' in at:
                    at['font-size'] = only_scaled(at['font-size'], True)

            if 'style' in at:
                at['style'] = _scale_style_lengths(at['style'], s, ndigits)
            if 'stroke-width' in at:
                try:
                    at['stroke-width'] = fmt_number(parse_length(at['stroke-width']) * s, ndigits)
                except Exception:
                    pass

            _scale_number_list_attr(at, 'stroke-dasharray', s, ndigits)
            _scale_number_list_attr(at, 'stroke-dashoffset', s, ndigits)

            if 'transform' in at:
                at['transform'] = _rescale_transform(at['transform'], x_min, y_min, s, ndigits)

        except Exception as e:
            print(f"⚠️  Element transform failed ({in_svg}) → {e}")
            continue

    _scale_filters(root, s, ndigits)
    _scale_animatetransform(root, x_min, y_min, s, ndigits)
    _scale_gradients_user_space(root, x_min, y_min, s, ndigits)
    _scale_animate_numeric(root, {'stroke-dashoffset'}, s, ndigits)
    _scale_animate_path_d(root, x_min, y_min, s, ndigits)

    root.set('viewBox', f'0 0 {target} {target}')
    for k in ('width', 'height'):
        root.set(k, str(target))

    try:
        ET.ElementTree(root).write(out_path, encoding='utf-8', xml_declaration=True)
        print(f"✅  Saved → {out_path}")
    except Exception as e:
        print(f"⚠️  Write failed ({in_svg}) → {e}")

def parallel_normalize_svg(input_data: InputData):
    try:
        normalize_svg(input_data.input_path, input_data.output_dir,
                      input_data.target_size, input_data.ndigits)
    except Exception as e:
        print(f"⚠️  Unexpected error ({input_data.input_path}) → {e}")

if __name__ == "__main__":
    demo_in  = "PATH_TO_INPUT_SVG_FILE"
    demo_out = "PATH_TO_OUTPUT_DIR"
    normalize_svg(demo_in, demo_out)