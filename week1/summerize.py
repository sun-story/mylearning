def normalize_token(token):
    cleaned = token.strip().lower()
    if not cleaned:
        return None
    if cleaned.startswith("#") or cleaned.startswith("//"):
        return None
    return cleaned


def parse_Entry(line, delimiter="="):
    if delimiter not in line:
        return None, None
    left, right = line.split(delimiter, 1)
    key = normalize_token(left)
    val_str = normalize_token(right)
    if key is None or val_str is None:
        return None, None
    
    if "," in val_str:
        items = [normalize_token(x) for x in val_str.split(",")]
        val = [x for x in items if x is not None]
    elif val_str.isdigit() or (val_str.startswith("-") and val_str[1:].isdigit()):
        val = int(val_str)
    else:
        val = val_str
    return key, val


def merge_values(existing, incoming):
    if isinstance(existing, list) and isinstance(incoming, list):
        return existing + [x for x in incoming if x not in existing]
    if isinstance(existing, int) and isinstance(incoming, int):
        return existing + incoming
    return incoming


def process_records(lines):
    registry = {}
    history = []

    for idx, raw_line in enumerate(lines):
        key, val = parse_Entry(raw_line)
        if key is None:
            continue
        
        if key in registry:
            registry[key] = merge_values(registry[key], val)
            history.append((idx, key, "updated"))
        else:
            registry[key] = val
            history.append((idx, key, "created"))

    summary = {
        "total_keys": len(registry),
        "data": registry,
        "log_count": len(history),
        "last_action": history[-1] if history else "none"
    }
    return summary