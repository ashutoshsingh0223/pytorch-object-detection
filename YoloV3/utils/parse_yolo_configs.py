from typing import Union, List, Dict, Any
from pathlib import Path
from collections import OrderedDict


def parse_yolo_config(path: Union['Path', str]):
    path = Path(path) if isinstance(path, str) else path
    cfg: List[str] = open(path, 'r').read().split('\n')

    lines: List[str] = [line.strip() for line in cfg if line.strip()]
    lines = [line for line in lines if line[0] != '#']

    blocks: List[Dict[str, Any]] = []
    block: Dict[str, Any] = OrderedDict()

    for line in lines:
        if line[0] == '[':
            if block:
                blocks.append(block)
            block = OrderedDict()
            block['type'] = line[1: -1]
        else:
            key, value = line.split('=')
            try:
                value = int(value)
            except ValueError:
                pass

            block[key] = value

    blocks.append(block)
    return blocks
