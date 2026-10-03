import os
from functools import lru_cache
import xml.etree.ElementTree as ET
from manimlib.utils.directories import get_cache_dir


def get_typst_dir() -> str:
    typst_dir = os.path.join(get_cache_dir(), "typst")
    os.makedirs(typst_dir, exist_ok=True)
    return typst_dir


def _parse_coord(val: str) -> float:
    if not val:
        return 0.0
    for unit in ("pt", "px", "mm", "cm", "in"):
        if val.endswith(unit):
            val = val[:-len(unit)]
            break
    try:
        return float(val)
    except ValueError:
        return 0.0


def flatten_typst_svg(svg_path: str) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    def get_tag(elem):
        return elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

    ns = root.tag.split("}")[0] + "}" if "}" in root.tag else ""

    glyph_defs = {}
    defs_elements = []

    for elem in list(root.iter()):
        if get_tag(elem) == "defs":
            defs_elements.append(elem)
            for child in elem.iter():
                elem_id = child.attrib.get("id")
                if not elem_id:
                    continue

                if get_tag(child) == "path" and child.attrib.get("d"):
                    glyph_defs[elem_id] = [dict(child.attrib)]
                else:
                    paths = []
                    c_trans = child.attrib.get("transform", "")
                    for p in child.iter():
                        if get_tag(p) == "path" and p.attrib.get("d"):
                            p_attrs = dict(p.attrib)
                            p_trans = p_attrs.get("transform", "")
                            trans_list = [t for t in (c_trans, p_trans) if t]
                            if trans_list:
                                p_attrs["transform"] = " ".join(trans_list)
                            for attr in ("fill", "stroke", "stroke-width", "opacity"):
                                if attr not in p_attrs and attr in child.attrib:
                                    p_attrs[attr] = child.attrib[attr]
                            paths.append(p_attrs)
                    if paths:
                        glyph_defs[elem_id] = paths

    parent_map = {c: p for p in root.iter() for c in p}

    def get_inherited_attr(elem, attr):
        curr = elem
        while curr is not None:
            val = curr.attrib.get(attr)
            if val:
                return val
            curr = parent_map.get(curr)
        return None

    for elem in list(root.iter()):
        if get_tag(elem) == "use":
            parent = parent_map.get(elem)
            if parent is None:
                continue

            href = elem.attrib.get("href") or elem.attrib.get(
                "{http://www.w3.org/1999/xlink}href"
            )
            if not href:
                parent.remove(elem)
                continue

            glyph_id = href.lstrip("#")
            if glyph_id not in glyph_defs:
                parent.remove(elem)
                continue

            x = _parse_coord(elem.attrib.get("x", "0"))
            y = _parse_coord(elem.attrib.get("y", "0"))
            use_transform = elem.attrib.get("transform", "")

            use_transforms = []
            if x != 0 or y != 0:
                use_transforms.append(f"translate({x}, {y})")
            if use_transform:
                use_transforms.append(use_transform)

            idx = list(parent).index(elem)
            parent.remove(elem)

            for path_data in glyph_defs[glyph_id]:
                d_attr = path_data.get("d")
                if not d_attr or not d_attr.strip():
                    continue

                new_path = ET.Element(f"{ns}path")
                new_path.attrib["d"] = d_attr

                combined_trans = list(use_transforms)
                if path_data.get("transform"):
                    combined_trans.append(path_data["transform"])
                if combined_trans:
                    new_path.attrib["transform"] = " ".join(combined_trans)

                for attr in ("fill", "stroke", "stroke-width", "opacity"):
                    val = (
                        elem.attrib.get(attr)
                        or path_data.get(attr)
                        or get_inherited_attr(parent, attr)
                    )
                    if val:
                        new_path.attrib[attr] = val

                parent.insert(idx, new_path)
                parent_map[new_path] = parent
                idx += 1

    for defs_elem in defs_elements:
        parent = parent_map.get(defs_elem)
        if parent is not None:
            parent.remove(defs_elem)

    parent_map = {c: p for p in root.iter() for c in p}
    for elem in list(root.iter()):
        if get_tag(elem) == "path":
            d_val = elem.attrib.get("d")
            if not d_val or not d_val.strip():
                parent = parent_map.get(elem)
                if parent is not None:
                    parent.remove(elem)

    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


@lru_cache()
def typst_to_svg(content: str) -> str:
    try:
        import typst
        return typst.compile(content.encode(), format="svg")
    except ImportError:
        raise RuntimeError(
            "Typst no está instalado. Por favor, instálalo siguiendo las instrucciones en https://typst.app/install"
        )
    except Exception as e:
        raise RuntimeError(f"Error de sintaxis en Typst:\n{e}")

    # flatten_typst_svg(svg_path)
    # return svg_path