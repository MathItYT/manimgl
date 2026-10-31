from __future__ import annotations

import itertools as it
from difflib import SequenceMatcher

from manimlib.animation.composition import AnimationGroup
from manimlib.animation.fading import FadeIn, FadeOut
from manimlib.animation.transform import Transform
from manimlib.mobject.mobject import Mobject
from manimlib.mobject.types.vectorized_mobject import VGroup
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.mobject.svg.string_mobject import StringMobject

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable
    from manimlib.scene.scene import Scene


class TransformMatchingParts(AnimationGroup):
    def __init__(
        self,
        source: Mobject,
        target: Mobject,
        matched_pairs: Iterable[tuple[Mobject, Mobject]] = [],
        match_animation: type = Transform,
        mismatch_animation: type = Transform,
        run_time: float = 2,
        lag_ratio: float = 0,
        **kwargs,
    ):
        self.source = source
        self.target = target
        self.match_animation = match_animation
        self.mismatch_animation = mismatch_animation
        self.anim_config = dict(**kwargs)

        # We will progressively build up a list of transforms
        # from pieces in source to those in target. These
        # two lists keep track of which pieces are accounted
        # for so far
        self.source_pieces = source.family_members_with_points()
        self.target_pieces = target.family_members_with_points()
        self.anims = []

        for pair in matched_pairs:
            self.add_transform(*pair)

        # Match any pairs with the same shape
        for pair in self.find_pairs_with_matching_shapes(self.source_pieces, self.target_pieces):
            self.add_transform(*pair)

        # Finally, account for mismatches
        for source_piece in self.source_pieces:
            if any([source_piece in anim.mobject.get_family() for anim in self.anims]):
                continue
            self.anims.append(FadeOut(
                source_piece,
                **self.anim_config
            ))
        for target_piece in self.target_pieces:
            if any([target_piece in anim.mobject.get_family() for anim in self.anims]):
                continue
            self.anims.append(FadeIn(
                target_piece,
                **self.anim_config
            ))

        super().__init__(
            *self.anims,
            run_time=run_time,
            lag_ratio=lag_ratio,
        )

    def add_transform(
        self,
        source: Mobject,
        target: Mobject,
    ):
        new_source_pieces = source.family_members_with_points()
        new_target_pieces = target.family_members_with_points()
        if len(new_source_pieces) == 0 or len(new_target_pieces) == 0:
            # Don't animate null sorces or null targets
            return
        source_is_new = all(char in self.source_pieces for char in new_source_pieces)
        target_is_new = all(char in self.target_pieces for char in new_target_pieces)
        if not source_is_new or not target_is_new:
            return

        transform_type = self.mismatch_animation
        if source.has_same_shape_as(target):
            transform_type = self.match_animation

        self.anims.append(transform_type(source, target, **self.anim_config))
        for char in new_source_pieces:
            self.source_pieces.remove(char)
        for char in new_target_pieces:
            self.target_pieces.remove(char)

    def find_pairs_with_matching_shapes(
        self,
        chars1: list[Mobject],
        chars2: list[Mobject]
    ) -> list[tuple[Mobject, Mobject]]:
        result = []
        for char1, char2 in it.product(chars1, chars2):
            if char1.has_same_shape_as(char2):
                result.append((char1, char2))
        return result

    def clean_up_from_scene(self, scene: Scene) -> None:
        super().clean_up_from_scene(scene)
        scene.remove(self.mobject, self.source)
        scene.add(self.target)


class TransformMatchingShapes(TransformMatchingParts):
    """Alias for TransformMatchingParts"""
    pass


class TransformMatchingStrings(TransformMatchingParts):
    def __init__(
        self,
        source: StringMobject,
        target: StringMobject,
        matched_keys: Iterable[str] = [],
        key_map: dict[str, str] = dict(),
        matched_pairs: Iterable[tuple[VMobject, VMobject]] = [],
        **kwargs,
    ):
        matched_pairs = [
            *matched_pairs,
            *self.matching_blocks(source, target, matched_keys, key_map),
        ]

        super().__init__(
            source, target,
            matched_pairs=matched_pairs,
            **kwargs,
        )

    @staticmethod
    def _annotate_tokens(mob: StringMobject) -> list[tuple[tuple[int, int], list[int], str, str, str]]:
        raw_tokens = sorted(
            [t for t in mob.get_token_submob_map() if t[1]],
            key=lambda t: t[0][0]
        )
        s = mob.string
        n = len(s)
        is_tex = mob.__class__.__name__ in ("Tex", "TexText", "SingleStringTex")
        script_spans: list[tuple[int, int, str, str]] = []

        def find_matching_delim(open_idx: int, open_ch: str, close_ch: str) -> int:
            depth = 1
            for j in range(open_idx + 1, n):
                if s[j] == open_ch and (not is_tex or j == 0 or s[j - 1] != "\\"):
                    depth += 1
                elif s[j] == close_ch and (not is_tex or j == 0 or s[j - 1] != "\\"):
                    depth -= 1
                    if depth == 0:
                        return j + 1
            return n

        def get_script_arg_end(idx: int) -> int:
            while idx < n and s[idx].isspace():
                idx += 1
            if idx >= n:
                return idx
            if is_tex and s[idx] == "{":
                return find_matching_delim(idx, "{", "}")
            if not is_tex and s[idx] == "(":
                return find_matching_delim(idx, "(", ")")
            for (t_start, t_end), _, _ in raw_tokens:
                if t_start >= idx:
                    return t_end
            return idx + 1

        for i, ch in enumerate(s):
            if ch in "^_" and (i == 0 or s[i - 1] != "\\"):
                arg_end = get_script_arg_end(i + 1)
                base_sym = ""
                for (t_start, t_end), _, tok_sym in reversed(raw_tokens):
                    if t_end <= i:
                        in_prev_script = False
                        for sp_start, sp_end, _, sp_base in script_spans:
                            if sp_start <= t_start and t_end <= sp_end:
                                base_sym = sp_base
                                in_prev_script = True
                                break
                        if not in_prev_script:
                            base_sym = tok_sym
                        break
                script_spans.append((i + 1, arg_end, ch, base_sym))

        annotated: list[tuple[tuple[int, int], list[int], str, str, str]] = []
        for span, idxs, sym in raw_tokens:
            t_start, t_end = span
            prefix = ""
            base = ""
            for sp_start, sp_end, op_ch, sp_base in script_spans:
                if sp_start <= t_start and t_end <= sp_end:
                    prefix += op_ch
                    if not base:
                        base = sp_base
            annotated.append((span, idxs, sym, prefix, base))
        return annotated

    def matching_blocks(
        self,
        source: StringMobject,
        target: StringMobject,
        matched_keys: Iterable[str],
        key_map: dict[str, str]
    ) -> list[tuple[VMobject, VMobject]]:
        tokens1 = self._annotate_tokens(source)
        tokens2 = self._annotate_tokens(target)

        # Start with user specified matches
        user_pairs = [(source[key1], target[key2]) for key1, key2 in key_map.items()]
        user_pairs += [(source[key], target[key]) for key in matched_keys]
        blocks: list[tuple[VMobject, VMobject]] = []
        for sub_source, sub_target in user_pairs:
            if (
                isinstance(sub_source, VGroup)
                and isinstance(sub_target, VGroup)
                and len(sub_source) == len(sub_target)
                and len(sub_source) > 0
            ):
                for s_part, t_part in zip(sub_source, sub_target):
                    blocks.append((s_part, t_part))
            else:
                blocks.append((sub_source, sub_target))

        # Nullify any intersections with those matches in the two token lists
        matched1 = [False] * len(tokens1)
        matched2 = [False] * len(tokens2)
        for sub_source, sub_target in blocks:
            s_fam = set(sub_source.family_members_with_points())
            t_fam = set(sub_target.family_members_with_points())
            for i, tok1 in enumerate(tokens1):
                if any(
                    source.submobjects[idx] in s_fam
                    for idx in tok1[1]
                    if idx < len(source.submobjects)
                ):
                    matched1[i] = True
            for j, tok2 in enumerate(tokens2):
                if any(
                    target.submobjects[idx] in t_fam
                    for idx in tok2[1]
                    if idx < len(target.submobjects)
                ):
                    matched2[j] = True

        # Group together longest matching substrings from most specific context to general
        for key_fn in (
            lambda t: f"{t[4]}::{t[3]}::{t[2]}" if t[3] else t[2],
            lambda t: f"{t[3]}::{t[2]}" if t[3] else t[2],
            lambda t: t[2],
        ):
            syms1 = [f"__null1_{i}__" if matched1[i] else key_fn(tok) for i, tok in enumerate(tokens1)]
            syms2 = [f"__null2_{j}__" if matched2[j] else key_fn(tok) for j, tok in enumerate(tokens2)]
            while True:
                matcher = SequenceMatcher(None, syms1, syms2, autojunk=False)
                match = matcher.find_longest_match(0, len(syms1), 0, len(syms2))
                if match.size == 0:
                    break

                for k in range(match.size):
                    i1 = match.a + k
                    i2 = match.b + k
                    s_idxs = [idx for idx in tokens1[i1][1] if idx < len(source.submobjects)]
                    t_idxs = [idx for idx in tokens2[i2][1] if idx < len(target.submobjects)]
                    if s_idxs and t_idxs:
                        s_mob = (
                            source.submobjects[s_idxs[0]]
                            if len(s_idxs) == 1
                            else VGroup(*(source.submobjects[idx] for idx in s_idxs))
                        )
                        t_mob = (
                            target.submobjects[t_idxs[0]]
                            if len(t_idxs) == 1
                            else VGroup(*(target.submobjects[idx] for idx in t_idxs))
                        )
                        blocks.append((s_mob, t_mob))
                    matched1[i1] = True
                    matched2[i2] = True
                    syms1[i1] = f"__null1_{i1}__"
                    syms2[i2] = f"__null2_{i2}__"

        return blocks


class TransformMatchingTex(TransformMatchingStrings):
    """Alias for TransformMatchingStrings"""
    pass
