#!/usr/bin/env python3
"""
arabic_fix.py — Repair Arabic text where PDF extraction reversed lam ligatures.

Some Arabic PDF fonts draw lam + alef (لا) and lam + meem (لم) as one glyph.
The extractor then writes the two letters in the wrong order:
    المدينة -> املدينة     الأرض -> األرض     زيكولا -> زيكوال
Queries are typed correctly, so without this repair their embeddings do not
match the indexed chunks and retrieval misses the right passages.
"""

import re

AR = "ء-ي"

# Definite article: "ال" + (م أ آ إ ا) extracted as "ا" + letter + "ل".
# Safe as a rule because these swapped forms are not real word starts.
# The optional prefix covers attached و ف ب ك (e.g. واملدينة -> والمدينة).
_ARTICLE = re.compile(rf"(?<![{AR}])([وفبك]?)ا([مأآإا])ل")

# Inside a word "ال" is ambiguous: خالد and قال are correct, زيكوال and كالم are
# broken. Only these reviewed words are repaired; everything else is left alone.
_WORD_FIXES = {
    "زيكوال": "زيكولا", "بزيكوال": "بزيكولا", "لزيكوال": "لزيكولا", "وزيكوال": "وزيكولا",
    "إال": "إلا", "أال": "ألا", "بأال": "بألا", "لوال": "لولا",
    "قليال": "قليلا", "طويال": "طويلا", "فعال": "فعلا", "أهال": "أهلا", "أوال": "أولا",
    "ليال": "ليلا", "قائال": "قائلا", "مقابال": "مقابلا", "خجال": "خجلا", "متعجال": "متعجلا",
    "كالم": "كلام", "كالمك": "كلامك", "كالمه": "كلامه", "هؤالء": "هؤلاء",
    "مالبس": "ملابس", "مالبسه": "ملابسه", "مالبسها": "ملابسها", "بمالبسه": "بملابسه",
    "المالبس": "الملابس",
    "ثالثة": "ثلاثة", "ثالث": "ثلاث", "ثالثمائة": "ثلاثمائة", "ثالثين": "ثلاثين",
    "أقالم": "أقلام", "األقالم": "الأقلام", "الأقالم": "الأقلام",
    "آالم": "آلام", "آالت": "آلات", "عالمة": "علامة", "عضالته": "عضلاته",
    "أطفاال": "أطفالا", "مالمحه": "ملامحه", "أحالم": "أحلام", "ظالم": "ظلام", "الظالم": "الظلام",
    "صالة": "صلاة", "رجالن": "رجلان", "وصال": "وصلا", "واصال": "واصلا", "يكمال": "يكملا",
    "أالقيه": "ألاقيه", "الإطالق": "الإطلاق", "الاستسالم": "الاستسلام",
    "احتفاالت": "احتفالات", "احتفاالتنا": "احتفالاتنا", "احتفاال": "احتفالا",
}
_WORD = re.compile(rf"[{AR}]+")


def fix_arabic_ligatures(text: str) -> str:
    text = _ARTICLE.sub(r"\1ال\2", text)
    return _WORD.sub(lambda m: _WORD_FIXES.get(m.group(0), m.group(0)), text)
