"""术语提取测试：分词质量与降级路径。

这是被真实数据打回来的一块——上线首日 term_stats 攒了 136 个词，
大半是双字滑窗跨词切出来的碎片（"个怎""要登""站字"），
导致 R2「话题加热」永远只能得到荒唐假设。

Run:  python -m unittest discover -s tests -p "test_*.py" -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_pkg = types.ModuleType("ph_terms_pkg")
_pkg.__path__ = [_ROOT]
sys.modules["ph_terms_pkg"] = _pkg

spec = importlib.util.spec_from_file_location("ph_terms_pkg.engine", _ROOT + "/engine.py")
eng = importlib.util.module_from_spec(spec)
sys.modules["ph_terms_pkg.engine"] = eng
spec.loader.exec_module(eng)


HAS_JIEBA = eng._jieba() is not None
NEED_JIEBA = unittest.skipUnless(HAS_JIEBA, "需要 jieba 才有词级切分")


class TestTerms(unittest.TestCase):
    @NEED_JIEBA
    def test_no_cross_word_fragments(self):
        t = eng.terms("这个怎么搞")
        for bad in ("个怎", "么搞", "这个", "怎么"):
            self.assertNotIn(bad, t, f"{bad} 是虚词或跨词碎片")

    @NEED_JIEBA
    def test_keeps_domain_words(self):
        t = eng.terms("国家助学金的材料我还没准备好")
        self.assertIn("助学金", t)
        self.assertIn("材料", t)

    @NEED_JIEBA
    def test_keeps_latin_terms(self):
        t = eng.terms("某课题 综述第 8 章")
        self.assertIn("课题", t)
        self.assertIn("综述", t)

    @NEED_JIEBA
    def test_drops_punctuation_and_single_chars(self):
        t = eng.terms("明天要交实验报告，这个怎么搞")
        self.assertIn("实验报告", t)
        for bad in ("，", "搞", "交"):
            self.assertNotIn(bad, t)

    def test_fallback_without_jieba_has_no_fragments(self):
        """把分词器摘掉，走双字回退，也不能吐虚词。"""
        old = eng._JIEBA
        eng._JIEBA = False
        try:
            t = eng.terms("这个怎么搞")
            self.assertTrue(all(w not in eng.STOPWORDS for w in t))
        finally:
            eng._JIEBA = old

    def test_empty_input(self):
        self.assertEqual(eng.terms(""), set())


if __name__ == "__main__":
    unittest.main(verbosity=2)
