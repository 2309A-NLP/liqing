"""
RAG 问答系统 — 运行全部测试 + 详细报告
工单编号：人工智能NLP-RAG-基于PDF文档的问答系统

用法：
  python tests/test_all.py           # 详细报告模式
  python tests/test_all.py --quiet   # 简洁模式
"""

import sys
import time
import argparse
from pathlib import Path

# 路径修复
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 上线 TensorFlow 噪音
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class TestCase:
    """包装测试函数，捕获执行详情"""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self.doc = (fn.__doc__ or "").strip().split("\n")[0]
        self.details = []

    def run(self):
        self.start = time.time()
        self.details = []
        self.passed = True
        self.error_msg = ""
        try:
            self.fn(self)  # 把 TestCase 自身传给测试函数
            self.elapsed = (time.time() - self.start) * 1000
        except AssertionError as e:
            self.elapsed = (time.time() - self.start) * 1000
            self.passed = False
            self.error_msg = str(e)[:300]
        except Exception as e:
            self.elapsed = (time.time() - self.start) * 1000
            self.passed = False
            self.error_msg = f"异常: {e}"

    def info(self, msg: str):
        """记录测试细节"""
        self.details.append(msg)

    def assert_eq(self, actual, expected, desc=""):
        """断言相等 + 记录"""
        ok = actual == expected
        tag = "✓" if ok else "✗"
        msg = f"  {tag} {desc or f'{actual} == {expected}'}"
        if not ok:
            msg += f"  (期望={expected}, 实际={actual})"
        self.details.append(msg)
        assert ok, f"{desc}: 期望={expected}, 实际={actual}"

    def assert_in(self, item, container, desc=""):
        """断言包含 + 记录"""
        ok = item in container
        tag = "✓" if ok else "✗"
        msg = f"  {tag} {desc or f'{item} in {type(container).__name__}'}"
        if ok:
            msg += f"  → 命中: {item[:80] if isinstance(item, str) else item}"
        else:
            msg += f"  → 未命中"
        self.details.append(msg)
        assert ok, f"{desc}: '{item}' 未找到"

    def assert_gt(self, actual, threshold, desc=""):
        """断言大于 + 记录"""
        ok = actual > threshold
        tag = "✓" if ok else "✗"
        msg = f"  {tag} {desc or f'{actual} > {threshold}'}"
        if ok:
            msg += f"  → {actual}"
        else:
            msg += f"  → {actual} <= {threshold}"
        self.details.append(msg)
        assert ok, f"{desc}: {actual} <= {threshold}"

    def assert_true(self, expr, desc=""):
        """断言真 + 记录"""
        ok = bool(expr)
        tag = "✓" if ok else "✗"
        msg = f"  {tag} {desc or str(expr)}"
        self.details.append(msg)
        assert ok, desc or f"表达式为假"


def run_module(module_name: str, desc: str, quiet: bool = False):
    """运行测试模块"""
    import importlib
    mod = importlib.import_module(module_name)

    # 收集测试函数
    test_fns = []
    for attr_name in dir(mod):
        if attr_name.startswith("test_"):
            test_fns.append(getattr(mod, attr_name))

    if not quiet:
        print(f"\n{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}▶ {desc}{RESET}")
        print(f"{CYAN}{'='*60}{RESET}")

    cases = []
    for fn in test_fns:
        tc = TestCase(fn)
        tc.run()
        cases.append(tc)

        if not quiet:
            icon = f"{GREEN}✓{RESET}" if tc.passed else f"{RED}✗{RESET}"
            lines = tc.doc.split("\n")
            title = lines[0]
            print(f"  {icon} {tc.name:<35} {tc.elapsed:>6.0f}ms  {title}")

            # 打印测试条件详情
            for d in tc.details:
                print(d)

            if not tc.passed:
                print(f"       {RED}→ {tc.error_msg}{RESET}")
            print()  # 空行分隔

    passed = sum(1 for c in cases if c.passed)
    total = len(cases)
    if not quiet:
        if passed == total:
            print(f"  {GREEN}结果: {passed}/{total} 通过{RESET}")
        else:
            print(f"  {RED}结果: {passed}/{total} 通过, {total-passed} 失败{RESET}")
    return cases


def run_all(quiet: bool = False):
    """运行全部测试"""
    test_modules = [
        ("tests.test_chunker", "✂️  分块模块 — MinerU blocks 分块器"),
        ("tests.test_retriever", "🔍 检索模块 — BM25 关键词索引"),
        ("tests.test_generator", "🤖 答案生成模块 — deepseek Prompt 构建"),
    ]

    all_cases = []
    if not quiet:
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}    RAG 问答系统 — 测试套件{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")
        print(f"  开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  环境: Python {sys.version.split()[0]}")
        print()

    suite_start = time.time()
    for module_name, desc in test_modules:
        try:
            cases = run_module(module_name, desc, quiet)
            all_cases.extend(cases)
        except ModuleNotFoundError as e:
            print(f"\n  {RED}✗ 模块加载失败: {e}{RESET}")

    suite_elapsed = time.time() - suite_start
    total_passed = sum(1 for c in all_cases if c.passed)
    total = len(all_cases)
    total_failed = total - total_passed

    # ── 汇总报告 ──
    if not quiet:
        print(f"\n{BOLD}{'='*60}{RESET}")
        print(f"{BOLD}   测试报告{RESET}")
        print(f"{BOLD}{'='*60}{RESET}")

        if total_failed == 0:
            print(f"  {GREEN}✅ 全部 {total_passed} 个测试通过{RESET}")
        else:
            print(f"  {RED}❌ {total_passed}/{total} 通过, {total_failed} 失败{RESET}")

        print(f"  总耗时: {suite_elapsed:.2f}s")
        print(f"  平均: {suite_elapsed / max(total, 1) * 1000:.0f}ms/个")

        # 最慢 3 个
        sorted_cases = sorted(all_cases, key=lambda c: c.elapsed, reverse=True)[:3]
        print(f"\n  {YELLOW}最慢测试:{RESET}")
        for tc in sorted_cases:
            warn = " ⚠️" if tc.elapsed > 300 else ""
            print(f"    {tc.name:<35} {tc.elapsed:>6.0f}ms  {tc.doc}{warn}")

        print()

    return total_failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="简洁模式")
    args = parser.parse_args()

    success = run_all(quiet=args.quiet)
    sys.exit(0 if success else 1)
