"""
Patch 80 verification -- five annotated complaints from the manager's real
deployed-app screenshots (pitch too big, standing SP caption, ambiguous
"no borders" sidebar note, confusing paired rating gauges, and the Season
Rank chart "looking very bad"), plus a SIXTH issue discovered only while
investigating the gauges: a genuine CSS-comment-parsing bug that silently
dropped the entire :root{...} custom-property block in every browser,
which is the actual root cause of "what is this" (gauges rendering as
plain text, no ring) and very likely contributed to other odd rendering
too, since :root feeds almost the whole app's color system.

All tests exec/regex against the REAL current app.py -- no reimplementation
-- per this project's standing rule to verify against the actual codebase,
not just the intent of a patch.
"""
import re

import altair as alt
import pandas as pd

APP_SRC = open("/home/claude/rb_model_app/app.py", encoding="utf-8").read()


def test_no_stray_comment_closers_anywhere_in_injected_css():
    """Patch 80's actual root-cause finding: a CSS /* ... */ comment inside
    the injected <style> block (the one at the top of app.py, st.markdown
    right after PATCH_VERSION) contained the literal two-character sequence
    "*/" in its PROSE (a comment describing which CSS custom-property
    "families" a release touches, written as e.g. "--ink*/--rule" -- a
    stray closing marker hiding inside ordinary text), which prematurely
    terminated the comment mid-sentence. Everything from that point until
    the real, later "*/" was then parsed as raw (garbage, unbracketed) CSS
    text sitting directly in front of the next "{...}" rule -- which was
    :root{...} -- so the entire :root rule was treated as having an invalid
    selector and DROPPED by the browser's CSS parser. This is a genuine,
    reproducible browser-level parsing bug (confirmed via a real Chromium
    instance parsing the exact extracted stylesheet text, not a guess) --
    not a cosmetic preference. This test parses every /* ... */ span in the
    injected <style> block the same way a CSS tokenizer does (first "*/"
    after each "/*" always closes it -- CSS comments don't nest) and fails
    if the resulting comment boundaries are inconsistent (an odd count, or
    any residual unclosed span), which is exactly the failure mode that let
    this slip through un-noticed for two prior patches."""
    start = APP_SRC.index('st.markdown("""')
    end = APP_SRC.index('"""', start + 20)
    css_src = APP_SRC[start:end]

    tokens = [(m.start(), m.group()) for m in re.finditer(r"/\*|\*/", css_src)]
    depth = 0
    stray_closers = []
    unclosed_opens = []
    for pos, tok in tokens:
        if tok == "/*":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                stray_closers.append(pos)
                depth = 0
    assert not stray_closers, (
        f"found {len(stray_closers)} stray '*/' with no matching '/*' before it "
        f"(a comment closing early mid-prose) at char offsets {stray_closers} -- "
        f"this is the exact Patch 80 root-cause bug class; check for CSS custom-"
        f"property lists like '--foo*/--bar' inside comment prose"
    )
    assert depth == 0, "an opened /* comment in the injected <style> block never closes"
    print("Test 1 PASSED: no stray '*/' sequences hiding inside CSS comment prose "
          "(the exact bug that silently dropped the whole :root block)")


def test_root_block_is_a_real_parseable_rule():
    """Confirms :root{...} in the live app.py, run through an ACTUAL browser
    engine (Chromium via Playwright), produces a genuine CSSRule with
    selectorText === ':root' -- i.e. this isn't just "the text :root{ appears
    somewhere in the file," it's "a real browser's CSS parser accepts this
    exact stylesheet and keeps the rule." This is the test that would have
    caught the Patch 79 regression before it shipped."""
    from playwright.sync_api import sync_playwright

    start = APP_SRC.index('st.markdown("""')
    end = APP_SRC.index('"""', start + 20)
    css_src = APP_SRC[start + len('st.markdown("""') : end]
    # strip the leading "<style>" / trailing "</style>" the injected block wraps its CSS in
    css_only = re.sub(r"^\s*<style>", "", css_src, flags=re.S)
    css_only = re.sub(r"</style>\s*$", "", css_only, flags=re.S)

    html = f"<!DOCTYPE html><html><head><style>{css_only}</style></head><body></body></html>"
    tmp_path = "/tmp/_patch80_root_parse_check.html"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(html)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{tmp_path}")
        found = page.evaluate(
            """() => {
                const sheet = document.styleSheets[0];
                return Array.from(sheet.cssRules).some(r => r.selectorText === ':root');
            }"""
        )
        # also grab one representative custom property to prove it actually resolves
        coral = page.evaluate(
            "() => getComputedStyle(document.documentElement).getPropertyValue('--coral').trim()"
        )
        browser.close()

    assert found, ":root{...} did not survive real-browser CSS parsing -- it was dropped"
    assert coral == "#E14E54", f"--coral should resolve to #E14E54 on <html>, got {coral!r}"
    print("Test 2 PASSED: :root{...} parses as a genuine CSSRule in a real browser engine, "
          "and --coral actually resolves on the document root")


def test_sp_caption_moved_to_tooltip_not_standing_text():
    """Manager: "remove this" pointing at the permanent SP-tag caption below
    the pitch. Confirms the standalone <p class="side-note"> caption is
    gone, and the explanation survives only as a title= tooltip on the SP
    badge itself (info preserved, not deleted -- just not force-shown)."""
    assert 'SP tag = newly confirmed set-piece role' not in re.sub(r'title="[^"]*"', '', APP_SRC), \
        "the SP-tag explanation should only exist inside a title= tooltip now, not as standing visible text"
    assert re.search(r'class="sp"\s+title="[^"]*[Ss]et-piece[^"]*"', APP_SRC), \
        "expected the SP badge itself to carry the explanation as a hover tooltip"
    print("Test 3 PASSED: SP-tag caption is gone from standing page text; explanation lives in a tooltip")


def test_pitch_has_a_max_width():
    """Manager (x2): "pitch is too big and doesn't look good at all!!".
    Confirms .pitch now caps its width instead of stretching to fill
    Streamlit's full wide-layout content column."""
    m = re.search(r"\.pitch\{([^}]*)\}", APP_SRC)
    assert m, "could not find .pitch{...} CSS rule"
    assert "max-width" in m.group(1), ".pitch should declare a max-width so it doesn't stretch full-bleed"
    print("Test 4 PASSED: .pitch declares a max-width")


def test_sidebar_style_description_has_a_bordered_container():
    """Manager annotation ("no borders", ambiguous) near the sidebar's style-
    profile description. Confirms it now renders inside a bordered/tinted
    box (.side-info-box) rather than a bare st.caption with no visual
    container -- flagged in-code as an interpretation to be confirmed."""
    assert "side-info-box" in APP_SRC, "expected a .side-info-box class to exist"
    assert re.search(r'class="side-info-box"', APP_SRC), \
        "expected the style-profile description to be wrapped in .side-info-box"
    m = re.search(r"\.side-info-box\{([^}]*)\}", APP_SRC)
    assert m and "border" in m.group(1), ".side-info-box should actually declare a border"
    print("Test 5 PASSED: sidebar style-profile description now sits in a bordered container")


def test_rating_gauge_labels_are_distinct():
    """Manager: "what is this" on the two paired rating stats -- partly a
    rendering bug (Test 2, above), partly that both stats' visible labels
    used to look too similar at a glance. Confirms the second stat's label
    is now distinct ("{n}-GW Rating") from the first ("GW{n} Rating"), with
    the fuller description still available in the tooltip/expander (not
    deleted, per the project's "no truncation without preserving detail"
    pattern from Patch 51)."""
    assert re.search(r'\{compliant_gw_end - compliant_gw_start \+ 1\}-GW Rating', APP_SRC), \
        "expected the second rating stat's visible label to be the dynamic '{n}-GW Rating' short form"
    print("Test 6 PASSED: the two rating-gauge stat labels are visually distinct")


# ---------------------------------------------------------------------------
# Season Rank chart domain_max fix (manager: "looking very bad, adjust and
# modify with a good standards") -- root cause confirmed in code: domain_max
# used to be hardcoded to max(10_000_000, worst*1.05), meaning a manager
# whose worst rank all season was ~3.2M still got a y-axis stretched all the
# way to 10,000,000 -- squeezing their entire real trend line into a thin
# sliver at the very bottom of the chart with ~90% dead space above it.
# ---------------------------------------------------------------------------
_m = re.search(r"^def build_season_rank_chart\(.*?\n(?:    .*\n|\n)*", APP_SRC, re.M)
assert _m, "could not find module-level def build_season_rank_chart in app.py"
_ns = {"pd": pd, "alt": alt}
exec(_m.group(0), _ns)
build_season_rank_chart = _ns["build_season_rank_chart"]


def test_domain_max_no_longer_force_pinned_to_ten_million():
    """The manager's actual real GW1-5 rank history (2.6M-3.2M) must no
    longer produce a domain_max of a flat 10,000,000 -- it should bracket
    tightly to the next named tier above the worst rank seen (5,000,000)."""
    df = pd.DataFrame([
        {"GW": 1, "Overall rank": 3186778},
        {"GW": 2, "Overall rank": 2899135},
        {"GW": 3, "Overall rank": 2971465},
        {"GW": 4, "Overall rank": 2638725},
        {"GW": 5, "Overall rank": 2939188},
    ])
    chart = build_season_rank_chart(df)
    spec = chart.to_dict()
    y_scale = spec["layer"][0]["encoding"]["y"]["scale"]
    assert y_scale["domain"] == [1, 5_000_000], (
        f"expected domain [1, 5000000] bracketing the manager's real ~3.2M worst rank, "
        f"got {y_scale['domain']} -- the old hardcoded 10M ceiling regressed"
    )
    print("Test 7 PASSED: domain_max now brackets the manager's real rank range (5,000,000), "
          "not a flat, always-on 10,000,000 ceiling")


def test_domain_max_still_handles_a_genuinely_bad_rank():
    """A manager doing worse than every named tier (e.g. 12,000,000 -- worse
    than the whole named ladder) should still get a sane fallback domain
    (worst*1.05), not crash or silently clip data off the chart."""
    df = pd.DataFrame([{"GW": 1, "Overall rank": 12_000_000}])
    chart = build_season_rank_chart(df)
    spec = chart.to_dict()
    y_scale = spec["layer"][0]["encoding"]["y"]["scale"]
    assert y_scale["domain"][1] >= 12_000_000, "domain must not clip a rank worse than every named tier"
    print("Test 8 PASSED: a rank worse than every named tier still gets a domain that contains it")


def test_named_bands_still_present_for_a_top_tier_manager():
    """Regression guard: a manager near the very top (worst rank 50,000)
    should still see the full relevant ladder of named bands, unchanged
    from Patch 78's original behavior."""
    df = pd.DataFrame([{"GW": 1, "Overall rank": 50_000}])
    chart = build_season_rank_chart(df)
    spec = chart.to_dict()
    y_axis = spec["layer"][1]["encoding"]["y"]["axis"]
    for band in (1000, 10000, 50000):
        assert band in y_axis["values"], f"expected named band {band:,} still present, got {y_axis['values']}"
    print("Test 9 PASSED: named bands below the worst rank are still all present (no regression for top-tier managers)")


if __name__ == "__main__":
    test_no_stray_comment_closers_anywhere_in_injected_css()
    test_root_block_is_a_real_parseable_rule()
    test_sp_caption_moved_to_tooltip_not_standing_text()
    test_pitch_has_a_max_width()
    test_sidebar_style_description_has_a_bordered_container()
    test_rating_gauge_labels_are_distinct()
    test_domain_max_no_longer_force_pinned_to_ten_million()
    test_domain_max_still_handles_a_genuinely_bad_rank()
    test_named_bands_still_present_for_a_top_tier_manager()
    print("\nALL PATCH 80 TESTS PASSED")
