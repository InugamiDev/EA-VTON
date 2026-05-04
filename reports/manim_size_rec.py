"""
EA-VTON Size Recommendation — Math & Logic Visualization
Run: manim -pqh reports/manim_size_rec.py SizeRecPipeline
"""

from manim import *
import numpy as np

# ── Color palette ──
BRAND = "#1d3f5e"
BRAND_LIGHT = "#7c92aa"
OK = "#20594a"
WARN = "#8b5d22"
BAD = "#8b3d33"
VN_COLOR = "#e74c3c"
US_COLOR = "#3498db"
COPULA_COLOR = "#9b59b6"


class SizeRecPipeline(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"
        self.scene_title()
        self.scene_problem()
        self.scene_two_populations()
        self.scene_independent_vs_copula()
        self.scene_density_ratio()
        self.scene_psis()
        self.scene_temper()
        self.scene_weighted_training()
        self.scene_results()
        self.scene_end()

    # ── Scene 1: Title ──
    def scene_title(self):
        title = Text("EA-VTON", font_size=72, color=WHITE, weight=BOLD)
        subtitle = Text(
            "Size Recommendation Pipeline",
            font_size=36,
            color=BRAND_LIGHT,
        )
        subtitle.next_to(title, DOWN, buff=0.4)
        method = Text(
            "Copula Density Ratio + PSIS + GBM",
            font_size=24,
            color=GREY_B,
        )
        method.next_to(subtitle, DOWN, buff=0.3)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=0.8)
        self.play(FadeIn(method, shift=UP * 0.2), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(Group(title, subtitle, method)), run_time=0.6)

    # ── Scene 2: The problem ──
    def scene_problem(self):
        header = Text("The Problem", font_size=42, color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.6)
        self.play(Write(header), run_time=0.6)

        # Data box
        data_box = RoundedRectangle(
            corner_radius=0.15, width=5, height=2.2, color=US_COLOR
        )
        data_box.shift(LEFT * 3 + DOWN * 0.3)
        data_label = Text("Training Data", font_size=20, color=US_COLOR)
        data_label.next_to(data_box, UP, buff=0.15)
        data_info = VGroup(
            Text("192K US reviews (RTR)", font_size=16, color=GREY_B),
            Text("Height μ = 165.6 cm", font_size=16, color=GREY_B),
            Text("Weight μ = 60.3 kg", font_size=16, color=GREY_B),
        ).arrange(DOWN, buff=0.15)
        data_info.move_to(data_box)

        # Target box
        target_box = RoundedRectangle(
            corner_radius=0.15, width=5, height=2.2, color=VN_COLOR
        )
        target_box.shift(RIGHT * 3 + DOWN * 0.3)
        target_label = Text("Target Users", font_size=20, color=VN_COLOR)
        target_label.next_to(target_box, UP, buff=0.15)
        target_info = VGroup(
            Text("Vietnamese shoppers", font_size=16, color=GREY_B),
            Text("Height μ = 156.2 cm", font_size=16, color=GREY_B),
            Text("Weight μ = 53.9 kg", font_size=16, color=GREY_B),
        ).arrange(DOWN, buff=0.15)
        target_info.move_to(target_box)

        self.play(
            Create(data_box), Write(data_label), FadeIn(data_info),
            run_time=1,
        )
        self.play(
            Create(target_box), Write(target_label), FadeIn(target_info),
            run_time=1,
        )

        # Gap arrow
        gap = Text("≠", font_size=60, color=WARN)
        gap.move_to(ORIGIN + DOWN * 0.3)
        gap_label = Text(
            "Distribution mismatch", font_size=18, color=WARN
        )
        gap_label.next_to(gap, DOWN, buff=0.2)

        self.play(Write(gap), FadeIn(gap_label), run_time=0.8)

        # Consequence
        consequence = Text(
            "Training directly on US data → systematic over-sizing for VN users",
            font_size=18,
            color=BAD,
        )
        consequence.to_edge(DOWN, buff=0.8)
        self.play(FadeIn(consequence, shift=UP * 0.2), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 3: Two populations as distributions ──
    def scene_two_populations(self):
        header = Text(
            "Two Different Populations", font_size=42, color=WHITE, weight=BOLD
        )
        header.to_edge(UP, buff=0.6)
        self.play(Write(header), run_time=0.6)

        # Axes for height distribution
        ax = Axes(
            x_range=[135, 195, 10],
            y_range=[0, 0.1, 0.02],
            x_length=10,
            y_length=4,
            axis_config={"color": GREY_B, "include_numbers": False},
        ).shift(DOWN * 0.5)

        # Add axis number labels manually with Text (no LaTeX)
        x_num_labels = VGroup()
        for val in [140, 150, 160, 170, 180, 190]:
            lbl = Text(str(val), font_size=16, color=GREY_B)
            lbl.next_to(ax.c2p(val, 0), DOWN, buff=0.15)
            x_num_labels.add(lbl)

        x_label = Text("Height (cm)", font_size=18, color=GREY_B)
        x_label.next_to(ax.x_axis, DOWN, buff=0.5)
        y_label = Text("Density", font_size=18, color=GREY_B)
        y_label.next_to(ax.y_axis, LEFT, buff=0.3).shift(UP * 0.5)

        # VN distribution N(156.2, 5.5)
        vn_curve = ax.plot(
            lambda x: (1 / (5.5 * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x - 156.2) / 5.5) ** 2),
            x_range=[135, 180],
            color=VN_COLOR,
        )
        vn_area = ax.get_area(vn_curve, x_range=[140, 172], color=VN_COLOR, opacity=0.2)
        vn_label = Text("VN (μ=156.2, σ=5.5)", font_size=16, color=VN_COLOR)
        vn_label.move_to(ax.c2p(150, 0.08))

        # US distribution N(165.6, 7.5)
        us_curve = ax.plot(
            lambda x: (1 / (7.5 * np.sqrt(2 * np.pi)))
            * np.exp(-0.5 * ((x - 165.6) / 7.5) ** 2),
            x_range=[140, 195],
            color=US_COLOR,
        )
        us_area = ax.get_area(us_curve, x_range=[143, 190], color=US_COLOR, opacity=0.15)
        us_label = Text("US (μ=165.6, σ=7.5)", font_size=16, color=US_COLOR)
        us_label.move_to(ax.c2p(177, 0.06))

        self.play(Create(ax), Write(x_label), Write(y_label), FadeIn(x_num_labels), run_time=1)
        self.play(Create(us_curve), FadeIn(us_area), Write(us_label), run_time=1.2)
        self.play(Create(vn_curve), FadeIn(vn_area), Write(vn_label), run_time=1.2)

        # Show gap
        gap_line = DashedLine(
            ax.c2p(156.2, 0), ax.c2p(156.2, 0.075), color=VN_COLOR, dash_length=0.1
        )
        gap_line2 = DashedLine(
            ax.c2p(165.6, 0), ax.c2p(165.6, 0.055), color=US_COLOR, dash_length=0.1
        )
        gap_arrow = DoubleArrow(
            ax.c2p(156.2, 0.08), ax.c2p(165.6, 0.08), color=WARN,
            buff=0, stroke_width=2, tip_length=0.15,
        )
        gap_text = Text("~9.4 cm gap", font_size=16, color=WARN)
        gap_text.next_to(gap_arrow, UP, buff=0.1)

        self.play(Create(gap_line), Create(gap_line2), run_time=0.6)
        self.play(Create(gap_arrow), Write(gap_text), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 4: Independent vs Copula ──
    def scene_independent_vs_copula(self):
        header = Text(
            "Independent vs Copula Weighting",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Left: independent
        left_title = Text("Independent", font_size=24, color=BRAND_LIGHT)
        left_title.shift(LEFT * 3.5 + UP * 2)

        left_eq = Text(
            "w(h,w) = [Pт(h)/Ps(h)] × [Pт(w)/Ps(w)]",
            font_size=22,
            color=WHITE,
        )
        left_eq.next_to(left_title, DOWN, buff=0.4)

        left_desc = Text(
            "Treats height and weight\nas independent variables",
            font_size=16,
            color=GREY_B,
        )
        left_desc.next_to(left_eq, DOWN, buff=0.4)

        left_problem = Text("Ignores correlation (ρ = 0.46)", font_size=16, color=BAD)
        left_problem.next_to(left_desc, DOWN, buff=0.3)

        left_ess = Text("ESS = 9,458 (11%)", font_size=18, color=BAD)
        left_ess.next_to(left_problem, DOWN, buff=0.3)

        # Right: copula
        right_title = Text("Copula", font_size=24, color=COPULA_COLOR)
        right_title.shift(RIGHT * 3.5 + UP * 2)

        right_eq = Text(
            "w(h,w) = [cт·fт(h)·fт(w)] / [cs·fs(h)·fs(w)]",
            font_size=22,
            color=WHITE,
        )
        right_eq.next_to(right_title, DOWN, buff=0.4)

        right_desc = Text(
            "Models height-weight\njoint distribution",
            font_size=16,
            color=GREY_B,
        )
        right_desc.next_to(right_eq, DOWN, buff=0.4)

        right_good = Text("Captures correlation (ρ = 0.46)", font_size=16, color=OK)
        right_good.next_to(right_desc, DOWN, buff=0.3)

        right_ess = Text("ESS = 20,806 (24%)", font_size=18, color=OK)
        right_ess.next_to(right_good, DOWN, buff=0.3)

        # Divider
        divider = DashedLine(UP * 2.5, DOWN * 2.5, color=GREY_D, dash_length=0.15)

        self.play(Write(left_title), Write(right_title), Create(divider), run_time=0.6)
        self.play(Write(left_eq), Write(right_eq), run_time=1)
        self.play(FadeIn(left_desc), FadeIn(right_desc), run_time=0.8)
        self.play(Write(left_problem), Write(right_good), run_time=0.8)
        self.play(Write(left_ess), Write(right_ess), run_time=0.8)

        # Highlight winner
        winner_box = SurroundingRectangle(
            VGroup(right_title, right_ess),
            color=OK,
            buff=0.3,
            corner_radius=0.1,
        )
        winner_label = Text("2.2× better ESS", font_size=20, color=OK, weight=BOLD)
        winner_label.next_to(winner_box, DOWN, buff=0.2)

        self.play(Create(winner_box), Write(winner_label), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 5: Density ratio explained ──
    def scene_density_ratio(self):
        header = Text(
            "Density Ratio: How It Works",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        eq = Text(
            "w(x) = P_target(x) / P_source(x)",
            font_size=32,
            color=WHITE,
        )
        eq.shift(UP * 1.5)
        self.play(Write(eq), run_time=1)

        # Three examples
        examples = VGroup()
        data = [
            ("155 cm, 50 kg", "Common in VN, rare in US", "w ≈ 8×", OK, "↑ weight"),
            ("160 cm, 55 kg", "Overlap region", "w ≈ 2×", BRAND_LIGHT, "~ weight"),
            ("170 cm, 65 kg", "Common in US, rare in VN", "w ≈ 0.5×", BAD, "↓ weight"),
        ]

        for i, (person, desc, weight, color, arrow) in enumerate(data):
            row = VGroup()
            person_text = Text(person, font_size=20, color=WHITE)
            desc_text = Text(desc, font_size=16, color=GREY_B)
            weight_text = Text(weight, font_size=24, color=color, weight=BOLD)
            arrow_text = Text(arrow, font_size=16, color=color)

            person_text.shift(LEFT * 4)
            desc_text.shift(LEFT * 0.5)
            weight_text.shift(RIGHT * 3)
            arrow_text.shift(RIGHT * 5)

            row.add(person_text, desc_text, weight_text, arrow_text)
            row.shift(DOWN * (0.2 + i * 1.0))
            examples.add(row)

        for row in examples:
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.6)

        meaning = Text(
            "Model pays more attention to samples that look like target users",
            font_size=18,
            color=COPULA_COLOR,
        )
        meaning.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(meaning, shift=UP * 0.2), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 6: PSIS ──
    def scene_psis(self):
        header = Text(
            "PSIS: Smoothing Extreme Weights",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Before PSIS: bar chart with one extreme
        before_label = Text("Before PSIS", font_size=22, color=BAD)
        before_label.shift(LEFT * 3.5 + UP * 1.5)

        before_bars = VGroup()
        raw_weights = [1.0, 1.2, 2.5, 3.0, 8.0, 57.0]
        max_h = 3.0
        for i, w in enumerate(raw_weights):
            h = (w / 57.0) * max_h
            bar = Rectangle(
                width=0.5, height=max(h, 0.06), color=US_COLOR, fill_opacity=0.7
            )
            bar.move_to(LEFT * 5.5 + RIGHT * i * 0.7 + DOWN * 0.3)
            bar.align_to(DOWN * 1.8, DOWN)
            label = Text(f"{w:.0f}×" if w >= 1 else f"{w:.1f}×", font_size=12, color=GREY_B)
            label.next_to(bar, DOWN, buff=0.1)
            before_bars.add(VGroup(bar, label))

        # Extreme weight highlight
        extreme_box = SurroundingRectangle(
            before_bars[-1][0], color=BAD, buff=0.05
        )
        extreme_label = Text("Dangerous!", font_size=14, color=BAD)
        extreme_label.next_to(before_bars[-1][0], UP, buff=0.1)

        self.play(Write(before_label), run_time=0.4)
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in before_bars], lag_ratio=0.1), run_time=1)
        self.play(*[FadeIn(b[1]) for b in before_bars], run_time=0.4)
        self.play(Create(extreme_box), Write(extreme_label), run_time=0.6)

        # After PSIS
        after_label = Text("After PSIS", font_size=22, color=OK)
        after_label.shift(RIGHT * 3.5 + UP * 1.5)

        after_bars = VGroup()
        smoothed_weights = [1.0, 1.2, 2.5, 3.0, 7.5, 18.0]
        for i, w in enumerate(smoothed_weights):
            h = (w / 57.0) * max_h
            bar = Rectangle(
                width=0.5, height=max(h, 0.06), color=OK, fill_opacity=0.7
            )
            bar.move_to(RIGHT * 1.5 + RIGHT * i * 0.7 + DOWN * 0.3)
            bar.align_to(DOWN * 1.8, DOWN)
            label = Text(f"{w:.0f}×" if w >= 1 else f"{w:.1f}×", font_size=12, color=GREY_B)
            label.next_to(bar, DOWN, buff=0.1)
            after_bars.add(VGroup(bar, label))

        self.play(Write(after_label), run_time=0.4)
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in after_bars], lag_ratio=0.1), run_time=1)
        self.play(*[FadeIn(b[1]) for b in after_bars], run_time=0.4)

        # Arrow between
        arrow = Arrow(LEFT * 1.2, RIGHT * 0.8, color=COPULA_COLOR)
        arrow_label = Text("GPD tail\nsmoothing", font_size=14, color=COPULA_COLOR)
        arrow_label.next_to(arrow, UP, buff=0.1)
        self.play(Create(arrow), Write(arrow_label), run_time=0.6)

        # k-hat diagnostic
        khat = Text("k̂ ≈ −0.001 (healthy tail ✓)", font_size=20, color=OK)
        khat.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(khat, shift=UP * 0.2), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 7: Tempering ──
    def scene_temper(self):
        header = Text(
            "Tempering: Final Smoothing",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        eq = Text(
            "w_final = w_PSIS ^ alpha",
            font_size=36,
            color=WHITE,
        )
        eq.shift(UP * 1.5)
        alpha_note = Text("alpha = 0.75", font_size=26, color=COPULA_COLOR)
        alpha_note.next_to(eq, RIGHT, buff=0.5)

        self.play(Write(eq), Write(alpha_note), run_time=1)

        # Examples
        examples = VGroup()
        temper_data = [
            ("57×", "57^{0.75}", "≈ 22×"),
            ("8×", "8^{0.75}", "≈ 4.8×"),
            ("2×", "2^{0.75}", "≈ 1.7×"),
            ("1×", "1^{0.75}", "= 1×"),
        ]

        for i, (before, calc, after) in enumerate(temper_data):
            before_t = Text(before, font_size=24, color=BAD if i == 0 else GREY_B)
            arrow = Text("->", font_size=24, color=GREY_D)
            calc_t = Text(calc, font_size=20, color=GREY_B)
            arrow2 = Text("->", font_size=24, color=GREY_D)
            after_t = Text(after, font_size=24, color=OK if i == 0 else GREY_B)

            row = VGroup(before_t, arrow, calc_t, arrow2, after_t).arrange(RIGHT, buff=0.4)
            row.shift(DOWN * (0.0 + i * 0.7))
            examples.add(row)

        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in examples], lag_ratio=0.2), run_time=1.5)

        note = Text(
            "Keeps the direction of weighting, reduces the extremes",
            font_size=18,
            color=COPULA_COLOR,
        )
        note.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(note, shift=UP * 0.2), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 8: Weighted training ──
    def scene_weighted_training(self):
        header = Text(
            "Weighted GBM Training",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        loss_eq = Text(
            "L = -SUM_i  w(x_i) * SUM_j  y_ij * log(p_ij)",
            font_size=26,
            color=WHITE,
        )
        loss_eq.shift(UP * 1.3)
        self.play(Write(loss_eq), run_time=1.2)

        # Annotate — highlight the whole equation briefly
        w_box = SurroundingRectangle(loss_eq, color=COPULA_COLOR, buff=0.1)
        w_note = Text("w(x_i) = importance weight from copula + PSIS + temper", font_size=14, color=COPULA_COLOR)
        w_note.next_to(w_box, DOWN, buff=0.3)

        self.play(Create(w_box), FadeIn(w_note), run_time=0.8)
        self.wait(1)

        # Training flow
        flow_items = [
            ("86K RTR samples\n+ 6 features", US_COLOR),
            ("Copula weights\nw(xᵢ)", COPULA_COLOR),
            ("LightGBM\n7-class classification", BRAND_LIGHT),
            ("gbm_copula_\ntempered_a075.pkl", OK),
        ]

        boxes = VGroup()
        for i, (text, color) in enumerate(flow_items):
            box = RoundedRectangle(
                corner_radius=0.1, width=2.8, height=1.2, color=color
            )
            label = Text(text, font_size=14, color=WHITE)
            label.move_to(box)
            group = VGroup(box, label)
            boxes.add(group)

        boxes.arrange(RIGHT, buff=0.5)
        boxes.shift(DOWN * 1.3)

        self.play(
            FadeOut(w_box), FadeOut(w_note),
            LaggedStart(*[FadeIn(b, shift=RIGHT * 0.3) for b in boxes], lag_ratio=0.15),
            run_time=1.5,
        )

        # Arrows
        for i in range(len(boxes) - 1):
            arr = Arrow(
                boxes[i].get_right(), boxes[i + 1].get_left(),
                color=GREY_B, buff=0.1, stroke_width=2,
            )
            self.play(Create(arr), run_time=0.3)

        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 9: Results ──
    def scene_results(self):
        header = Text("Results", font_size=42, color=WHITE, weight=BOLD)
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Table header
        col_labels = ["Model", "Full W-1", "VN W-1", "VN Bias", "ESS"]
        rows = [
            ["Baseline (no weight)", "83.37%", "70.15%", "-0.172", "86K"],
            ["Independent", "83.15%", "70.92%", "-0.187", "9.4K"],
            ["Copula+PSIS+Temper", "83.18%", "70.97%", "-0.213", "20.8K"],
        ]

        table_group = VGroup()
        col_widths = [3.5, 1.8, 1.8, 1.5, 1.5]
        x_start = -5.0

        # Header row
        for j, label in enumerate(col_labels):
            x = x_start + sum(col_widths[:j]) + col_widths[j] / 2
            t = Text(label, font_size=16, color=BRAND_LIGHT, weight=BOLD)
            t.move_to([x, 1.5, 0])
            table_group.add(t)

        header_line = Line(
            [x_start, 1.2, 0],
            [x_start + sum(col_widths), 1.2, 0],
            color=GREY_D,
        )
        table_group.add(header_line)

        # Data rows
        row_colors = [GREY_B, GREY_B, OK]
        for i, (row, rc) in enumerate(zip(rows, row_colors)):
            y = 0.6 - i * 0.8
            for j, cell in enumerate(row):
                x = x_start + sum(col_widths[:j]) + col_widths[j] / 2
                color = rc
                if i == 2 and j in [2, 4]:
                    color = OK
                t = Text(cell, font_size=16, color=color)
                t.move_to([x, y, 0])
                table_group.add(t)

        self.play(FadeIn(table_group), run_time=1.5)

        # Highlight row
        highlight = SurroundingRectangle(
            VGroup(*[table_group[k] for k in range(12, 17)]),
            color=OK,
            buff=0.1,
            corner_radius=0.05,
        )
        best_label = Text("Best for VN target", font_size=18, color=OK, weight=BOLD)
        best_label.shift(DOWN * 1.8)

        self.play(Create(highlight), Write(best_label), run_time=0.8)

        # Key takeaways
        takeaways = VGroup(
            Text("VN Within-1: 70.15% → 70.97% (+0.82pp)", font_size=18, color=OK),
            Text("Full test: 83.37% → 83.18% (minimal degradation)", font_size=18, color=GREY_B),
            Text("ESS 2.2× better than independent weighting", font_size=18, color=COPULA_COLOR),
        ).arrange(DOWN, buff=0.2)
        takeaways.shift(DOWN * 2.8)

        self.play(LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in takeaways], lag_ratio=0.2), run_time=1.2)
        self.wait(2.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 10: End ──
    def scene_end(self):
        title = Text("EA-VTON", font_size=60, color=WHITE, weight=BOLD)
        subtitle = Text(
            "Size Recommendation: Copula + PSIS + GBM",
            font_size=28,
            color=COPULA_COLOR,
        )
        subtitle.next_to(title, DOWN, buff=0.4)

        repo = Text(
            "github.com/InugamiDev/EA-VTON",
            font_size=20,
            color=GREY_B,
        )
        repo.next_to(subtitle, DOWN, buff=0.6)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.6)
        self.play(FadeIn(repo, shift=UP * 0.2), run_time=0.4)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
