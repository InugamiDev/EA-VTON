"""
EA-VTON Virtual Try-On — Pipeline & Architecture Visualization
Run: manim -pqh reports/manim_vton.py VTONPipeline
"""

from manim import *
import numpy as np

# ── Color palette ──
BRAND = "#1d3f5e"
BRAND_LIGHT = "#7c92aa"
OK = "#20594a"
WARN = "#8b5d22"
BAD = "#8b3d33"
PERSON_COLOR = "#e67e22"
GARMENT_COLOR = "#27ae60"
WARP_COLOR = "#9b59b6"
COMPOSE_COLOR = "#2980b9"


class VTONPipeline(Scene):
    def construct(self):
        self.camera.background_color = "#0d1117"
        self.scene_title()
        self.scene_overview()
        self.scene_feature_extraction()
        self.scene_tps_warping()
        self.scene_tps_math()
        self.scene_composition()
        self.scene_training_curriculum()
        self.scene_full_system()
        self.scene_end()

    # ── Scene 1: Title ──
    def scene_title(self):
        title = Text("EA-VTON", font_size=72, color=WHITE, weight=BOLD)
        subtitle = Text(
            "Virtual Try-On Pipeline",
            font_size=36,
            color=BRAND_LIGHT,
        )
        subtitle.next_to(title, DOWN, buff=0.4)
        method = Text(
            "TPS Warping + Composition U-Net",
            font_size=24,
            color=GREY_B,
        )
        method.next_to(subtitle, DOWN, buff=0.3)

        self.play(Write(title), run_time=1)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=0.8)
        self.play(FadeIn(method, shift=UP * 0.2), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(Group(title, subtitle, method)), run_time=0.6)

    # ── Scene 2: High-level overview ──
    def scene_overview(self):
        header = Text(
            "What VTON Does", font_size=42, color=WHITE, weight=BOLD
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Person box
        person_box = RoundedRectangle(
            corner_radius=0.15, width=2.5, height=3.2, color=PERSON_COLOR
        )
        person_box.shift(LEFT * 4.5)
        person_icon = Text("🧑", font_size=48)
        person_icon.move_to(person_box).shift(UP * 0.3)
        person_label = Text("Person\nPhoto", font_size=16, color=PERSON_COLOR)
        person_label.move_to(person_box).shift(DOWN * 0.8)

        # Garment box
        garment_box = RoundedRectangle(
            corner_radius=0.15, width=2.5, height=3.2, color=GARMENT_COLOR
        )
        garment_box.shift(LEFT * 1.5)
        garment_icon = Text("👕", font_size=48)
        garment_icon.move_to(garment_box).shift(UP * 0.3)
        garment_label = Text("Garment\nImage", font_size=16, color=GARMENT_COLOR)
        garment_label.move_to(garment_box).shift(DOWN * 0.8)

        # Processing box
        proc_box = RoundedRectangle(
            corner_radius=0.15, width=3.0, height=3.2, color=WARP_COLOR
        )
        proc_box.shift(RIGHT * 1.8)
        proc_label = Text("VTON\nEngine", font_size=22, color=WARP_COLOR, weight=BOLD)
        proc_label.move_to(proc_box)

        # Result box
        result_box = RoundedRectangle(
            corner_radius=0.15, width=2.5, height=3.2, color=OK
        )
        result_box.shift(RIGHT * 5)
        result_icon = Text("🧑‍👕", font_size=48)
        result_icon.move_to(result_box).shift(UP * 0.3)
        result_label = Text("Try-On\nResult", font_size=16, color=OK)
        result_label.move_to(result_box).shift(DOWN * 0.8)

        self.play(
            FadeIn(person_box), FadeIn(person_icon), FadeIn(person_label),
            FadeIn(garment_box), FadeIn(garment_icon), FadeIn(garment_label),
            run_time=1,
        )

        # Arrows
        arr1 = Arrow(person_box.get_right(), proc_box.get_left(), color=GREY_B, buff=0.15)
        arr2 = Arrow(garment_box.get_right(), proc_box.get_left(), color=GREY_B, buff=0.15)
        arr3 = Arrow(proc_box.get_right(), result_box.get_left(), color=OK, buff=0.15)

        self.play(
            Create(arr1), Create(arr2),
            FadeIn(proc_box), FadeIn(proc_label),
            run_time=0.8,
        )
        self.play(
            Create(arr3),
            FadeIn(result_box), FadeIn(result_icon), FadeIn(result_label),
            run_time=0.8,
        )

        note = Text(
            "Engineering pipeline — not a research contribution",
            font_size=18,
            color=WARN,
        )
        note.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(note), run_time=0.6)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 3: Feature extraction ──
    def scene_feature_extraction(self):
        header = Text(
            "Step 1: Feature Extraction",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Person image → three outputs
        person = RoundedRectangle(
            corner_radius=0.1, width=2.2, height=2.8, color=PERSON_COLOR
        )
        person.shift(LEFT * 5 + DOWN * 0.3)
        person_label = Text("Person Photo", font_size=16, color=PERSON_COLOR)
        person_label.next_to(person, UP, buff=0.15)

        # Three feature outputs
        features = [
            ("Pose Heatmaps\n(18 channels)", "#e74c3c", "MediaPipe\nPoseLandmarker"),
            ("Human Parsing\n(20 classes)", "#3498db", "Segmentation\nmap"),
            ("Agnostic Image\n(garment erased)", "#2ecc71", "Person without\noriginal clothing"),
        ]

        feat_boxes = VGroup()
        for i, (name, color, desc) in enumerate(features):
            box = RoundedRectangle(
                corner_radius=0.1, width=2.8, height=2.0, color=color
            )
            box.shift(RIGHT * (-1 + i * 3.3) + DOWN * 0.3)
            title = Text(name, font_size=14, color=color, weight=BOLD)
            title.move_to(box).shift(UP * 0.3)
            subtitle = Text(desc, font_size=12, color=GREY_B)
            subtitle.move_to(box).shift(DOWN * 0.4)
            feat_boxes.add(VGroup(box, title, subtitle))

        self.play(FadeIn(person), Write(person_label), run_time=0.6)

        for feat in feat_boxes:
            arr = Arrow(
                person.get_right(), feat[0].get_left(),
                color=GREY_B, buff=0.1, stroke_width=2,
            )
            self.play(Create(arr), FadeIn(feat), run_time=0.5)

        note = Text(
            "These features feed into both warping and composition stages",
            font_size=16,
            color=GREY_B,
        )
        note.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 4: TPS warping visual ──
    def scene_tps_warping(self):
        header = Text(
            "Step 2: TPS Warping",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        explain = Text(
            "Thin Plate Spline deforms the flat garment to match body shape",
            font_size=18,
            color=GREY_B,
        )
        explain.next_to(header, DOWN, buff=0.3)
        self.play(FadeIn(explain), run_time=0.4)

        # Flat garment grid (5x5)
        flat_grid = VGroup()
        grid_size = 5
        spacing = 0.5
        offset_left = LEFT * 4 + DOWN * 0.5

        for i in range(grid_size):
            for j in range(grid_size):
                dot = Dot(
                    offset_left + RIGHT * j * spacing + UP * i * spacing,
                    radius=0.06,
                    color=GARMENT_COLOR,
                )
                flat_grid.add(dot)

        # Grid lines
        flat_lines = VGroup()
        for i in range(grid_size):
            for j in range(grid_size - 1):
                idx1 = i * grid_size + j
                idx2 = i * grid_size + j + 1
                line = Line(
                    flat_grid[idx1].get_center(),
                    flat_grid[idx2].get_center(),
                    color=GARMENT_COLOR,
                    stroke_width=1.5,
                    stroke_opacity=0.5,
                )
                flat_lines.add(line)
            if i < grid_size - 1:
                for j in range(grid_size):
                    idx1 = i * grid_size + j
                    idx2 = (i + 1) * grid_size + j
                    line = Line(
                        flat_grid[idx1].get_center(),
                        flat_grid[idx2].get_center(),
                        color=GARMENT_COLOR,
                        stroke_width=1.5,
                        stroke_opacity=0.5,
                    )
                    flat_lines.add(line)

        flat_label = Text("Flat garment\n(5×5 control grid)", font_size=14, color=GARMENT_COLOR)
        flat_label.next_to(VGroup(flat_grid, flat_lines), DOWN, buff=0.3)

        self.play(Create(flat_grid), Create(flat_lines), Write(flat_label), run_time=1)

        # Arrow
        warp_arrow = Arrow(LEFT * 1.5 + DOWN * 0.5, RIGHT * 0.5 + DOWN * 0.5, color=WARP_COLOR, buff=0)
        warp_text = Text("ControlPointNet\npredicts offsets", font_size=14, color=WARP_COLOR)
        warp_text.next_to(warp_arrow, UP, buff=0.15)
        self.play(Create(warp_arrow), Write(warp_text), run_time=0.6)

        # Warped grid
        warped_grid = VGroup()
        offset_right = RIGHT * 2.5 + DOWN * 0.5

        # Create warped positions (simulate body curvature)
        warped_positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                base = offset_right + RIGHT * j * spacing + UP * i * spacing
                # Add body-like deformation
                cx, cy = j - 2, i - 2  # center
                dx = -0.08 * cy * np.sin(cx * 0.5)  # horizontal squeeze
                dy = 0.05 * np.sin(cx * 0.7) * (1 + 0.3 * abs(cy))
                pos = base + RIGHT * dx + UP * dy
                warped_positions.append(pos)
                dot = Dot(pos, radius=0.06, color=WARP_COLOR)
                warped_grid.add(dot)

        warped_lines = VGroup()
        for i in range(grid_size):
            for j in range(grid_size - 1):
                idx1 = i * grid_size + j
                idx2 = i * grid_size + j + 1
                line = Line(
                    warped_positions[idx1],
                    warped_positions[idx2],
                    color=WARP_COLOR,
                    stroke_width=1.5,
                    stroke_opacity=0.5,
                )
                warped_lines.add(line)
            if i < grid_size - 1:
                for j in range(grid_size):
                    idx1 = i * grid_size + j
                    idx2 = (i + 1) * grid_size + j
                    line = Line(
                        warped_positions[idx1],
                        warped_positions[idx2],
                        color=WARP_COLOR,
                        stroke_width=1.5,
                        stroke_opacity=0.5,
                    )
                    warped_lines.add(line)

        warped_label = Text("Warped to body shape", font_size=14, color=WARP_COLOR)
        warped_label.next_to(VGroup(warped_grid, warped_lines), DOWN, buff=0.3)

        self.play(Create(warped_grid), Create(warped_lines), Write(warped_label), run_time=1.2)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 5: TPS math ──
    def scene_tps_math(self):
        header = Text(
            "TPS: The Math",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # ControlPointNet
        cpn_title = Text("ControlPointNet", font_size=24, color=WARP_COLOR, weight=BOLD)
        cpn_title.shift(UP * 1.8)

        cpn_eq = Text(
            "delta_p = CNN(concat(F_person, F_garment))",
            font_size=22,
            color=WHITE,
        )
        cpn_eq.next_to(cpn_title, DOWN, buff=0.3)

        cpn_desc = Text(
            "Predicts 25 control point offsets (5×5 grid × 2D)",
            font_size=16,
            color=GREY_B,
        )
        cpn_desc.next_to(cpn_eq, DOWN, buff=0.2)

        self.play(Write(cpn_title), run_time=0.4)
        self.play(Write(cpn_eq), FadeIn(cpn_desc), run_time=0.8)

        # TPS equation
        tps_title = Text("TPS Transformation", font_size=24, color=WARP_COLOR, weight=BOLD)
        tps_title.shift(DOWN * 0.3)

        tps_eq = Text(
            "f(x) = A*x + b + SUM_i  w_i * U(||x - p_i||)",
            font_size=22,
            color=WHITE,
        )
        tps_eq.next_to(tps_title, DOWN, buff=0.3)

        tps_parts = VGroup(
            Text("A·x + b  = affine (global rotation/scale)", font_size=14, color=GREY_B),
            Text("U(r) = r² log(r)  = radial basis function", font_size=14, color=GREY_B),
            Text("wᵢ  = per-control-point deformation weights", font_size=14, color=GREY_B),
        ).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        tps_parts.next_to(tps_eq, DOWN, buff=0.3)

        self.play(Write(tps_title), run_time=0.4)
        self.play(Write(tps_eq), run_time=1)
        self.play(FadeIn(tps_parts), run_time=0.8)

        # Regularization
        reg_eq = Text(
            "L_reg = lambda * ||delta_p||^2",
            font_size=22,
            color=WARN,
        )
        reg_eq.shift(DOWN * 2.5)
        reg_desc = Text(
            "Penalizes large offsets → prevents extreme deformation",
            font_size=14,
            color=WARN,
        )
        reg_desc.next_to(reg_eq, DOWN, buff=0.15)

        self.play(Write(reg_eq), FadeIn(reg_desc), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 6: Composition U-Net ──
    def scene_composition(self):
        header = Text(
            "Step 3: Composition U-Net",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Input channels
        inputs = VGroup()
        input_data = [
            ("Agnostic\n(3 ch)", PERSON_COLOR),
            ("Warped\nGarment\n(3 ch)", GARMENT_COLOR),
            ("Pose\nHeatmaps\n(18 ch)", "#e74c3c"),
        ]

        for i, (name, color) in enumerate(input_data):
            box = RoundedRectangle(
                corner_radius=0.1, width=1.8, height=1.6, color=color
            )
            box.shift(LEFT * 5 + DOWN * 0.3 + DOWN * i * 1.8)
            label = Text(name, font_size=12, color=color)
            label.move_to(box)
            inputs.add(VGroup(box, label))

        self.play(LaggedStart(*[FadeIn(inp) for inp in inputs], lag_ratio=0.15), run_time=1)

        # Concat arrow
        concat_point = LEFT * 2.5 + DOWN * 0.3
        concat_label = Text("concat\n24 ch", font_size=14, color=GREY_B)
        concat_label.move_to(concat_point)

        for inp in inputs:
            arr = Arrow(
                inp[0].get_right(), concat_point + LEFT * 0.5,
                color=GREY_B, buff=0.1, stroke_width=2,
            )
            self.play(Create(arr), run_time=0.2)

        self.play(Write(concat_label), run_time=0.4)

        # U-Net architecture (simplified)
        # Encoder blocks
        encoder_sizes = [64, 128, 256, 512, 512]
        encoder_blocks = VGroup()
        x_pos = -0.5
        for i, size in enumerate(encoder_sizes):
            h = 2.5 - i * 0.35
            w = 0.35
            block = Rectangle(
                width=w, height=h,
                color=COMPOSE_COLOR,
                fill_opacity=0.3 + i * 0.12,
            )
            block.move_to([x_pos + i * 0.5, -0.3, 0])
            encoder_blocks.add(block)

        # Decoder blocks
        decoder_sizes = [512, 256, 128, 64]
        decoder_blocks = VGroup()
        for i, size in enumerate(decoder_sizes):
            h = 1.15 + i * 0.35
            w = 0.35
            block = Rectangle(
                width=w, height=h,
                color=OK,
                fill_opacity=0.3 + (3 - i) * 0.12,
            )
            block.move_to([x_pos + (5 + i) * 0.5, -0.3, 0])
            decoder_blocks.add(block)

        unet_label = Text("U-Net (5-level encoder-decoder)", font_size=14, color=COMPOSE_COLOR)
        unet_label.shift(DOWN * 2.2)

        arr_to_unet = Arrow(
            concat_point + RIGHT * 0.5, encoder_blocks[0].get_left(),
            color=GREY_B, buff=0.1, stroke_width=2,
        )

        self.play(
            Create(arr_to_unet),
            LaggedStart(*[FadeIn(b) for b in encoder_blocks], lag_ratio=0.1),
            LaggedStart(*[FadeIn(b) for b in decoder_blocks], lag_ratio=0.1),
            Write(unet_label),
            run_time=1.5,
        )

        # Skip connections
        for i in range(4):
            skip = CurvedArrow(
                encoder_blocks[i].get_top(),
                decoder_blocks[3 - i].get_top(),
                color=BRAND_LIGHT,
                angle=-0.5,
                stroke_width=1.5,
            )
            self.play(Create(skip), run_time=0.2)

        # Output
        output_box = RoundedRectangle(
            corner_radius=0.1, width=2.0, height=1.6, color=OK
        )
        output_box.shift(RIGHT * 4.5 + DOWN * 0.3)

        output_eq = Text(
            "out = m*warp + (1-m)*render",
            font_size=14,
            color=OK,
        )
        output_eq.move_to(output_box)

        out_label = Text("Final image", font_size=14, color=OK)
        out_label.next_to(output_box, DOWN, buff=0.15)

        arr_out = Arrow(
            decoder_blocks[-1].get_right(), output_box.get_left(),
            color=OK, buff=0.1, stroke_width=2,
        )

        self.play(Create(arr_out), FadeIn(output_box), Write(output_eq), Write(out_label), run_time=0.8)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 7: Training curriculum ──
    def scene_training_curriculum(self):
        header = Text(
            "3-Stage Curriculum Training",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        stages = [
            {
                "name": "Stage 1: Warping Only",
                "epochs": "30 epochs, ~4h on M2",
                "loss": "L = ||Gw - Ggt||_1 + lambda*||dp||^2",
                "desc": "Learn garment → body deformation\nComposition U-Net frozen",
                "color": GARMENT_COLOR,
            },
            {
                "name": "Stage 2: Joint Training",
                "epochs": "50 epochs, ~8h on M2",
                "loss": "L = L1 + L_perceptual + lambda*||dp||^2",
                "desc": "Warping + composition together\nVGG-19 perceptual loss added",
                "color": COMPOSE_COLOR,
            },
            {
                "name": "Stage 3: Adversarial",
                "epochs": "40 epochs, ~6h on M2",
                "loss": "L = L2 + L_GAN",
                "desc": "PatchGAN discriminator\nfor texture realism",
                "color": WARP_COLOR,
            },
        ]

        stage_groups = VGroup()
        for i, s in enumerate(stages):
            box = RoundedRectangle(
                corner_radius=0.1, width=3.5, height=3.5, color=s["color"]
            )

            name = Text(s["name"], font_size=16, color=s["color"], weight=BOLD)
            name.move_to(box).shift(UP * 1.2)

            epochs = Text(s["epochs"], font_size=13, color=GREY_B)
            epochs.next_to(name, DOWN, buff=0.2)

            loss = Text(s["loss"], font_size=16, color=WHITE)
            loss.next_to(epochs, DOWN, buff=0.3)

            desc = Text(s["desc"], font_size=12, color=GREY_B)
            desc.next_to(loss, DOWN, buff=0.3)

            group = VGroup(box, name, epochs, loss, desc)
            stage_groups.add(group)

        stage_groups.arrange(RIGHT, buff=0.3)
        stage_groups.shift(DOWN * 0.3)

        for i, sg in enumerate(stage_groups):
            self.play(FadeIn(sg, shift=UP * 0.3), run_time=0.8)
            if i < len(stage_groups) - 1:
                arr = Arrow(
                    sg[0].get_right(), stage_groups[i + 1][0].get_left(),
                    color=GREY_B, buff=0.1, stroke_width=2,
                )
                self.play(Create(arr), run_time=0.3)

        total = Text(
            "Total: ~18h on M2 CPU, batch_size=4, 11,647 training pairs",
            font_size=16,
            color=BRAND_LIGHT,
        )
        total.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(total), run_time=0.5)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 8: Full system architecture ──
    def scene_full_system(self):
        header = Text(
            "Full EA-VTON System",
            font_size=38,
            color=WHITE,
            weight=BOLD,
        )
        header.to_edge(UP, buff=0.5)
        self.play(Write(header), run_time=0.6)

        # Service boxes
        services = [
            ("Frontend\n(Next.js)", GREY_B, LEFT * 5.5),
            ("API Gateway\n(Express :3001)", BRAND_LIGHT, LEFT * 2.5),
            ("Feature Service\n(:8001)", PERSON_COLOR, RIGHT * 1 + UP * 1.5),
            ("Recommendation\n(:8003)", WARP_COLOR, RIGHT * 1),
            ("VTON Service\n(:8002)", GARMENT_COLOR, RIGHT * 1 + DOWN * 1.5),
        ]

        svc_boxes = []
        for name, color, pos in services:
            box = RoundedRectangle(
                corner_radius=0.1, width=2.8, height=1.2, color=color
            )
            box.move_to(pos + DOWN * 0.3)
            label = Text(name, font_size=13, color=color)
            label.move_to(box)
            svc_boxes.append(VGroup(box, label))

        self.play(
            LaggedStart(*[FadeIn(s) for s in svc_boxes], lag_ratio=0.15),
            run_time=1.5,
        )

        # Arrows
        connections = [
            (0, 1, "request"),
            (1, 2, "photo"),
            (1, 3, "features"),
            (1, 4, "images"),
        ]

        for src, dst, label_text in connections:
            arr = Arrow(
                svc_boxes[src][0].get_right(),
                svc_boxes[dst][0].get_left(),
                color=GREY_B,
                buff=0.1,
                stroke_width=2,
            )
            self.play(Create(arr), run_time=0.3)

        # Results
        results = VGroup()
        result_data = [
            ("Size: M (conf 0.40)", WARP_COLOR),
            ("Measurements: chest 88cm, waist 70cm", BRAND_LIGHT),
            ("Try-on image: 40ms latency", GARMENT_COLOR),
        ]

        for i, (text, color) in enumerate(result_data):
            t = Text(text, font_size=16, color=color)
            t.shift(DOWN * 2.2 + DOWN * i * 0.4)
            results.add(t)

        result_header = Text("Pipeline output (~770ms end-to-end)", font_size=18, color=OK, weight=BOLD)
        result_header.shift(DOWN * 1.7)

        self.play(Write(result_header), run_time=0.4)
        self.play(
            LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in results], lag_ratio=0.15),
            run_time=1,
        )
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)

    # ── Scene 9: End ──
    def scene_end(self):
        title = Text("EA-VTON", font_size=60, color=WHITE, weight=BOLD)

        branches = VGroup(
            Text("Size Recommendation → Copula + PSIS + GBM", font_size=22, color=WARP_COLOR),
            Text("Virtual Try-On → TPS Warping + Composition U-Net", font_size=22, color=GARMENT_COLOR),
        ).arrange(DOWN, buff=0.3)
        branches.next_to(title, DOWN, buff=0.6)

        repo = Text(
            "github.com/InugamiDev/EA-VTON",
            font_size=20,
            color=GREY_B,
        )
        repo.next_to(branches, DOWN, buff=0.6)

        self.play(Write(title), run_time=0.8)
        self.play(FadeIn(branches, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(repo, shift=UP * 0.2), run_time=0.4)
        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)
