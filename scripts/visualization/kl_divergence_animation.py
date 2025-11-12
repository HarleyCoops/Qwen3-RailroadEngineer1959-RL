#!/usr/bin/env python3
"""
Epic Manim Animation: KL Divergence Analysis for Dakota Grammar RL Training

This script creates a visually stunning animation showing KL divergence metrics
over training, demonstrating policy stability and adaptation.

Usage:
    manim -pql scripts/visualization/kl_divergence_animation.py KLDivergenceEpic
    manim -pqh scripts/visualization/kl_divergence_animation.py KLDivergenceEpic  # High quality

Requirements:
    pip install manim pandas numpy
"""

from manim import *
import pandas as pd
import numpy as np
from pathlib import Path


class KLDivergenceEpic(Scene):
    """Epic visualization of KL divergence metrics during RL training."""
    
    def construct(self):
        # Load data
        csv_path = Path("wandb_analysis/kl_divergence_curve.csv")
        if not csv_path.exists():
            raise FileNotFoundError(f"Data file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Extract key metrics
        steps = df['_step'].values
        masked_kl_mean = df['masked_mismatch_kl/mean'].values
        overall_kl_mean = df['mismatch_kl/mean'].values
        unmasked_kl_mean = df['unmasked_mismatch_kl/mean'].values
        
        # Normalize steps to 0-1 for animation
        max_step = steps.max()
        normalized_steps = steps / max_step
        
        # Create axes
        axes = Axes(
            x_range=[0, max_step, 200],
            y_range=[0, 12, 2],
            x_length=10,
            y_length=6,
            axis_config={"color": BLUE, "include_numbers": True},
            tips=False,
        )
        
        # Title
        title = Text(
            "Policy Stability: KL Divergence Analysis",
            font_size=48,
            color=YELLOW
        ).to_edge(UP)
        
        subtitle = Text(
            "Dakota Grammar RL Training - Qwen3-0.6B",
            font_size=24,
            color=GRAY
        ).next_to(title, DOWN, buff=0.3)
        
        # Create curves
        masked_curve = axes.plot_line_graph(
            x_values=steps,
            y_values=masked_kl_mean,
            add_vertex_dots=False,
            line_color=RED,
            stroke_width=3
        )
        
        overall_curve = axes.plot_line_graph(
            x_values=steps,
            y_values=overall_kl_mean,
            add_vertex_dots=False,
            line_color=ORANGE,
            stroke_width=3
        )
        
        unmasked_curve = axes.plot_line_graph(
            x_values=steps,
            y_values=unmasked_kl_mean * 100,  # Scale for visibility
            add_vertex_dots=False,
            line_color=GREEN,
            stroke_width=3
        )
        
        # Labels
        masked_label = Text("Masked KL (Mean)", font_size=24, color=RED).to_corner(UR, buff=0.5)
        overall_label = Text("Overall KL (Mean)", font_size=24, color=ORANGE).next_to(masked_label, DOWN, buff=0.2)
        unmasked_label = Text("Unmasked KL ×100", font_size=24, color=GREEN).next_to(overall_label, DOWN, buff=0.2)
        
        # Legend
        legend = VGroup(masked_label, overall_label, unmasked_label)
        
        # Animation sequence
        self.play(
            Write(title),
            Write(subtitle),
            run_time=2
        )
        self.wait(1)
        
        self.play(
            Create(axes),
            run_time=2
        )
        self.wait(0.5)
        
        # Animate curves appearing
        self.play(
            Create(masked_curve),
            run_time=3
        )
        self.play(
            Write(masked_label),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Create(overall_curve),
            run_time=3
        )
        self.play(
            Write(overall_label),
            run_time=1
        )
        self.wait(0.5)
        
        self.play(
            Create(unmasked_curve),
            run_time=3
        )
        self.play(
            Write(unmasked_label),
            run_time=1
        )
        self.wait(1)
        
        # Highlight key statistics
        final_masked = masked_kl_mean[-1]
        final_overall = overall_kl_mean[-1]
        final_unmasked = unmasked_kl_mean[-1]
        
        stats_text = VGroup(
            Text(f"Final Masked KL: {final_masked:.2f}", font_size=28, color=RED),
            Text(f"Final Overall KL: {final_overall:.2f}", font_size=28, color=ORANGE),
            Text(f"Final Unmasked KL: {final_unmasked:.4f}", font_size=28, color=GREEN),
        ).arrange(DOWN, buff=0.3).to_corner(UL, buff=0.5)
        
        self.play(
            Write(stats_text),
            run_time=2
        )
        self.wait(2)
        
        # Add interpretation text
        interpretation = Text(
            "Policy adapted for Dakota patterns while\npreserving general language understanding",
            font_size=20,
            color=WHITE
        ).to_edge(DOWN, buff=0.5)
        
        self.play(
            Write(interpretation),
            run_time=2
        )
        self.wait(3)
        
        # Fade out
        self.play(
            FadeOut(VGroup(*self.mobjects)),
            run_time=2
        )


class KLDivergenceMultiPanel(Scene):
    """Multi-panel view showing different KL divergence metrics."""
    
    def construct(self):
        csv_path = Path("wandb_analysis/kl_divergence_curve.csv")
        df = pd.read_csv(csv_path)
        
        steps = df['_step'].values
        max_step = steps.max()
        
        # Title
        title = Text(
            "KL Divergence: Policy Adaptation Analysis",
            font_size=42,
            color=YELLOW
        ).to_edge(UP)
        
        # Create three panels
        panel1_axes = Axes(
            x_range=[0, max_step, 200],
            y_range=[0, 12, 2],
            x_length=4,
            y_length=2.5,
            axis_config={"color": BLUE, "font_size": 16},
        ).shift(LEFT * 3.5 + UP * 0.5)
        
        panel2_axes = Axes(
            x_range=[0, max_step, 200],
            y_range=[0, 5, 1],
            x_length=4,
            y_length=2.5,
            axis_config={"color": BLUE, "font_size": 16},
        ).shift(UP * 0.5)
        
        panel3_axes = Axes(
            x_range=[0, max_step, 200],
            y_range=[0, 0.2, 0.05],
            x_length=4,
            y_length=2.5,
            axis_config={"color": BLUE, "font_size": 16},
        ).shift(RIGHT * 3.5 + UP * 0.5)
        
        # Panel 1: Masked KL
        masked_curve = panel1_axes.plot_line_graph(
            x_values=steps,
            y_values=df['masked_mismatch_kl/mean'].values,
            line_color=RED,
            stroke_width=2
        )
        panel1_label = Text("Masked KL", font_size=20, color=RED).next_to(panel1_axes, DOWN, buff=0.2)
        
        # Panel 2: Overall KL
        overall_curve = panel2_axes.plot_line_graph(
            x_values=steps,
            y_values=df['mismatch_kl/mean'].values,
            line_color=ORANGE,
            stroke_width=2
        )
        panel2_label = Text("Overall KL", font_size=20, color=ORANGE).next_to(panel2_axes, DOWN, buff=0.2)
        
        # Panel 3: Unmasked KL
        unmasked_curve = panel3_axes.plot_line_graph(
            x_values=steps,
            y_values=df['unmasked_mismatch_kl/mean'].values,
            line_color=GREEN,
            stroke_width=2
        )
        panel3_label = Text("Unmasked KL", font_size=20, color=GREEN).next_to(panel3_axes, DOWN, buff=0.2)
        
        # Animate
        self.play(Write(title), run_time=1)
        self.wait(0.5)
        
        self.play(
            Create(panel1_axes),
            Create(masked_curve),
            Write(panel1_label),
            run_time=2
        )
        
        self.play(
            Create(panel2_axes),
            Create(overall_curve),
            Write(panel2_label),
            run_time=2
        )
        
        self.play(
            Create(panel3_axes),
            Create(unmasked_curve),
            Write(panel3_label),
            run_time=2
        )
        
        self.wait(3)


class KLDivergence3D(ThreeDScene):
    """3D visualization of KL divergence as a trajectory through policy space."""
    
    def construct(self):
        csv_path = Path("wandb_analysis/kl_divergence_curve.csv")
        df = pd.read_csv(csv_path)
        
        steps = df['_step'].values
        masked_kl = df['masked_mismatch_kl/mean'].values
        overall_kl = df['mismatch_kl/mean'].values
        unmasked_kl = df['unmasked_mismatch_kl/mean'].values
        
        # Normalize steps to 0-10 for better visualization
        normalized_steps = (steps / steps.max()) * 10
        
        # Set up 3D axes
        # X: Masked KL, Y: Overall KL, Z: Unmasked KL (scaled)
        axes = ThreeDAxes(
            x_range=[0, 12, 2],  # Masked KL range
            y_range=[0, 5, 1],   # Overall KL range
            z_range=[0, 0.2, 0.05],  # Unmasked KL range
            x_length=8,
            y_length=6,
            z_length=4,
            axis_config={"color": BLUE, "include_numbers": True},
        )
        
        # Create 3D trajectory: each point is (masked_kl, overall_kl, unmasked_kl)
        # This shows how the policy moves through KL divergence space
        trajectory_points = [
            axes.coords_to_point(m, o, u)
            for m, o, u in zip(masked_kl[::5], overall_kl[::5], unmasked_kl[::5])
        ]
        
        # Create smooth 3D curve
        trajectory_3d = VMobject().set_points_as_corners(trajectory_points)
        trajectory_3d.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN)
        trajectory_3d.set_stroke(width=4)
        
        # Add dots at key points
        start_dot = Dot3D(
            point=axes.coords_to_point(masked_kl[0], overall_kl[0], unmasked_kl[0]),
            color=GREEN,
            radius=0.15
        )
        
        end_dot = Dot3D(
            point=axes.coords_to_point(masked_kl[-1], overall_kl[-1], unmasked_kl[-1]),
            color=RED,
            radius=0.15
        )
        
        # Axis labels
        x_label = axes.get_x_axis_label("Masked KL", direction=RIGHT, buff=0.3)
        y_label = axes.get_y_axis_label("Overall KL", direction=UP, buff=0.3)
        z_label = axes.get_z_axis_label("Unmasked KL", direction=OUT, buff=0.3)
        
        # Title
        title = Text("3D Policy Trajectory in KL Divergence Space", font_size=36, color=YELLOW).to_edge(UP)
        subtitle = Text(
            "X: Masked KL | Y: Overall KL | Z: Unmasked KL",
            font_size=20,
            color=GRAY
        ).next_to(title, DOWN, buff=0.2)
        
        # Legend
        legend_start = Text("Start", font_size=18, color=GREEN).to_corner(UL, buff=0.5)
        legend_end = Text("End", font_size=18, color=RED).next_to(legend_start, DOWN, buff=0.2)
        
        # Set camera orientation
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        
        # Animate
        self.play(Write(title), Write(subtitle), run_time=1.5)
        self.wait(0.5)
        
        self.play(Create(axes), run_time=2)
        self.play(
            Write(x_label),
            Write(y_label),
            Write(z_label),
            run_time=1.5
        )
        
        # Show start point
        self.play(Create(start_dot), Write(legend_start), run_time=1)
        
        # Animate trajectory drawing
        self.play(
            Create(trajectory_3d),
            run_time=6,
            rate_func=linear
        )
        
        # Show end point
        self.play(Create(end_dot), Write(legend_end), run_time=1)
        self.wait(1)
        
        # Rotate camera to show 3D structure
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(8)
        self.stop_ambient_camera_rotation()
        
        # Final view
        self.move_camera(phi=70 * DEGREES, theta=45 * DEGREES)
        self.wait(2)


class KLDivergence3DSurface(ThreeDScene):
    """3D surface showing KL divergence landscape over time."""
    
    def construct(self):
        csv_path = Path("wandb_analysis/kl_divergence_curve.csv")
        df = pd.read_csv(csv_path)
        
        steps = df['_step'].values
        masked_kl = df['masked_mismatch_kl/mean'].values
        overall_kl = df['mismatch_kl/mean'].values
        unmasked_kl = df['unmasked_mismatch_kl/mean'].values
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[0, 1000, 200],  # Steps
            y_range=[0, 12, 2],      # Masked KL
            z_range=[0, 5, 1],       # Overall KL
            x_length=8,
            y_length=6,
            z_length=4,
        )
        
        # Create surface: z = overall_kl, y = masked_kl, x = steps
        # This creates a 3D surface showing the relationship
        surface = Surface(
            lambda u, v: axes.coords_to_point(
                u * 1000,
                np.interp(u * 1000, steps, masked_kl),
                np.interp(u * 1000, steps, overall_kl)
            ),
            u_range=[0, 1],
            v_range=[0, 1],
            resolution=(50, 50),
            fill_opacity=0.5,
            fill_color=BLUE,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        
        # Create trajectory line on the surface
        trajectory_points = [
            axes.coords_to_point(
                s,
                np.interp(s, steps, masked_kl),
                np.interp(s, steps, overall_kl)
            )
            for s in steps[::10]
        ]
        trajectory = VMobject().set_points_as_corners(trajectory_points)
        trajectory.set_color(RED).set_stroke(width=4)
        
        title = Text("KL Divergence Surface", font_size=36, color=YELLOW).to_edge(UP)
        
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        
        self.play(Write(title), run_time=1)
        self.play(Create(axes), run_time=2)
        self.play(Create(surface), run_time=3)
        self.play(Create(trajectory), run_time=3)
        
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(6)
        self.stop_ambient_camera_rotation()
        
        self.wait(2)


class KLDivergence3DTrajectory(ThreeDScene):
    """Enhanced 3D trajectory showing policy evolution through KL space."""
    
    def construct(self):
        csv_path = Path("wandb_analysis/kl_divergence_curve.csv")
        df = pd.read_csv(csv_path)
        
        steps = df['_step'].values
        masked_kl = df['masked_mismatch_kl/mean'].values
        overall_kl = df['mismatch_kl/mean'].values
        unmasked_kl = df['unmasked_mismatch_kl/mean'].values
        
        # Create 3D axes with proper ranges
        axes = ThreeDAxes(
            x_range=[0, 12, 2],      # Masked KL
            y_range=[0, 5, 1],       # Overall KL  
            z_range=[0, 0.2, 0.05],  # Unmasked KL
            x_length=9,
            y_length=7,
            z_length=5,
            axis_config={"color": BLUE, "include_numbers": True},
        )
        
        # Create trajectory with color gradient based on step
        trajectory_points = []
        colors = []
        for i, (m, o, u) in enumerate(zip(masked_kl[::2], overall_kl[::2], unmasked_kl[::2])):
            point = axes.coords_to_point(m, o, u)
            trajectory_points.append(point)
            # Color gradient: green (start) -> yellow -> orange -> red (end)
            t = i / len(masked_kl[::2])
            if t < 0.33:
                colors.append(GREEN)
            elif t < 0.66:
                colors.append(YELLOW)
            else:
                colors.append(RED)
        
        # Create trajectory as connected segments with colors
        trajectory_segments = VGroup()
        for i in range(len(trajectory_points) - 1):
            segment = Line3D(
                start=trajectory_points[i],
                end=trajectory_points[i + 1],
                color=colors[i],
                stroke_width=3
            )
            trajectory_segments.add(segment)
        
        # Add spheres at key milestones
        milestones = [0, len(steps)//4, len(steps)//2, 3*len(steps)//4, len(steps)-1]
        milestone_dots = VGroup()
        milestone_labels = VGroup()
        
        for idx in milestones:
            m, o, u = masked_kl[idx], overall_kl[idx], unmasked_kl[idx]
            dot = Sphere(
                center=axes.coords_to_point(m, o, u),
                radius=0.2,
                color=colors[idx // 2] if idx // 2 < len(colors) else RED,
                resolution=(8, 8)
            )
            milestone_dots.add(dot)
            
            # Label with step number
            label = Text(
                f"Step {steps[idx]}",
                font_size=16,
                color=WHITE
            ).move_to(axes.coords_to_point(m, o, u) + OUT * 0.5)
            milestone_labels.add(label)
        
        # Axis labels
        x_label = axes.get_x_axis_label(
            Tex("Masked KL Divergence"),
            direction=RIGHT,
            buff=0.4
        )
        y_label = axes.get_y_axis_label(
            Tex("Overall KL Divergence"),
            direction=UP,
            buff=0.4
        )
        z_label = axes.get_z_axis_label(
            Tex("Unmasked KL Divergence"),
            direction=OUT,
            buff=0.4
        )
        
        # Title
        title = Text(
            "Policy Evolution in 3D KL Divergence Space",
            font_size=40,
            color=YELLOW,
            weight=BOLD
        ).to_edge(UP)
        
        subtitle = Text(
            "Trajectory shows how policy adapts: High masked KL, moderate overall, low unmasked",
            font_size=18,
            color=GRAY
        ).next_to(title, DOWN, buff=0.3)
        
        # Set initial camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=45 * DEGREES)
        
        # Animate
        self.play(Write(title), Write(subtitle), run_time=2)
        self.wait(1)
        
        self.play(Create(axes), run_time=2)
        self.play(
            Write(x_label),
            Write(y_label),
            Write(z_label),
            run_time=2
        )
        
        # Animate trajectory drawing
        self.play(
            *[Create(seg) for seg in trajectory_segments],
            run_time=8,
            rate_func=linear
        )
        
        # Add milestone markers
        self.play(
            *[Create(dot) for dot in milestone_dots],
            *[Write(label) for label in milestone_labels],
            run_time=3
        )
        
        self.wait(2)
        
        # Rotate camera to show 3D structure
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(10)
        self.stop_ambient_camera_rotation()
        
        # Final static view
        self.move_camera(phi=70 * DEGREES, theta=45 * DEGREES)
        
        # Add summary text
        summary = Text(
            "Policy adapts significantly for Dakota (high masked KL)\n"
            "while preserving general language (low unmasked KL)",
            font_size=20,
            color=WHITE
        ).to_edge(DOWN, buff=0.5)
        
        self.play(Write(summary), run_time=2)
        self.wait(3)


class KLDivergenceEpicComplete(Scene):
    """Complete epic visualization with all features."""
    
    def construct(self):
        csv_path = Path("wandb_analysis/kl_divergence_curve.csv")
        df = pd.read_csv(csv_path)
        
        steps = df['_step'].values
        masked_kl_mean = df['masked_mismatch_kl/mean'].values
        overall_kl_mean = df['mismatch_kl/mean'].values
        unmasked_kl_mean = df['unmasked_mismatch_kl/mean'].values
        
        # SCENE 1: Title and Introduction
        title = Text(
            "Policy Stability Analysis",
            font_size=56,
            color=YELLOW,
            weight=BOLD
        )
        subtitle = Text(
            "KL Divergence Metrics During Dakota Grammar RL Training",
            font_size=32,
            color=GRAY
        ).next_to(title, DOWN, buff=0.5)
        
        model_info = Text(
            "Model: Qwen3-0.6B | Training Steps: 1,000 | Samples: 256,000",
            font_size=20,
            color=LIGHT_GRAY
        ).next_to(subtitle, DOWN, buff=0.8)
        
        self.play(
            FadeIn(title, shift=UP),
            FadeIn(subtitle, shift=UP),
            FadeIn(model_info, shift=UP),
            run_time=2
        )
        self.wait(2)
        self.play(
            FadeOut(title),
            FadeOut(subtitle),
            FadeOut(model_info),
            run_time=1
        )
        
        # SCENE 2: Main visualization
        axes = Axes(
            x_range=[0, 1000, 200],
            y_range=[0, 12, 2],
            x_length=11,
            y_length=6.5,
            axis_config={
                "color": BLUE,
                "include_numbers": True,
                "font_size": 20,
            },
            x_axis_config={"label": "Training Step"},
            y_axis_config={"label": "KL Divergence"},
            tips=False,
        )
        
        # Add axis labels
        x_label = axes.get_x_axis_label("Training Step", direction=DOWN, buff=0.3)
        y_label = axes.get_y_axis_label("KL Divergence", direction=LEFT, buff=0.3)
        
        # Create smooth curves using interpolation
        from scipy.interpolate import interp1d
        try:
            f_masked = interp1d(steps, masked_kl_mean, kind='cubic')
            f_overall = interp1d(steps, overall_kl_mean, kind='cubic')
            f_unmasked = interp1d(steps, unmasked_kl_mean * 100, kind='cubic')
            
            smooth_steps = np.linspace(steps[0], steps[-1], 500)
            smooth_masked = f_masked(smooth_steps)
            smooth_overall = f_overall(smooth_steps)
            smooth_unmasked = f_unmasked(smooth_steps)
        except:
            # Fallback if scipy not available
            smooth_steps = steps
            smooth_masked = masked_kl_mean
            smooth_overall = overall_kl_mean
            smooth_unmasked = unmasked_kl_mean * 100
        
        # Create curves with gradient effect
        masked_curve = axes.plot(
            lambda x: np.interp(x, smooth_steps, smooth_masked),
            x_range=[0, 1000],
            color=RED,
            stroke_width=4
        )
        
        overall_curve = axes.plot(
            lambda x: np.interp(x, smooth_steps, smooth_overall),
            x_range=[0, 1000],
            color=ORANGE,
            stroke_width=4
        )
        
        unmasked_curve = axes.plot(
            lambda x: np.interp(x, smooth_steps, smooth_unmasked),
            x_range=[0, 1000],
            color=GREEN,
            stroke_width=4
        )
        
        # Create axes and labels
        self.play(Create(axes), run_time=1.5)
        self.play(Write(x_label), Write(y_label), run_time=1)
        
        # Animate curves drawing
        self.play(
            Create(masked_curve),
            run_time=4,
            rate_func=linear
        )
        
        masked_label = Text("Masked KL (Mean)", font_size=24, color=RED)
        masked_value = Text(f"Final: {masked_kl_mean[-1]:.2f}", font_size=20, color=RED)
        masked_group = VGroup(masked_label, masked_value).arrange(DOWN, buff=0.1).to_corner(UR, buff=0.5)
        
        self.play(Write(masked_group), run_time=1)
        self.wait(0.5)
        
        self.play(
            Create(overall_curve),
            run_time=4,
            rate_func=linear
        )
        
        overall_label = Text("Overall KL (Mean)", font_size=24, color=ORANGE)
        overall_value = Text(f"Final: {overall_kl_mean[-1]:.2f}", font_size=20, color=ORANGE)
        overall_group = VGroup(overall_label, overall_value).arrange(DOWN, buff=0.1).next_to(masked_group, DOWN, buff=0.3)
        
        self.play(Write(overall_group), run_time=1)
        self.wait(0.5)
        
        self.play(
            Create(unmasked_curve),
            run_time=4,
            rate_func=linear
        )
        
        unmasked_label = Text("Unmasked KL ×100", font_size=24, color=GREEN)
        unmasked_value = Text(f"Final: {unmasked_kl_mean[-1]:.4f}", font_size=20, color=GREEN)
        unmasked_group = VGroup(unmasked_label, unmasked_value).arrange(DOWN, buff=0.1).next_to(overall_group, DOWN, buff=0.3)
        
        self.play(Write(unmasked_group), run_time=1)
        self.wait(1)
        
        # SCENE 3: Key insights
        insight_box = Rectangle(
            width=10,
            height=2,
            color=YELLOW,
            fill_opacity=0.2
        ).to_edge(DOWN, buff=0.5)
        
        insight_text = Text(
            "Policy adapted significantly for Dakota patterns (masked KL: 9.32)\n"
            "while preserving general language understanding (unmasked KL: 0.042)",
            font_size=22,
            color=WHITE
        ).move_to(insight_box)
        
        self.play(
            Create(insight_box),
            Write(insight_text),
            run_time=2
        )
        self.wait(3)
        
        # SCENE 4: Statistical highlights
        stats_title = Text("Key Statistics", font_size=32, color=YELLOW).to_edge(UP, buff=0.3)
        
        stats = VGroup(
            Text(f"Masked KL Mean: {masked_kl_mean.mean():.2f}", font_size=24, color=RED),
            Text(f"Overall KL Mean: {overall_kl_mean.mean():.2f}", font_size=24, color=ORANGE),
            Text(f"Unmasked KL Mean: {unmasked_kl_mean.mean():.4f}", font_size=24, color=GREEN),
            Text(f"Max Masked KL: {masked_kl_mean.max():.2f}", font_size=24, color=RED),
        ).arrange(DOWN, buff=0.4).move_to(ORIGIN)
        
        self.play(
            FadeOut(VGroup(axes, masked_curve, overall_curve, unmasked_curve, 
                          masked_group, overall_group, unmasked_group, 
                          insight_box, insight_text, x_label, y_label)),
            Write(stats_title),
            run_time=1
        )
        
        self.play(
            *[Write(stat) for stat in stats],
            run_time=2
        )
        self.wait(3)
        
        # Final fade
        self.play(
            FadeOut(VGroup(*self.mobjects)),
            run_time=2
        )


if __name__ == "__main__":
    # For testing
    print("Manim animation script for KL Divergence visualization")
    print("Run with: manim -pql scripts/visualization/kl_divergence_animation.py KLDivergenceEpicComplete")

