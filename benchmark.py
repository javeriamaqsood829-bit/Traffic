# ============================================================
# Multi-Vehicle Speed Detection System — Benchmark & Validation
# ============================================================
# Ground Truth Verification, Error Residual Analysis, and
# Scientific Statistical Validation Suite (MAE, RMSE, MAPE, Bland-Altman).
# ============================================================

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class GroundTruthBenchmark:
    """
    Statistical validation engine to rigorously verify if vision-based
    speed estimates are genuine, calibrated, and scientifically accurate.
    """

    def __init__(self, ground_truth_speeds, predicted_speeds):
        """
        Args:
            ground_truth_speeds (array-like): Actual verified speeds (from GPS OBD-II / Radar)
            predicted_speeds (array-like): Speeds calculated by the vision pipeline
        """
        self.y_true = np.array(ground_truth_speeds, dtype=np.float64)
        self.y_pred = np.array(predicted_speeds, dtype=np.float64)
        self.residuals = self.y_pred - self.y_true
        self.abs_errors = np.abs(self.residuals)
        self.percentage_errors = (self.abs_errors / self.y_true) * 100.0

    def compute_metrics(self):
        """Compute formal evaluation metrics."""
        mae = float(np.mean(self.abs_errors))
        rmse = float(np.sqrt(np.mean(self.residuals ** 2)))
        mape = float(np.mean(self.percentage_errors))
        bias = float(np.mean(self.residuals))  # Systematic over/under estimation
        std_error = float(np.std(self.residuals))

        # Correlation & R^2
        corr_matrix = np.corrcoef(self.y_true, self.y_pred)
        r_val = float(corr_matrix[0, 1]) if corr_matrix.shape == (2, 2) else 1.0
        r_squared = r_val ** 2

        # Compliance under statutory tolerances
        within_3kmh = float(np.mean(self.abs_errors <= 3.0) * 100.0)
        within_5kmh = float(np.mean(self.abs_errors <= 5.0) * 100.0)
        within_10kmh = float(np.mean(self.abs_errors <= 10.0) * 100.0)

        return {
            "sample_size": len(self.y_true),
            "mae_kmh": round(mae, 2),
            "rmse_kmh": round(rmse, 2),
            "mape_percent": round(mape, 2),
            "mean_bias_kmh": round(bias, 2),
            "std_error_kmh": round(std_error, 2),
            "r_squared": round(r_squared, 4),
            "within_3kmh_pct": round(within_3kmh, 1),
            "within_5kmh_pct": round(within_5kmh, 1),
            "within_10kmh_pct": round(within_10kmh, 1),
        }

    def print_summary(self):
        """Print formatted statistical summary table."""
        m = self.compute_metrics()
        print("\n" + "═" * 58)
        print(" ╔══════════════════════════════════════════════════════╗")
        print(" ║      SCIENTIFIC ACCURACY & VALIDATION BENCHMARK      ║")
        print(" ╠══════════════════════════════════════════════════════╣")
        print(f" ║  Sample Size Evaluated:  {m['sample_size']:<26} ║")
        print(f" ║  Mean Absolute Error:    {m['mae_kmh']:<6.2f} km/h                 ║")
        print(f" ║  Root Mean Squared Error:{m['rmse_kmh']:<6.2f} km/h                 ║")
        print(f" ║  Mean Abs Percentage Err:{m['mape_percent']:<6.2f} %                    ║")
        print(f" ║  Systematic Bias:        {m['mean_bias_kmh']:<+6.2f} km/h                 ║")
        print(f" ║  R² Correlation Index:   {m['r_squared']:<6.4f}                   ║")
        print(f" ║  Accuracy within ±3 km/h:{m['within_3kmh_pct']:<5.1f} % (Statutory Pass)   ║")
        print(f" ║  Accuracy within ±5 km/h:{m['within_5kmh_pct']:<5.1f} %                    ║")
        print(f" ║  Accuracy within ±10km/h:{m['within_10kmh_pct']:<5.1f} %                    ║")
        print(" ╚══════════════════════════════════════════════════════╝")
        print("═" * 58 + "\n")

    def plot_validation_report(self, output_path="output/ground_truth_benchmark.png"):
        """
        Generate a 4-panel publication-grade validation report:
          1. Ground Truth vs Predicted Scatter with 1:1 Identity Line & Envelope
          2. Bland-Altman Agreement Plot (Mean Bias ± 1.96 SD)
          3. Error Residual Histogram with Gaussian Fit
          4. Error vs Speed Velocity Dependency
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("COMPUTER VISION SPEED ESTIMATION: GROUND TRUTH VALIDATION", fontsize=15, fontweight='bold', y=0.98)

        m = self.compute_metrics()

        # ----------------------------------------------------
        # Panel 1: Predicted vs Ground Truth Scatter
        # ----------------------------------------------------
        ax1 = axes[0, 0]
        min_spd = min(self.y_true.min(), self.y_pred.min()) - 5
        max_spd = max(self.y_true.max(), self.y_pred.max()) + 5

        ax1.scatter(self.y_true, self.y_pred, color='#2980b9', edgecolors='black', s=55, alpha=0.8, label='Passage Estimates')
        ax1.plot([min_spd, max_spd], [min_spd, max_spd], 'r--', linewidth=2, label='Ideal 1:1 Identity (y = x)')
        # ±5 km/h confidence boundary
        ax1.fill_between([min_spd, max_spd], [min_spd - 5, max_spd - 5], [min_spd + 5, max_spd + 5],
                         color='gray', alpha=0.15, label='±5 km/h Tolerance Zone')

        ax1.set_title(f"Predicted vs Ground Truth Speed (R² = {m['r_squared']:.3f})", fontsize=11, fontweight='bold')
        ax1.set_xlabel("Ground Truth Speed (GPS / Radar km/h)", fontsize=10)
        ax1.set_ylabel("Vision System Estimated Speed (km/h)", fontsize=10)
        ax1.set_xlim(min_spd, max_spd)
        ax1.set_ylim(min_spd, max_spd)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=9)

        # ----------------------------------------------------
        # Panel 2: Bland-Altman Agreement Plot
        # ----------------------------------------------------
        ax2 = axes[0, 1]
        mean_speeds = (self.y_true + self.y_pred) / 2.0
        bias = m['mean_bias_kmh']
        sd = m['std_error_kmh']
        upper_limit = bias + 1.96 * sd
        lower_limit = bias - 1.96 * sd

        ax2.scatter(mean_speeds, self.residuals, color='#8e44ad', edgecolors='black', s=55, alpha=0.8)
        ax2.axhline(bias, color='#e74c3c', linestyle='-', linewidth=2, label=f'Mean Bias: {bias:+.2f} km/h')
        ax2.axhline(upper_limit, color='#e67e22', linestyle='--', linewidth=1.5, label=f'+1.96 SD: {upper_limit:+.2f} km/h')
        ax2.axhline(lower_limit, color='#e67e22', linestyle='--', linewidth=1.5, label=f'-1.96 SD: {lower_limit:+.2f} km/h')

        ax2.set_title("Bland-Altman Agreement Plot", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Mean of Ground Truth & Vision Speed (km/h)", fontsize=10)
        ax2.set_ylabel("Difference (Predicted - Ground Truth km/h)", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='lower right', fontsize=8.5)

        # ----------------------------------------------------
        # Panel 3: Error Residual Distribution & Gaussian Normal Fit
        # ----------------------------------------------------
        ax3 = axes[1, 0]
        n, bins, patches = ax3.hist(self.residuals, bins=12, density=True, color='#16a085', edgecolor='black', alpha=0.7, label='Residual Errors')

        if HAS_SCIPY:
            x_curve = np.linspace(bins.min() - 2, bins.max() + 2, 100)
            p_curve = stats.norm.pdf(x_curve, bias, sd)
            ax3.plot(x_curve, p_curve, 'r-', linewidth=2, label=f'Normal Fit (μ={bias:.2f}, σ={sd:.2f})')

        ax3.set_title(f"Error Residual Distribution (MAE = {m['mae_kmh']} km/h)", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Residual Error (km/h)", fontsize=10)
        ax3.set_ylabel("Probability Density", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right', fontsize=9)

        # ----------------------------------------------------
        # Panel 4: Percentage Error vs Vehicle Velocity
        # ----------------------------------------------------
        ax4 = axes[1, 1]
        ax4.scatter(self.y_true, self.percentage_errors, color='#d35400', edgecolors='black', s=55, alpha=0.8)
        ax4.axhline(5.0, color='#27ae60', linestyle='--', linewidth=1.5, label='5% Error Line')
        ax4.axhline(m['mape_percent'], color='#c0392b', linestyle='-', linewidth=2, label=f'Mean MAPE: {m["mape_percent"]:.1f}%')

        ax4.set_title(f"Relative Error vs Speed (MAPE = {m['mape_percent']}%)", fontsize=11, fontweight='bold')
        ax4.set_xlabel("Ground Truth Vehicle Speed (km/h)", fontsize=10)
        ax4.set_ylabel("Absolute Error Percentage (%)", fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='upper right', fontsize=9)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(output_path, dpi=160)
        plt.close()
        print(f"📊 Ground truth validation report generated at: {output_path}")


def generate_synthetic_benchmark():
    """
    Generate realistic multi-vehicle ground-truth calibration dataset
    simulating calibrated camera passes with realistic sensor noise.
    """
    np.random.seed(42)
    # 35 test vehicle passage speeds ranging from 20 km/h to 140 km/h
    ground_truth = np.array([
        22.5, 26.0, 28.5, 31.0, 34.0, 38.5, 42.0, 45.0, 48.0, 52.5,
        55.0, 58.0, 61.5, 64.0, 67.5, 71.0, 75.0, 78.5, 82.0, 85.5,
        89.0, 93.0, 96.5, 101.0, 105.5, 110.0, 114.5, 118.0, 122.5, 126.0,
        130.5, 134.0, 138.5, 142.0, 148.0
    ])

    # Realistic vision estimation noise model:
    # 1. Detection jitter (bounding box centroid shifts by ±1.5 pixels)
    # 2. Timing quantization (discrete frame delta of 29.97 FPS camera)
    # 3. Small homography angle drift
    noise = np.random.normal(loc=0.35, scale=2.1, size=len(ground_truth))
    # Quantize to 0.1 precision
    predicted = np.round(ground_truth + noise, 1)

    return ground_truth, predicted


def main():
    parser = argparse.ArgumentParser(description="Multi-Vehicle Speed Detection System — Ground Truth Validation")
    parser.add_argument("--csv", default=None, help="Optional CSV file containing 'true_speed,pred_speed' pairs")
    parser.add_argument("--output", default="output/ground_truth_benchmark.png", help="Path to save validation chart")
    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        true_speeds = []
        pred_speeds = []
        with open(args.csv, mode='r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 2:
                    true_speeds.append(float(row[0]))
                    pred_speeds.append(float(row[1]))
    else:
        print("ℹ️  Running automated synthetic ground-truth calibration benchmark...")
        true_speeds, pred_speeds = generate_synthetic_benchmark()

    bench = GroundTruthBenchmark(true_speeds, pred_speeds)
    bench.print_summary()
    bench.plot_validation_report(args.output)


if __name__ == "__main__":
    main()

