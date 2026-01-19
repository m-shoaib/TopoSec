"""
Generate figures from experimental data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FigureGenerator:
    """Generate publication-quality figures"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        
    def load_data(self):
        """Load required data"""
        self.detections = pd.read_csv(f"{self.data_dir}/detections.csv")
        self.events = pd.read_csv(f"{self.data_dir}/lldp_events.csv")
        self.performance = pd.read_csv(f"{self.data_dir}/performance_metrics.csv")
        
        # Convert numeric columns
        self.detections['latency_ms'] = pd.to_numeric(self.detections['latency_ms'], errors='coerce')
        self.detections['dc_confidence'] = pd.to_numeric(self.detections['dc_confidence'], errors='coerce')
    
    def plot_detection_rates(self, output_file="analysis/output/detection_rates.png"):
        """Plot detection rates by attack type"""
        # Calculate detection rates
        attack_events = self.events[self.events['attack_type'].notna()]
        detection_rates = []
        
        for attack_type in sorted(attack_events['attack_type'].unique()):
            type_events = attack_events[attack_events['attack_type'] == attack_type]
            detected = len(type_events[type_events['detected'] == 'YES'])
            total = len(type_events)
            rate = (detected / total) * 100 if total > 0 else 0
            detection_rates.append((attack_type, rate))
        
        # Sort by rate
        detection_rates.sort(key=lambda x: x[1], reverse=True)
        attack_types, rates = zip(*detection_rates)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(range(len(rates)), rates, color=sns.color_palette("husl", len(rates)))
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # Formatting
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Detection Rate (%)', fontsize=12)
        ax.set_title('TopoSec Detection Performance by Attack Type', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([a.replace('_', ' ').title() for a in attack_types], 
                          rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Saved detection rates plot to {output_file}")
    
    def plot_latency_comparison(self, output_file="analysis/output/latency_comparison.png"):
        """Plot latency comparison: decoy vs behavioral"""
        # Separate decoy and behavioral detections
        decoy_detections = self.detections[self.detections['decoy_link_hit'] == 'YES']
        behavioral_detections = self.detections[self.detections['decoy_link_hit'] == 'NO']
        
        if len(decoy_detections) > 0 and len(behavioral_detections) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Box plot
            data = [decoy_detections['latency_ms'].dropna(), 
                   behavioral_detections['latency_ms'].dropna()]
            labels = ['Decoy-based', 'Behavioral']
            
            bp = ax1.boxplot(data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = ['lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax1.set_ylabel('Detection Latency (ms)', fontsize=12)
            ax1.set_title('Latency Distribution Comparison', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Bar chart with statistics
            decoy_mean = decoy_detections['latency_ms'].mean()
            behavioral_mean = behavioral_detections['latency_ms'].mean()
            improvement = ((behavioral_mean - decoy_mean) / behavioral_mean) * 100
            
            means = [decoy_mean, behavioral_mean]
            bars = ax2.bar(['Decoy-based', 'Behavioral'], means, 
                          color=['lightgreen', 'lightcoral'])
            
            # Add value labels
            for bar, mean in zip(bars, means):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{mean:.1f}ms', ha='center', va='bottom', fontsize=11)
            
            ax2.set_ylabel('Average Latency (ms)', fontsize=12)
            ax2.set_title(f'Average Latency: {improvement:.1f}% Improvement', 
                         fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ Saved latency comparison plot to {output_file}")
    
    def plot_performance_overhead(self, output_file="analysis/output/performance_overhead.png"):
        """Plot system performance overhead"""
        if self.performance is not None:
            # Filter relevant data
            perf_data = self.performance[self.performance['measurement_type'].isin(
                ['normal_operation', 'under_attack'])]
            
            # Convert to numeric
            perf_data['cpu_overhead_percent'] = pd.to_numeric(
                perf_data['cpu_overhead_percent'], errors='coerce')
            perf_data['memory_usage_mb'] = pd.to_numeric(
                perf_data['memory_usage_mb'], errors='coerce')
            
            # Group by measurement type
            grouped = perf_data.groupby('measurement_type').agg({
                'cpu_overhead_percent': 'mean',
                'memory_usage_mb': 'mean'
            }).reset_index()
            
            # Plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # CPU overhead
            cpu_data = grouped[['measurement_type', 'cpu_overhead_percent']]
            cpu_data['measurement_type'] = cpu_data['measurement_type'].replace({
                'normal_operation': 'Normal',
                'under_attack': 'Under Attack'
            })
            
            bars1 = ax1.bar(cpu_data['measurement_type'], cpu_data['cpu_overhead_percent'],
                           color=['skyblue', 'salmon'])
            
            for bar, val in zip(bars1, cpu_data['cpu_overhead_percent']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{val:.1f}%', ha='center', va='bottom')
            
            ax1.set_ylabel('CPU Overhead (%)', fontsize=12)
            ax1.set_title('CPU Overhead Comparison', fontsize=13, fontweight='bold')
            ax1.set_ylim(0, max(cpu_data['cpu_overhead_percent']) * 1.2)
            
            # Memory usage
            mem_data = grouped[['measurement_type', 'memory_usage_mb']]
            mem_data['measurement_type'] = mem_data['measurement_type'].replace({
                'normal_operation': 'Normal',
                'under_attack': 'Under Attack'
            })
            
            bars2 = ax2.bar(mem_data['measurement_type'], mem_data['memory_usage_mb'],
                           color=['lightblue', 'lightcoral'])
            
            for bar, val in zip(bars2, mem_data['memory_usage_mb']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{val:.1f} MB', ha='center', va='bottom')
            
            ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
            ax2.set_title('Memory Usage Comparison', fontsize=13, fontweight='bold')
            ax2.set_ylim(0, max(mem_data['memory_usage_mb']) * 1.2)
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✓ Saved performance overhead plot to {output_file}")
    
    def plot_decoy_hit_analysis(self, output_file="analysis/output/decoy_hits.png"):
        """Plot decoy hit analysis"""
        # Load decoy analysis
        try:
            decoy_analysis = pd.read_csv(f"{self.data_dir}/decoy_link_analysis.csv")
            
            if len(decoy_analysis) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Top decoy links by hit count
                top_decoy = decoy_analysis.nlargest(8, 'hit_count')
                
                bars = ax1.barh(range(len(top_decoy)), top_decoy['hit_count'],
                               color=sns.color_palette("viridis", len(top_decoy)))
                
                ax1.set_yticks(range(len(top_decoy)))
                ax1.set_yticklabels(top_decoy['decoy_link_id'])
                ax1.set_xlabel('Number of Attacks Detected', fontsize=12)
                ax1.set_title('Most Effective Decoy Links', fontsize=13, fontweight='bold')
                ax1.invert_yaxis()
                
                # Add value labels
                for i, (bar, count) in enumerate(zip(bars, top_decoy['hit_count'])):
                    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{int(count)}', va='center')
                
                # Attack types caught by decoys
                if 'attack_types_detected' in decoy_analysis.columns:
                    # Count attack types across all decoys
                    attack_counts = {}
                    for types_str in decoy_analysis['attack_types_detected']:
                        types = [t.strip() for t in types_str.split(',')]
                        for attack_type in types:
                            attack_counts[attack_type] = attack_counts.get(attack_type, 0) + 1
                    
                    if attack_counts:
                        attack_types, counts = zip(*sorted(attack_counts.items(), 
                                                         key=lambda x: x[1], reverse=True))
                        
                        bars2 = ax2.bar(range(len(counts)), counts,
                                       color=sns.color_palette("Set2", len(counts)))
                        
                        ax2.set_xticks(range(len(counts)))
                        ax2.set_xticklabels([a.replace('_', ' ').title() for a in attack_types],
                                           rotation=45, ha='right')
                        ax2.set_ylabel('Number of Decoy Links', fontsize=12)
                        ax2.set_title('Attack Types Caught by Decoy Links', 
                                     fontsize=13, fontweight='bold')
                
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✓ Saved decoy hit analysis plot to {output_file}")
                
        except FileNotFoundError:
            print("⚠️  Decoy analysis file not found, skipping plot")
    
    def generate_all_figures(self):
        """Generate all figures"""
        print("Generating publication figures...")
        
        self.load_data()
        
        # Create output directory
        os.makedirs("analysis/output", exist_ok=True)
        
        # Generate figures
        self.plot_detection_rates()
        self.plot_latency_comparison()
        self.plot_performance_overhead()
        self.plot_decoy_hit_analysis()
        
        print("\n✅ All figures generated in 'analysis/output/' directory")


if __name__ == "__main__":
    generator = FigureGenerator()
    generator.generate_all_figures()