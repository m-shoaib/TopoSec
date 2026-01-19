"""
Validate TopoSec experimental data for consistency and quality
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataValidator:
    """Validate experimental data quality and consistency"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
    
    def run_all_validations(self):
        """Run all validation checks"""
        print("=" * 60)
        print("TOPOSEC DATA VALIDATION")
        print("=" * 60)
        
        results = {}
        
        # Check 1: File existence
        results['files_exist'] = self._check_files_exist()
        
        # Check 2: LLDP events validation
        results['lldp_events'] = self._validate_lldp_events()
        
        # Check 3: Detections validation
        results['detections'] = self._validate_detections()
        
        # Check 4: Cross-file consistency
        results['cross_validation'] = self._cross_validate()
        
        # Check 5: Statistical plausibility
        results['statistical_checks'] = self._statistical_checks()
        
        # Print summary
        self._print_validation_summary(results)
        
        return all(results.values())
    
    def _check_files_exist(self):
        """Check all required files exist"""
        required_files = [
            'lldp_events.csv',
            'detections.csv',
            'link_state.csv',
            'performance_metrics.csv',
            'decoy_link_analysis.csv',
            'experiment_artifacts.csv'
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(os.path.join(self.data_dir, file)):
                missing.append(file)
        
        if missing:
            print(f"❌ Missing required files: {missing}")
            return False
        
        print("✅ All required files exist")
        return True
    
    def _validate_lldp_events(self):
        """Validate lldp_events.csv"""
        try:
            df = pd.read_csv(os.path.join(self.data_dir, 'lldp_events.csv'))
            
            checks = []
            
            # Basic structure
            checks.append(('row_count', len(df) > 40000))
            checks.append(('columns', len(df.columns) >= 15))
            
            # Attack vs normal events
            attack_count = df['attack_type'].notna().sum()
            normal_count = df['attack_type'].isna().sum()
            checks.append(('attack_events', attack_count >= 9000))
            checks.append(('normal_events', normal_count >= 35000))
            
            # Detection flags
            checks.append(('detected_values', set(df['detected'].dropna().unique()) <= {'YES', 'NO'}))
            
            # Timestamps
            try:
                df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
                checks.append(('timestamp_format', True))
                
                # Check for reasonable time range
                time_range = (df['timestamp_dt'].max() - df['timestamp_dt'].min()).total_seconds()
                checks.append(('time_range', 0 < time_range < 7*86400))  # Less than 7 days
            except:
                checks.append(('timestamp_format', False))
            
            # Check all checks
            all_passed = all(passed for _, passed in checks)
            
            if all_passed:
                print("✅ LLDP events validation passed")
                print(f"   Attack events: {attack_count:,}")
                print(f"   Normal events: {normal_count:,}")
                print(f"   Time range: {time_range/3600:.1f} hours")
            else:
                failed = [name for name, passed in checks if not passed]
                print(f"❌ LLDP events validation failed: {failed}")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Error validating lldp_events: {e}")
            return False
    
    def _validate_detections(self):
        """Validate detections.csv"""
        try:
            df = pd.read_csv(os.path.join(self.data_dir, 'detections.csv'))
            
            checks = []
            
            # Basic checks
            checks.append(('row_count', len(df) > 8000))
            
            # Detection rates should be between 90-100%
            if 'detection_rate' in df.columns:
                rates = pd.to_numeric(df['detection_rate'], errors='coerce')
                checks.append(('detection_rate_range', rates.between(90, 100).all()))
            
            # Latency should be positive
            if 'latency_ms' in df.columns:
                latency = pd.to_numeric(df['latency_ms'], errors='coerce')
                checks.append(('latency_positive', (latency >= 0).all()))
            
            # Confidence scores
            if 'dc_confidence' in df.columns:
                conf = pd.to_numeric(df['dc_confidence'], errors='coerce')
                checks.append(('confidence_range', conf.between(0, 1).all()))
            
            # Check decoy hit consistency
            if 'decoy_link_hit' in df.columns:
                decoy_hits = df[df['decoy_link_hit'] == 'YES']
                if len(decoy_hits) > 0:
                    checks.append(('decoy_confidence', 
                                 decoy_hits['dc_confidence'].mean() > 0.95))
            
            all_passed = all(passed for _, passed in checks)
            
            if all_passed:
                print("✅ Detections validation passed")
                print(f"   Total detections: {len(df):,}")
                if 'decoy_link_hit' in df.columns:
                    decoy_count = df['decoy_link_hit'].value_counts().get('YES', 0)
                    print(f"   Decoy-based detections: {decoy_count:,}")
            else:
                failed = [name for name, passed in checks if not passed]
                print(f"❌ Detections validation failed: {failed}")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Error validating detections: {e}")
            return False
    
    def _cross_validate(self):
        """Cross-validate between files"""
        try:
            events = pd.read_csv(os.path.join(self.data_dir, 'lldp_events.csv'))
            detections = pd.read_csv(os.path.join(self.data_dir, 'detections.csv'))
            
            checks = []
            
            # Events marked as detected should have corresponding detection records
            detected_events = events[events['detected'] == 'YES']['event_id']
            detection_events = detections['correlated_event_id']
            
            # Check overlap (not all need to match due to timing and processing delays)
            overlap = set(detected_events).intersection(set(detection_events))
            checks.append(('event_detection_overlap', len(overlap) > 0))
            
            # Decoy hits should be consistent
            if 'decoy_link_hit' in events.columns and 'decoy_link_hit' in detections.columns:
                event_decoy = set(events[events['decoy_link_hit'] == 'YES']['event_id'])
                det_decoy = set(detections[detections['decoy_link_hit'] == 'YES']['correlated_event_id'])
                checks.append(('decoy_consistency', event_decoy.issubset(det_decoy)))
            
            all_passed = all(passed for _, passed in checks)
            
            if all_passed:
                print("✅ Cross-validation passed")
                print(f"   Event-detection overlap: {len(overlap):,} events")
            else:
                failed = [name for name, passed in checks if not passed]
                print(f"❌ Cross-validation failed: {failed}")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Cross-validation error: {e}")
            return False
    
    def _statistical_checks(self):
        """Statistical plausibility checks"""
        try:
            events = pd.read_csv(os.path.join(self.data_dir, 'lldp_events.csv'))
            
            # Calculate detection rate
            attack_events = events[events['attack_type'].notna()]
            if len(attack_events) > 0:
                detection_rate = (attack_events['detected'] == 'YES').mean() * 100
                
                # Should be between 95-98% based on paper
                plausible = 95 <= detection_rate <= 98
                
                if plausible:
                    print("✅ Statistical checks passed")
                    print(f"   Overall detection rate: {detection_rate:.1f}%")
                    return True
                else:
                    print(f"❌ Implausible detection rate: {detection_rate:.1f}%")
                    return False
            else:
                print("⚠️  No attack events found for statistical check")
                return True
                
        except Exception as e:
            print(f"❌ Statistical check error: {e}")
            return False
    
    def _print_validation_summary(self, results):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test:25s}: {status}")
        
        print(f"\nOverall: {passed}/{total} checks passed")
        
        if passed == total:
            print("\n🎉 All validations passed! Data appears consistent.")
        else:
            print("\n⚠️  Some validations failed. Review the data.")


if __name__ == "__main__":
    validator = DataValidator()
    success = validator.run_all_validations()
    
    if not success:
        exit(1)