#!/usr/bin/env python3
"""
Benchmark All Configurations
===========================

This script runs the comprehensive test suite that validates all 1,680 possible configurations
of the Differential CFD-ML framework across different flow types, advection schemes, 
pressure solvers, and timestepping modes.

Usage:
    python benchmark_all.py [--mode quick|medium|full|custom]

Test Modes:
- quick: 20 configs, 50 frames each (~10-20 minutes)
- medium: 50 configs, 100 frames each (~1-2 hours)  
- full: All 1,680 configs (~1-2 days)
- custom: User-defined configuration
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from test_framework import TestFramework, TestMode
    print("Successfully imported test framework")
except ImportError as e:
    print(f"Failed to import test framework: {e}")
    sys.exit(1)


def main():
    """Main function to run benchmark tests"""
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark tests for Differential CFD-ML framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_all.py --mode quick
  python benchmark_all.py --mode medium  
  python benchmark_all.py --mode full
  python benchmark_all.py --mode custom --configs 100 --frames 200
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['quick', 'medium', 'full', 'custom'],
        default='quick',
        help='Test mode (default: quick)'
    )
    
    parser.add_argument(
        '--configs',
        type=int,
        default=20,
        help='Number of configurations for custom mode (default: 20)'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=50,
        help='Number of frames per configuration for custom mode (default: 50)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel (experimental)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Differential CFD-ML Framework - Comprehensive Benchmark Suite")
    print("=" * 80)
    
    # Determine test mode
    if args.mode == 'quick':
        test_mode = TestMode.QUICK
        print(f"Mode: Quick Test (20 configs, 50 frames)")
        print(f"Estimated time: 10-20 minutes")
        
    elif args.mode == 'medium':
        test_mode = TestMode.MEDIUM
        print(f"Mode: Medium Test (50 configs, 100 frames)")
        print(f"Estimated time: 1-2 hours")
        
    elif args.mode == 'full':
        test_mode = TestMode.FULL
        print(f"Mode: Full Test (1,680 configs)")
        print(f"Estimated time: 1-2 days")
        
    elif args.mode == 'custom':
        print(f"Mode: Custom Test ({args.configs} configs, {args.frames} frames)")
        print(f"Estimated time: {args.configs * args.frames / 1000:.1f} hours")
        
        # Create custom test mode
        class CustomTestMode:
            def __init__(self, configs, frames):
                self.num_configs = configs
                self.frames_per_config = frames
                self.name = f"custom_{configs}_{frames}"
        
        test_mode = CustomTestMode(args.configs, args.frames)
    
    print(f"Output directory: {args.output}")
    print(f"Parallel execution: {args.parallel}")
    print()
    
    # Create test framework
    try:
        framework = TestFramework(
            output_dir=args.output,
            parallel=args.parallel
        )
        print("Test framework initialized successfully")
    except Exception as e:
        print(f"Failed to initialize test framework: {e}")
        sys.exit(1)
    
    # Run tests
    print("\nStarting benchmark tests...")
    print("Press Ctrl+C to interrupt")
    print("-" * 80)
    
    try:
        results = framework.run_tests(test_mode)
        
        print("\n" + "=" * 80)
        print("Benchmark completed successfully!")
        print("=" * 80)
        
        # Print summary
        if hasattr(results, 'total_configs'):
            print(f"Total configurations tested: {results.total_configs}")
            print(f"Successful: {results.successful}")
            print(f"Failed: {results.failed}")
            print(f"Success rate: {100 * results.successful / results.total_configs:.1f}%")
        
        if hasattr(results, 'avg_performance'):
            print(f"Average performance: {results.avg_performance:.1f} steps/sec")
        
        print(f"\nResults saved to: {args.output}/")
        print("Check the following files:")
        print(f"  - {args.output}/test_results.csv")
        print(f"  - {args.output}/summary.txt")
        print(f"  - {args.output}/failed_configs.txt")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        print("Partial results may be available in output directory")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
