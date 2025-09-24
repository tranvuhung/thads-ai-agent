#!/usr/bin/env python3
"""
Test Runner for Semantic Search System

This script provides a comprehensive test runner for the semantic search system
with various testing options and configurations.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_command(command: List[str], cwd: Optional[str] = None) -> int:
    """Run a command and return exit code"""
    print(f"Running: {' '.join(command)}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    try:
        result = subprocess.run(command, cwd=cwd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1

def install_test_dependencies():
    """Install test dependencies"""
    print("Installing test dependencies...")
    
    requirements_file = "tests/requirements-test.txt"
    if os.path.exists(requirements_file):
        return run_command([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    else:
        print(f"Requirements file {requirements_file} not found")
        return 1

def run_unit_tests(verbose: bool = False, coverage: bool = True):
    """Run unit tests"""
    print("\n" + "="*50)
    print("RUNNING UNIT TESTS")
    print("="*50)
    
    command = [sys.executable, "-m", "pytest", "tests/", "-m", "unit"]
    
    if verbose:
        command.append("-v")
    
    if coverage:
        command.extend(["--cov=src", "--cov-report=term-missing"])
    
    return run_command(command)

def run_integration_tests(verbose: bool = False):
    """Run integration tests"""
    print("\n" + "="*50)
    print("RUNNING INTEGRATION TESTS")
    print("="*50)
    
    command = [sys.executable, "-m", "pytest", "tests/", "-m", "integration"]
    
    if verbose:
        command.append("-v")
    
    return run_command(command)

def run_performance_tests(verbose: bool = False):
    """Run performance tests"""
    print("\n" + "="*50)
    print("RUNNING PERFORMANCE TESTS")
    print("="*50)
    
    command = [sys.executable, "-m", "pytest", "tests/", "-m", "performance"]
    
    if verbose:
        command.append("-v")
    
    command.extend(["--benchmark-only", "--benchmark-sort=mean"])
    
    return run_command(command)

def run_specific_test_file(test_file: str, verbose: bool = False):
    """Run a specific test file"""
    print(f"\n" + "="*50)
    print(f"RUNNING SPECIFIC TEST: {test_file}")
    print("="*50)
    
    command = [sys.executable, "-m", "pytest", test_file]
    
    if verbose:
        command.append("-v")
    
    return run_command(command)

def run_tests_by_marker(marker: str, verbose: bool = False):
    """Run tests by marker"""
    print(f"\n" + "="*50)
    print(f"RUNNING TESTS WITH MARKER: {marker}")
    print("="*50)
    
    command = [sys.executable, "-m", "pytest", "tests/", "-m", marker]
    
    if verbose:
        command.append("-v")
    
    return run_command(command)

def run_all_tests(verbose: bool = False, coverage: bool = True, 
                 include_performance: bool = False):
    """Run all tests"""
    print("\n" + "="*50)
    print("RUNNING ALL TESTS")
    print("="*50)
    
    command = [sys.executable, "-m", "pytest", "tests/"]
    
    if not include_performance:
        command.extend(["-m", "not performance"])
    
    if verbose:
        command.append("-v")
    
    if coverage:
        command.extend([
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    return run_command(command)

def run_linting():
    """Run code linting"""
    print("\n" + "="*50)
    print("RUNNING CODE LINTING")
    print("="*50)
    
    exit_code = 0
    
    # Run flake8
    print("Running flake8...")
    result = run_command([sys.executable, "-m", "flake8", "src/", "tests/"])
    if result != 0:
        exit_code = result
    
    # Run black check
    print("Running black check...")
    result = run_command([sys.executable, "-m", "black", "--check", "src/", "tests/"])
    if result != 0:
        exit_code = result
    
    # Run isort check
    print("Running isort check...")
    result = run_command([sys.executable, "-m", "isort", "--check-only", "src/", "tests/"])
    if result != 0:
        exit_code = result
    
    return exit_code

def run_type_checking():
    """Run type checking with mypy"""
    print("\n" + "="*50)
    print("RUNNING TYPE CHECKING")
    print("="*50)
    
    return run_command([sys.executable, "-m", "mypy", "src/"])

def generate_test_report():
    """Generate comprehensive test report"""
    print("\n" + "="*50)
    print("GENERATING TEST REPORT")
    print("="*50)
    
    # Run tests with XML output for CI/CD
    command = [
        sys.executable, "-m", "pytest", "tests/",
        "--junitxml=test-results.xml",
        "--cov=src",
        "--cov-report=xml",
        "--cov-report=html:htmlcov",
        "-m", "not performance"
    ]
    
    return run_command(command)

def clean_test_artifacts():
    """Clean test artifacts and cache"""
    print("\n" + "="*50)
    print("CLEANING TEST ARTIFACTS")
    print("="*50)
    
    artifacts = [
        ".pytest_cache",
        "__pycache__",
        "htmlcov",
        ".coverage",
        "coverage.xml",
        "test-results.xml",
        ".mypy_cache"
    ]
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            if os.path.isdir(artifact):
                import shutil
                shutil.rmtree(artifact)
                print(f"Removed directory: {artifact}")
            else:
                os.remove(artifact)
                print(f"Removed file: {artifact}")
    
    # Clean __pycache__ directories recursively
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                dir_path = os.path.join(root, dir_name)
                import shutil
                shutil.rmtree(dir_path)
                print(f"Removed __pycache__: {dir_path}")

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Semantic Search System Test Runner")
    
    parser.add_argument("--install-deps", action="store_true",
                       help="Install test dependencies")
    parser.add_argument("--unit", action="store_true",
                       help="Run unit tests only")
    parser.add_argument("--integration", action="store_true",
                       help="Run integration tests only")
    parser.add_argument("--performance", action="store_true",
                       help="Run performance tests only")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")
    parser.add_argument("--file", type=str,
                       help="Run specific test file")
    parser.add_argument("--marker", type=str,
                       help="Run tests with specific marker")
    parser.add_argument("--lint", action="store_true",
                       help="Run code linting")
    parser.add_argument("--type-check", action="store_true",
                       help="Run type checking")
    parser.add_argument("--report", action="store_true",
                       help="Generate test report")
    parser.add_argument("--clean", action="store_true",
                       help="Clean test artifacts")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--no-coverage", action="store_true",
                       help="Disable coverage reporting")
    parser.add_argument("--include-performance", action="store_true",
                       help="Include performance tests in all tests")
    
    args = parser.parse_args()
    
    # If no specific action is specified, run all tests
    if not any([args.install_deps, args.unit, args.integration, args.performance,
               args.all, args.file, args.marker, args.lint, args.type_check,
               args.report, args.clean]):
        args.all = True
    
    exit_code = 0
    start_time = time.time()
    
    try:
        if args.install_deps:
            result = install_test_dependencies()
            if result != 0:
                exit_code = result
        
        if args.clean:
            clean_test_artifacts()
        
        if args.lint:
            result = run_linting()
            if result != 0:
                exit_code = result
        
        if args.type_check:
            result = run_type_checking()
            if result != 0:
                exit_code = result
        
        if args.unit:
            result = run_unit_tests(args.verbose, not args.no_coverage)
            if result != 0:
                exit_code = result
        
        if args.integration:
            result = run_integration_tests(args.verbose)
            if result != 0:
                exit_code = result
        
        if args.performance:
            result = run_performance_tests(args.verbose)
            if result != 0:
                exit_code = result
        
        if args.file:
            result = run_specific_test_file(args.file, args.verbose)
            if result != 0:
                exit_code = result
        
        if args.marker:
            result = run_tests_by_marker(args.marker, args.verbose)
            if result != 0:
                exit_code = result
        
        if args.all:
            result = run_all_tests(args.verbose, not args.no_coverage, 
                                 args.include_performance)
            if result != 0:
                exit_code = result
        
        if args.report:
            result = generate_test_report()
            if result != 0:
                exit_code = result
    
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        exit_code = 130
    
    except Exception as e:
        print(f"\nError during test execution: {e}")
        exit_code = 1
    
    finally:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n" + "="*50)
        print(f"TEST EXECUTION COMPLETED")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Exit code: {exit_code}")
        print("="*50)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()