

import subprocess


if __name__ == '__main__':
    subprocess.call("run_lake_policy_comparison.py")
    subprocess.call("run_lake_timing_tests.py")
    subprocess.call("run_lake_q_learning_test.py")
    subprocess.call("run_hanoi_policy_comparison.py")
    subprocess.call("run_hanoi_timing_test.py")
    subprocess.call("run_hanoi_q_learning_test.py")