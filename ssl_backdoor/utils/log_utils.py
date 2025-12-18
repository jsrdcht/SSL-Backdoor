import sys
import os
import time

class TeeLogger(object):
    """
    A simple logger that redirects stdout/stderr to both terminal and a file.
    Designed to capture all print() outputs without modifying existing code.
    This is necessary because Python's standard logging module does not automatically 
    capture print() statements, and modifying all print() calls in legacy code is often impractical.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        # Use line buffering or unbuffered if possible, but "a" mode text file is standard
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensure logs are written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self.log:
            self.log.close()

def setup_logging(output_dir, experiment_name="log"):
    """
    Sets up logging to file and stdout.
    
    Args:
        output_dir (str): Directory to save log files.
        experiment_name (str): Prefix for the log file name.
        
    Returns:
        tuple: (log_path, experiment_id)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    experiment_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{experiment_name}_{experiment_id}.txt")
    
    print(f"Logs will be saved to {log_path}")
    
    # Redirect stdout and stderr
    # We only redirect if it hasn't been redirected yet to avoid nested loops if called multiple times
    if not isinstance(sys.stdout, TeeLogger):
        sys.stdout = TeeLogger(log_path)
        sys.stderr = sys.stdout
    
    return log_path, experiment_id
