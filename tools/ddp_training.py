import os
import sys
import argparse
import yaml

# Import trainer API and config loader
from ssl_backdoor.ssl_trainers.trainer import get_trainer
from ssl_backdoor.ssl_trainers.utils import load_config

def main():
    parser = argparse.ArgumentParser(description='Run training; CLI args override config keys')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='Base config path, supports .py or .yaml')
    parser.add_argument('--attack_config', type=str, default=None, required=True,
                        help='Attack config file path (.yaml)')
    parser.add_argument('--test_config', type=str, default=None,
                        help='Test config file path (.yaml)')
    
    args = parser.parse_args()
    
    # 1. Load base config
    config = load_config(args.config)
    print(f"Loaded base config: {args.config}")
    
    # 2. Load and merge attack config if provided
    if args.attack_config:
        try:
            attack_config = load_config(args.attack_config) # Use shared loader
            if isinstance(attack_config, dict):
                print(f"Loaded attack config: {args.attack_config}")
                # Merge, giving attack config precedence on key conflicts.
                config.update(attack_config)
                print("Attack config merged.")
            else:
                print(f"Warning: attack config {args.attack_config} is not a dict; skip merge.")
        except Exception as e:
            print(f"Warning: failed to load attack config {args.attack_config}: {e}; skip merge.")

    # 2.5 Load and store test config if provided.
    if args.test_config:
        try:
            test_config_dict = load_config(args.test_config)
            if isinstance(test_config_dict, dict):
                print(f"Loaded test config: {args.test_config}")
                config['test_config'] = test_config_dict
            else:
                print(f"Warning: test config {args.test_config} is not a dict; skip merge.")
        except Exception as e:
            print(f"Warning: failed to load test config {args.test_config}: {e}; skip merge.")

    print("\nFinal training config:", config)
    print("\nFinal test config:", config.get('test_config'))
    
    # 5. Instantiate trainer with updated config
    trainer = get_trainer(config)
    
    # 6. Prepare evaluation config.
    eval_frequency = config.get('eval_frequency', 50)
    print(f"eval_frequency: {eval_frequency}, type: {type(eval_frequency)}")
    

    # 7. Start training
    trainer = get_trainer(config)
    trainer()

if __name__ == '__main__':
    main() 
