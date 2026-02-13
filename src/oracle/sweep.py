import wandb
import argparse

from oracle.train import run_training_loop, model_choices

class SweepArgs:
    """
    Mock class to simulate the argparse.Namespace object expected by train.py.
    """
    def __init__(self, config, model_name):
        self.model = model_name
        # Map parameters from wandb config (the values chosen by the sweep)
        self.num_epochs = config.get('num_epochs', 100)
        self.batch_size = config.get('batch_size', 1024)
        self.lr = config.get('lr', 1e-5)
        self.max_n_per_class = config.get('max_n_per_class', None)
        self.alpha = config.get('alpha', 0.0)
        self.gamma = config.get('gamma', 1.0)
        # Defaults for non-swept parameters
        self.dir = None
        self.load_weights = None

def sweep_train():
    """
    The function wandb.agent calls. Each call is one set of hyperparameters.
    """
    # 1. Initialize wandb within the agent process. 
    # This picks up the unique hyperparameters for this specific trial.
    run = wandb.init()
    
    # 2. Extract the model name (passed through the sweep config)
    model_name = wandb.config.model_choice
    
    # 3. Create the mock args object that train.py expects
    args = SweepArgs(wandb.config, model_name)
    
    # 4. Run the training loop.
    # When this calls get_wandb_run(args) inside train.py, 
    # wandb.init() will return the CURRENT run initialized above.
    run_training_loop(args)

def main():
    parser = argparse.ArgumentParser(description='ORACLE Sweep Runner')
    parser.add_argument('model', choices=model_choices, help='Model to sweep.')
    parser.add_argument('--count', type=int, default=10, help='Number of runs.')
    cmd_args = parser.parse_args()

    model = cmd_args.model

    if model=="BTSv2_PSonly":

        sweep_config = {
            'method': 'bayes',  # Optimization strategy
            'metric': {
                'name': 'Max (f1 score)',
                'goal': 'maximize'
            },
            'parameters': {
                'model_choice': {'value': model},
                'lr': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-7,
                    'max': 1e-5
                },
                'batch_size': {
                    'values': [32, 64, 128]
                },
                'alpha': {
                    'distribution': 'uniform',
                    'min': 0.0,
                    'max': 0.5
                },
                'gamma': {
                    'value': 1.0
                },
                'num_epochs': {
                    'value': 1000
                }
            }
        }
    
    elif model=="BTSv2":

        sweep_config = {
            'method': 'bayes',  # Optimization strategy
            'metric': {
                'name': 'Max (f1 score)',
                'goal': 'maximize'
            },
            'parameters': {
                'model_choice': {'value': model},
                'lr': {
                    'distribution': 'log_uniform_values',
                    'min': 1e-6,
                    'max': 1e-3
                },
                'batch_size': {
                    'values': [32, 64, 128, 256, 512]
                },
                'alpha': {
                    'distribution': 'uniform',
                    'min': 0.0,
                    'max': 0.5
                },
                'gamma': {
                    'value': 1.0
                },
                'num_epochs': {
                    'value': 1000
                }
            }
        }

        # sweep_config = {
        #     'method': 'bayes',  # Optimization strategy
        #     'metric': {
        #         'name': 'Max (f1 score)',
        #         'goal': 'maximize'
        #     },
        #     'parameters': {
        #         'model_choice': {'value': model},
        #         'lr': {
        #             'distribution': 'log_uniform_values',
        #             'min': 1e-6,
        #             'max': 1e-3
        #         },
        #         'batch_size': {
        #             'values': [32, 64, 128, 256, 512]
        #         },
        #         'alpha': {
        #             'distribution': 'uniform',
        #             'min': 0.0,
        #             'max': 0.5
        #         },
        #         'gamma': {
        #             'distribution': 'uniform',
        #             'min': 0.5,
        #             'max': 1.0
        #         },
        #         'num_epochs': {
        #             'value': 1000
        #         }
        #     }
        # }

    # Initialize the sweep on the wandb servers
    sweep_id = wandb.sweep(
        sweep_config, 
        project="ORACLE", 
        entity="vedshah-email-northwestern-university"
    )
    
    # Run the agent
    wandb.agent(sweep_id, function=sweep_train, count=cmd_args.count)

if __name__ == '__main__':
    main()