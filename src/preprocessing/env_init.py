import os
from pathlib import Path
from typing import Optional

import dotenv
from omegaconf import DictConfig, OmegaConf

def read_environment_variable(env_name: str, default: Optional[str] = None) -> str:
    """
    Safely retrieve the value of an environment variable. If the variable is not
    defined or its value is empty, it raises an error unless a default value is provided.
    
    Parameters:
    env_name (str): The name of the environment variable.
    default (str, optional): The default value for the environment variable.

    Returns:
    str: The value of the environment variable.
    """
    env_value: str = os.getenv(env_name, default)

    if env_value is None:
        raise KeyError(f"'{env_name}' not defined and no default value is provided.")

    if not env_value:
        raise ValueError(
            f"'{env_name}' is not set and no default value is provided."
        )

    return env_value


def load_environment_variables_from_file(env_file: Optional[str] = None) -> None:
    """
    Load environment variables from a file. This is similar to sourcing a bash script
    that exports variables. 

    Parameters:
    env_file (str, optional): The file that contains the definitions of the 
                              environment variables. If None, it searches for 
                              a '.env' file in the project directory.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# Load environment variables
load_environment_variables_from_file()

# Set the current working directory to the project root
PROJECT_ROOT: Path = Path(read_environment_variable("PROJECT_ROOT"))

# Check if project root exists
if PROJECT_ROOT.exists():
    os.chdir(PROJECT_ROOT)
    