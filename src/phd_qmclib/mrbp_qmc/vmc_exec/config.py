from .io import IO_FILE_HANDLER_TYPES
from ..config import Loader

CONFIG_FILE_EXTENSIONS = ('.yml', '.yaml', '.toml')
YAML_EXTENSIONS = ('.yml', '.yaml')

UNIX_NEWLINE = '\n'

# The common loader instance.
loader = Loader(CONFIG_FILE_EXTENSIONS, IO_FILE_HANDLER_TYPES)
