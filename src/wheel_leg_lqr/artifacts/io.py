"""Output resolution and UTF-8 writing, independent of model calculations."""
from pathlib import Path

def project_root():
    root = Path(__file__).resolve().parents[3]
    return root if (root / 'pyproject.toml').is_file() and (root / 'src/wheel_leg_lqr').is_dir() else None

def default_output_dir(component):
    return (project_root() or Path.cwd()) / 'outputs' / component

def resolve_output_dir(output_dir):
    path = Path(output_dir).resolve()
    root = project_root()
    if root is not None:
        protected = root / 'data/reference'
        if path == protected or protected in path.parents:
            raise ValueError(f'output-dir must not point into reference data {protected}')
    return path

def write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip('\n') + '\n', encoding='utf-8', newline='\n')
