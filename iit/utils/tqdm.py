def in_notebook() -> bool:
    try:
        # This will only work in Jupyter notebooks
        shell = get_ipython().__class__.__name__ # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other types of interactive shells
    except NameError:
        return False  # Probably standard Python interpreter
    
if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm