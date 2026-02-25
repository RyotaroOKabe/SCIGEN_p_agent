good_colors = {'orange': "#e69f00", 'sky': "#56b4e9", 'green': "#009e73",
               'yellow': "#f0e473", 'blue': "#0072b2", 'red': "#d55e00", 'pink': "#cc79a7"
}

palette = list(good_colors.values())
sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

def visualize_dict_structure(d, indent=0):
    """
    Recursively visualizes the structure of nested dictionaries.

    Parameters:
        d (dict): The dictionary to visualize.
        indent (int): The current indentation level for nested dictionaries.
    """
    if not isinstance(d, dict):
        print(" " * indent + str(d))
        return

    for key, value in d.items():
        print(" " * indent + str(key) + ":")
        if isinstance(value, dict):
            visualize_dict_structure(value, indent + 4)
        else:
            print(" " * (indent + 4) + str(value))