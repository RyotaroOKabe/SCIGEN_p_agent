# Contributing to SCIGEN_p_agent

Thank you for your interest in contributing to SCIGEN_p_agent! This document provides guidelines for contributing code, documentation, and improvements.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## Getting Started

1. **Fork the repository** (if applicable)
2. **Clone your fork**:
   ```bash
   git clone <your-fork-url>
   cd SCIGEN_p_agent
   ```
3. **Create a branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Set up development environment**:
   ```bash
   conda env create -f setup/env.yml
   conda activate scigen_py312
   ```

## Development Workflow

### 1. Make Changes

- Make your changes in a feature branch
- Follow the code style guidelines (see below)
- Add tests if applicable
- Update documentation as needed

### 2. Test Your Changes

```bash
# Run validation script
python scripts/validation/validate_setup.py

# Test training (debug mode)
python scigen/run.py \
    data=mp_20 \
    model=diffusion_w_type \
    train.pl_trainer.fast_dev_run=True
```

### 3. Commit Your Changes

Use clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of changes"
```

**Commit Message Format**:
- Use present tense ("Add feature" not "Added feature")
- First line should be concise (< 50 characters)
- Add detailed description if needed (after blank line)

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request with:
- Clear description of changes
- Reference to related issues (if any)
- Screenshots/plots if applicable

## Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (soft limit)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Grouped and sorted
  ```python
  # Standard library
  import os
  from pathlib import Path
  
  # Third-party
  import torch
  import numpy as np
  
  # Local
  from scigen.common.utils import PROJECT_ROOT
  ```

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """Brief description.
    
    Longer description explaining what the function does,
    parameters, return values, and any exceptions.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When something goes wrong
    
    Example:
        >>> result = function_name(arg1, arg2)
        >>> print(result)
    """
```

### Type Hints

Add type hints to function signatures:

```python
from typing import List, Optional, Dict
from pathlib import Path

def process_data(
    data_path: Path,
    batch_size: int = 32,
    num_workers: Optional[int] = None
) -> List[Dict[str, torch.Tensor]]:
    ...
```

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_` (e.g., `_internal_method`)

### Code Formatting

We recommend using:
- **black**: Code formatter
- **isort**: Import sorter
- **flake8**: Linter

```bash
# Format code
black scigen/

# Sort imports
isort scigen/

# Check style
flake8 scigen/
```

## Documentation Guidelines

### Code Documentation

- Add docstrings to all public functions and classes
- Include type hints
- Add examples for complex functions
- Document edge cases and exceptions

### User Documentation

- Update README.md for user-facing changes
- Add to WORKFLOW.md for workflow changes
- Update API.md for API changes
- Add examples for new features

## Testing

### Unit Tests

Add unit tests for new functionality:

```python
# tests/test_module.py
import unittest
from scigen.module import function_name

class TestModule(unittest.TestCase):
    def test_function_name(self):
        result = function_name(input)
        self.assertEqual(result, expected)
```

### Integration Tests

Test complete workflows:

```bash
# Test data preparation
python data_prep/alex_process.py

# Test training
python scigen/run.py data=mp_20 train.pl_trainer.fast_dev_run=True
```

## Pull Request Process

1. **Update Documentation**: Ensure all documentation is up to date
2. **Add Tests**: Include tests for new functionality
3. **Check Style**: Run formatters and linters
4. **Test Changes**: Verify changes work as expected
5. **Write Clear Description**: Explain what and why, not just how

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No breaking changes (or documented)
- [ ] Commit messages are clear

## Areas for Contribution

### High Priority

- **Bug fixes**: Fix reported issues
- **Documentation**: Improve clarity and completeness
- **Performance**: Optimize slow operations
- **Testing**: Add missing tests

### Medium Priority

- **New features**: Add requested functionality
- **Code quality**: Refactor and improve code
- **Examples**: Add usage examples
- **Tutorials**: Create tutorial notebooks

### Low Priority

- **Code style**: Improve formatting
- **Comments**: Add clarifying comments
- **Documentation**: Fix typos and grammar

## Questions?

- Open an issue for bugs or feature requests
- Check existing documentation first
- Ask in discussions (if available)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md (if exists)
- Release notes
- Documentation acknowledgments

Thank you for contributing to SCIGEN_p_agent!



