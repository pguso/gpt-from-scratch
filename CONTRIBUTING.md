# Contributing to GPT From Scratch

Thank you for your interest in contributing! This project aims to be the best educational resource for understanding GPT models.

## How to Contribute

### Report Bugs
- Use GitHub Issues
- Include code snippets to reproduce
- Describe expected vs actual behavior

### Improve Documentation
- Fix typos
- Add clarifications
- Create tutorials
- Improve examples

### Add Features
- New model variants
- Training optimizations
- Visualization tools
- Example notebooks

### Write Tests
- Add educational tests
- Improve test coverage
- Add visual demonstrations

## Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch

# Create environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,notebooks]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/

# Type checking
mypy src/
```

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Keep functions focused
- Add comments for complex logic

## Pull Request Process

1. Create a feature branch
2. Make your changes
3. Add tests
4. Update documentation
5. Run tests and formatting
6. Submit PR with clear description

## Questions?

Open a GitHub Discussion or issue!
