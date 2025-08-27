from setuptools import setup

VERSION = "0.1.3"

setup(
    name="llm-claude-code-plugin",
    description="LLM plugin for Claude Code",
    author="Edward Champion",
    version=VERSION,
    license="Apache License, Version 2.0",
    py_modules=["llm_claude_code"],
    install_requires=["llm>=0.4"],
    extras_require={"test": ["pytest", "pytest-cov", "ruff"]},
    entry_points={"llm": ["claude_code = llm_claude_code"]},
    project_urls={
        "Source": "https://github.com/anthropics/claude-code",
    },
    classifiers=["License :: OSI Approved :: Apache Software License"],
    python_requires=">=3.7",
)
