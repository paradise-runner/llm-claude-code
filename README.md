# llm-claude-code-plugin

[![PyPI](https://img.shields.io/pypi/v/llm-claude-code.svg)](https://pypi.org/project/llm-claude-code-plugin/)
[![Changelog](https://img.shields.io/github/v/release/paradise-runner/llm-claude-code?include_prereleases&label=changelog)](https://github.com/paradise-runner/llm-claude-code/releases)
[![Tests](https://github.com/paradise-runner/llm-claude-code/workflows/Tests/badge.svg)](https://github.com/paradise-runner/llm-claude-code/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/paradise-runner/llm-claude-code/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/paradise-runner/llm-claude-code/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Claude Code models (Sonnet and Opus) via the Claude Code CLI.

## Installation

Install this plugin in the same environment as LLM:

```bash
uv add llm-claude-code-plugin
```

## Prerequisites

This plugin requires the Claude Code CLI to be installed and available. The plugin looks for the Claude Code executable at:

- `~/.claude/local/claude` (default)
- Custom path specified via the `claude_path` option

Make sure you have Claude Code installed and configured before using this plugin.

## Usage

### Available Models

This plugin provides five models:

- `code/sonnet` - Claude Code with latest Sonnet model (default)
- `code/opus` - Claude Code with latest Opus model
- `code/sonnet-4` - Claude Code with Sonnet 4 (claude-sonnet-4-20250514)
- `code/opus-4` - Claude Code with Opus 4 (claude-opus-4-20250514)
- `code/opus-4.1` - Claude Code with Opus 4.1 (claude-opus-4-1-20250805)

List available models:

```bash
llm models list
```

### Basic Usage

Execute a prompt using Claude Code Sonnet:

```bash
llm -m code/sonnet "Explain how to use Python decorators"
```

Execute a prompt using Claude Code Opus:

```bash
llm -m code/opus "Write a Python function to calculate fibonacci numbers"
```

Execute a prompt using Claude Code Sonnet 4:

```bash
llm -m code/sonnet-4 "Review this code for best practices"
```

### Model Options

The plugin supports the following options:

- `claude_path`: Custom path to the Claude Code CLI executable

Example using a custom Claude Code path:

```bash
llm -m code/sonnet -o claude_path /usr/local/bin/claude "Your prompt here"
```

### File Attachments

This plugin supports attaching various file types to your prompts, including:

- Text files (`.txt`, `.py`, `.js`, `.html`, `.css`, `.md`, etc.)
- Code files (Python, JavaScript, C, Java, Shell scripts)
- Data files (JSON, XML, YAML)
- Images (PNG, JPEG, GIF, WebP)

Attach files using the `-a` flag:

```bash
llm -m code/sonnet -a myfile.py "Explain this Python code"
```

Attach multiple files:

```bash
llm -m code/sonnet -a file1.py -a file2.js "Compare these two implementations"
```

### URL Support

When URLs are included in prompts or attachments, the plugin automatically enables the WebFetch tool for Claude Code:

```bash
llm -m code/sonnet "Analyze the content at https://example.com"
```

### Interactive Chat

Start an interactive chat session:

```bash
llm chat -m code/sonnet
```

Or with Opus:

```bash
llm chat -m code/opus
```

Or with Sonnet 4:

```bash
llm chat -m code/sonnet-4
```

### Configuration

You can set the Claude Code path permanently using LLM's configuration:

```bash
llm logs path  # Find your logs directory
# Edit the config file to set default claude_path
```

## Features

- **Five Model Support**: Access latest Sonnet/Opus and specific versioned models (Sonnet 4, Opus 4, Opus 4.1)
- **File Attachments**: Support for text, code, data, and image files
- **URL Processing**: Automatic WebFetch tool enablement for URL analysis  
- **Streaming Support**: Real-time response streaming
- **Error Handling**: Comprehensive error reporting and timeout management
- **Flexible Path Configuration**: Custom Claude Code executable paths

## Technical Details

- **Timeout**: Commands timeout after 5 minutes to prevent hanging
- **Temporary Files**: Attachment content is safely handled via temporary files
- **Error Recovery**: Graceful handling of missing executables and other errors
- **Content Types**: Extensive MIME type support for attachments

## Troubleshooting

### Claude Code CLI Not Found

If you see an error about the Claude Code CLI not being found:

1. Ensure Claude Code is installed and accessible
2. Check that the executable is at `~/.claude/local/claude` or specify a custom path
3. Verify the executable has proper permissions

Example with custom path:

```bash
llm -m code/sonnet -o claude_path /path/to/your/claude "Your prompt"
```

### Timeout Issues

If commands are timing out:

- Complex prompts may take longer to process
- Large file attachments can increase processing time
- The default timeout is 5 minutes

### File Attachment Issues

If file attachments aren't working:

- Ensure files exist and are readable
- Check that file types are supported
- Large files may cause performance issues

## Development

To set up this plugin for development:

```bash
git clone https://github.com/paradise-runner/llm-claude-code.git
cd llm-claude-code-plugin
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

The plugin uses Python's `subprocess` module to execute the Claude Code CLI and supports:

- Multiple attachment types through MIME type detection
- Streaming responses via generator functions  
- Configurable command paths and options
- Robust error handling and cleanup

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.