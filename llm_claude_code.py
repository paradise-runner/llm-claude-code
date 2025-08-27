import subprocess
import tempfile
import os
from typing import Optional
import llm


@llm.hookimpl
def register_models(register):
    register(ClaudeCodeSonnetLatest())
    register(ClaudeCodeOpusLatest())
    register(ClaudeCodeOpus4())
    register(ClaudeCodeOpus4point1())
    register(ClaudeCodeSonnet4())


class ClaudeCodeBase(llm.Model):
    def __init__(self, claude_code_model_id: Optional[str] = None):
        super().__init__()
        self.claude_code_model_id = claude_code_model_id

    def __str__(self):
        return "Claude Code: {}".format(self.model_id)

    can_stream = True

    # Declare supported attachment types - be very permissive
    attachment_types = {
        "text/plain",
        "text/x-python",
        "text/javascript",
        "text/html",
        "text/css",
        "text/markdown",
        "text/x-c",
        "text/x-java-source",
        "text/x-sh",
        "application/json",
        "application/xml",
        "application/yaml",
        "application/octet-stream",
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
    }

    class Options(llm.Options):
        claude_path: Optional[str] = None

    def get_claude_command(self):
        """Get the claude command path from options or use default."""
        if (
            hasattr(self, "options")
            and self.options
            and hasattr(self.options, "claude_path")
            and self.options.claude_path
        ):
            return self.options.claude_path
        current_user_directory = os.path.expanduser("~")
        return os.path.join(current_user_directory, ".claude/local/claude")

    def execute(self, prompt, stream, response, conversation):
        """Execute Claude Code with the given prompt and attachments."""
        try:
            # Build the Claude Code command
            cmd = [self.get_claude_command()]

            # Add model parameter if specified
            if self.claude_code_model_id:
                cmd.extend(["--model", self.claude_code_model_id])

            # Build the enhanced prompt with file paths included
            enhanced_prompt = prompt.prompt
            temp_files = []

            if hasattr(prompt, "attachments") and prompt.attachments:
                file_paths = []
                urls = []
                for attachment in prompt.attachments:
                    if hasattr(attachment, "content") and attachment.content:
                        # Create temporary file for content
                        temp_fd, temp_path = tempfile.mkstemp(suffix=".txt")
                        try:
                            with os.fdopen(temp_fd, "w") as f:
                                # Handle both string and bytes content
                                content = attachment.content
                                if isinstance(content, bytes):
                                    content = content.decode("utf-8", errors="replace")
                                f.write(content)
                        except Exception:
                            # Fallback: write as binary
                            with os.fdopen(temp_fd, "wb") as f:
                                if isinstance(attachment.content, str):
                                    f.write(attachment.content.encode("utf-8"))
                                else:
                                    f.write(attachment.content)
                        temp_files.append(temp_path)
                        file_paths.append(temp_path)
                    elif (
                        hasattr(attachment, "path")
                        and attachment.path
                        and os.path.exists(attachment.path)
                    ):
                        file_paths.append(attachment.path)
                    elif hasattr(attachment, "url") and attachment.url:
                        urls.append(attachment.url)

                # Include file paths and URLs in the prompt itself
                attachments_to_add = []
                if file_paths:
                    attachments_to_add.extend(file_paths)
                if urls:
                    attachments_to_add.extend(urls)

                if attachments_to_add:
                    attachments_str = " ".join(attachments_to_add)
                    enhanced_prompt = f"{enhanced_prompt} {attachments_str}"

                # Add --allowed-tools WebFetch if URLs are present
                if urls:
                    cmd.extend(["--allowed-tools", "WebFetch"])
                    cmd.append("-p")

            # Add the enhanced prompt as the last argument
            cmd.append(enhanced_prompt)

            try:
                # Execute Claude Code
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode == 0:
                    yield result.stdout
                else:
                    error_msg = (
                        f"Claude Code error (exit code {result.returncode}):\n{result}"
                    )
                    yield error_msg

            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass

        except subprocess.TimeoutExpired:
            yield "Claude Code execution timed out (5 minutes)"
        except FileNotFoundError:
            yield f"Claude Code CLI not found at '{self.get_claude_command()}'. Please ensure claude command is available or set claude_path option."
        except Exception as e:
            yield f"Error executing Claude Code: {str(e)}"


class ClaudeCodeSonnetLatest(ClaudeCodeBase):
    """Claude Code Sonnet model provider."""

    model_id = "code/sonnet"

    def __init__(self):
        super().__init__(claude_code_model_id="sonnet")


class ClaudeCodeOpusLatest(ClaudeCodeBase):
    """Claude Code Opus model provider."""

    model_id = "code/opus"

    def __init__(self):
        super().__init__(claude_code_model_id="opus")


class ClaudeCodeOpus4(ClaudeCodeBase):
    """Claude Code Opus 4 model provider."""

    model_id = "code/opus-4"

    def __init__(self):
        super().__init__(claude_code_model_id="claude-opus-4-20250514")


class ClaudeCodeOpus4point1(ClaudeCodeBase):
    """Claude Code Opus 4.1 model provider."""

    model_id = "code/opus-4.1"

    def __init__(self):
        super().__init__(claude_code_model_id="claude-opus-4-1-20250805")


class ClaudeCodeSonnet4(ClaudeCodeBase):
    """Claude Code Sonnet 4 model provider."""

    model_id = "code/sonnet-4"

    def __init__(self):
        super().__init__(claude_code_model_id="claude-sonnet-4-20250514")
