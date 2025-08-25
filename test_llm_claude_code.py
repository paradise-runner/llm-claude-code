import pytest
import os
import subprocess
from unittest.mock import Mock, patch, mock_open
from llm_claude_code import (
    ClaudeCodeBase,
    ClaudeCodeSonnetLatest,
    ClaudeCodeOpusLatest,
    ClaudeCodeOpus4,
    ClaudeCodeOpus4point1,
    ClaudeCodeSonnet4,
    register_models,
)


class TestClaudeCodeBase:
    """Test cases for ClaudeCodeBase functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = ClaudeCodeBase("test-model")
        self.mock_prompt = Mock()
        self.mock_prompt.prompt = "test prompt"
        self.mock_prompt.attachments = []

    def test_init_with_model_id(self):
        """Test initialization with model ID."""
        model = ClaudeCodeBase("custom-model")
        assert model.claude_code_model_id == "custom-model"

    def test_init_without_model_id(self):
        """Test initialization without model ID."""
        model = ClaudeCodeBase()
        assert model.claude_code_model_id is None

    def test_str_representation(self):
        """Test string representation of model."""
        model = ClaudeCodeBase("test-model")
        model.model_id = "test/model"
        assert str(model) == "Claude Code: test/model"

    def test_can_stream(self):
        """Test that streaming is supported."""
        assert ClaudeCodeBase.can_stream is True

    def test_attachment_types(self):
        """Test supported attachment types."""
        expected_types = {
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
        assert ClaudeCodeBase.attachment_types == expected_types

    def test_get_claude_command_with_custom_path(self):
        """Test getting claude command with custom path from options."""
        options = Mock()
        options.claude_path = "/custom/path/claude"
        self.model.options = options

        assert self.model.get_claude_command() == "/custom/path/claude"

    def test_get_claude_command_default_path(self):
        """Test getting default claude command path."""
        expected_path = os.path.join(os.path.expanduser("~"), ".claude/local/claude")
        assert self.model.get_claude_command() == expected_path

    def test_get_claude_command_no_options(self):
        """Test getting claude command when options is None."""
        self.model.options = None
        expected_path = os.path.join(os.path.expanduser("~"), ".claude/local/claude")
        assert self.model.get_claude_command() == expected_path

    @patch("subprocess.run")
    def test_execute_basic_prompt(self, mock_subprocess):
        """Test execution with basic prompt and no attachments."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"

        result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert result == ["test output"]
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "test prompt" in call_args

    @patch("subprocess.run")
    def test_execute_with_model_id(self, mock_subprocess):
        """Test execution includes model parameter when model_id is set."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"

        list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        call_args = mock_subprocess.call_args[0][0]
        assert "--model" in call_args
        assert "test-model" in call_args

    @patch("subprocess.run")
    def test_execute_without_model_id(self, mock_subprocess):
        """Test execution without model parameter when model_id is None."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"

        model = ClaudeCodeBase()
        list(model.execute(self.mock_prompt, False, Mock(), Mock()))

        call_args = mock_subprocess.call_args[0][0]
        assert "--model" not in call_args

    @patch("subprocess.run")
    def test_execute_error_return_code(self, mock_subprocess):
        """Test execution with non-zero return code."""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stdout = "error output"

        result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert len(result) == 1
        assert "Claude Code error (exit code 1)" in result[0]

    @patch("subprocess.run")
    def test_execute_timeout(self, mock_subprocess):
        """Test execution timeout handling."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("cmd", 300)

        result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert result == ["Claude Code execution timed out (5 minutes)"]

    @patch("subprocess.run")
    def test_execute_file_not_found(self, mock_subprocess):
        """Test execution when Claude CLI is not found."""
        mock_subprocess.side_effect = FileNotFoundError()

        result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert len(result) == 1
        assert "Claude Code CLI not found" in result[0]
        assert "Please ensure claude command is available" in result[0]

    @patch("subprocess.run")
    def test_execute_generic_exception(self, mock_subprocess):
        """Test execution with generic exception."""
        mock_subprocess.side_effect = Exception("test error")

        result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert result == ["Error executing Claude Code: test error"]

    @patch("tempfile.mkstemp")
    @patch("subprocess.run")
    def test_execute_with_content_attachment(self, mock_subprocess, mock_mkstemp):
        """Test execution with attachment containing content."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"
        mock_mkstemp.return_value = (1, "/tmp/test_file")

        attachment = Mock()
        attachment.content = "test content"
        self.mock_prompt.attachments = [attachment]

        with patch("os.fdopen", mock_open()) as mock_file:
            with patch("os.unlink") as mock_unlink:
                result = list(
                    self.model.execute(self.mock_prompt, False, Mock(), Mock())
                )

        assert result == ["test output"]
        mock_file.assert_called_once()
        mock_unlink.assert_called_once_with("/tmp/test_file")

    @patch("tempfile.mkstemp")
    @patch("subprocess.run")
    def test_execute_with_bytes_content_attachment(self, mock_subprocess, mock_mkstemp):
        """Test execution with attachment containing bytes content."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"
        mock_mkstemp.return_value = (1, "/tmp/test_file")

        attachment = Mock()
        attachment.content = b"test bytes content"
        self.mock_prompt.attachments = [attachment]

        with patch("os.fdopen", mock_open()) as mock_file:
            with patch("os.unlink"):
                result = list(
                    self.model.execute(self.mock_prompt, False, Mock(), Mock())
                )

        assert result == ["test output"]
        mock_file.assert_called_once()

    @patch("tempfile.mkstemp")
    @patch("subprocess.run")
    def test_execute_with_content_write_error_fallback(
        self, mock_subprocess, mock_mkstemp
    ):
        """Test execution with content attachment that fails to write as text."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"
        mock_mkstemp.return_value = (1, "/tmp/test_file")

        attachment = Mock()
        attachment.content = "test content"
        self.mock_prompt.attachments = [attachment]

        # Mock text write to fail, binary write to succeed
        text_mock = mock_open()
        text_mock.return_value.write.side_effect = Exception("write error")
        binary_mock = mock_open()

        with patch("os.fdopen") as mock_fdopen:
            mock_fdopen.side_effect = [text_mock.return_value, binary_mock.return_value]
            with patch("os.unlink"):
                result = list(
                    self.model.execute(self.mock_prompt, False, Mock(), Mock())
                )

        assert result == ["test output"]
        assert mock_fdopen.call_count == 2

    @patch("tempfile.mkstemp")
    @patch("subprocess.run")
    def test_execute_with_bytes_content_fallback(self, mock_subprocess, mock_mkstemp):
        """Test execution with bytes content in fallback binary write mode."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"
        mock_mkstemp.return_value = (1, "/tmp/test_file")

        attachment = Mock()
        attachment.content = b"bytes content"
        self.mock_prompt.attachments = [attachment]

        # Mock text write to fail, then binary write succeeds with bytes
        text_mock = mock_open()
        text_mock.return_value.write.side_effect = Exception("write error")
        binary_mock = mock_open()

        with patch("os.fdopen") as mock_fdopen:
            mock_fdopen.side_effect = [text_mock.return_value, binary_mock.return_value]
            with patch("os.unlink"):
                result = list(
                    self.model.execute(self.mock_prompt, False, Mock(), Mock())
                )

        assert result == ["test output"]
        # Verify that the binary mock's write method was called with bytes
        binary_mock.return_value.write.assert_called_once_with(b"bytes content")

    @patch("subprocess.run")
    def test_execute_with_file_path_attachment(self, mock_subprocess):
        """Test execution with attachment containing file path."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"

        attachment = Mock()
        attachment.path = "/existing/file.txt"
        attachment.content = None
        self.mock_prompt.attachments = [attachment]

        with patch("os.path.exists", return_value=True):
            result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert result == ["test output"]
        call_args = mock_subprocess.call_args[0][0]
        assert "/existing/file.txt" in " ".join(call_args)

    @patch("subprocess.run")
    def test_execute_with_nonexistent_file_path_attachment(self, mock_subprocess):
        """Test execution with attachment containing non-existent file path."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"

        attachment = Mock()
        attachment.path = "/nonexistent/file.txt"
        attachment.content = None
        # Ensure hasattr works correctly by adding url attribute
        attachment.url = None
        self.mock_prompt.attachments = [attachment]

        with patch("os.path.exists", return_value=False):
            result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert result == ["test output"]
        call_args = mock_subprocess.call_args[0][0]
        assert "/nonexistent/file.txt" not in " ".join(call_args)

    @patch("subprocess.run")
    def test_execute_with_url_attachment(self, mock_subprocess):
        """Test execution with attachment containing URL."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"

        attachment = Mock()
        attachment.url = "https://example.com/file.txt"
        attachment.content = None
        attachment.path = None
        self.mock_prompt.attachments = [attachment]

        result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert result == ["test output"]
        call_args = mock_subprocess.call_args[0][0]
        assert "--allowed-tools" in call_args
        assert "WebFetch" in call_args
        assert "-p" in call_args
        assert "https://example.com/file.txt" in " ".join(call_args)

    @patch("subprocess.run")
    def test_execute_with_mixed_attachments(self, mock_subprocess):
        """Test execution with multiple different types of attachments."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"

        content_attachment = Mock()
        content_attachment.content = "test content"

        file_attachment = Mock()
        file_attachment.path = "/existing/file.txt"
        file_attachment.content = None

        url_attachment = Mock()
        url_attachment.url = "https://example.com/file.txt"
        url_attachment.content = None
        url_attachment.path = None

        self.mock_prompt.attachments = [
            content_attachment,
            file_attachment,
            url_attachment,
        ]

        with patch("tempfile.mkstemp", return_value=(1, "/tmp/test_file")):
            with patch("os.fdopen", mock_open()):
                with patch("os.path.exists", return_value=True):
                    with patch("os.unlink"):
                        result = list(
                            self.model.execute(self.mock_prompt, False, Mock(), Mock())
                        )

        assert result == ["test output"]
        call_args = mock_subprocess.call_args[0][0]
        assert "--allowed-tools" in call_args
        assert "WebFetch" in call_args

    @patch("tempfile.mkstemp")
    @patch("subprocess.run")
    def test_temp_file_cleanup_on_error(self, mock_subprocess, mock_mkstemp):
        """Test that temporary files are cleaned up even when subprocess fails."""
        mock_subprocess.side_effect = Exception("subprocess error")
        mock_mkstemp.return_value = (1, "/tmp/test_file")

        attachment = Mock()
        attachment.content = "test content"
        self.mock_prompt.attachments = [attachment]

        with patch("os.fdopen", mock_open()):
            with patch("os.unlink") as mock_unlink:
                result = list(
                    self.model.execute(self.mock_prompt, False, Mock(), Mock())
                )

        assert "Error executing Claude Code" in result[0]
        mock_unlink.assert_called_once_with("/tmp/test_file")

    @patch("tempfile.mkstemp")
    @patch("subprocess.run")
    def test_temp_file_cleanup_failure_ignored(self, mock_subprocess, mock_mkstemp):
        """Test that temp file cleanup failures are silently ignored."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "test output"
        mock_mkstemp.return_value = (1, "/tmp/test_file")

        attachment = Mock()
        attachment.content = "test content"
        self.mock_prompt.attachments = [attachment]

        with patch("os.fdopen", mock_open()):
            with patch("os.unlink", side_effect=OSError("cleanup failed")):
                result = list(
                    self.model.execute(self.mock_prompt, False, Mock(), Mock())
                )

        # Should still succeed despite cleanup failure
        assert result == ["test output"]

    def test_execute_attachment_without_content_path_or_url(self):
        """Test execution with attachment that has no content, path, or URL."""
        attachment = Mock()
        attachment.content = None
        attachment.path = None
        attachment.url = None
        self.mock_prompt.attachments = [attachment]

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "test output"

            result = list(self.model.execute(self.mock_prompt, False, Mock(), Mock()))

        assert result == ["test output"]
        # Should not add any attachments to command
        call_args = mock_subprocess.call_args[0][0]
        assert len([arg for arg in call_args if arg == "test prompt"]) == 1


class TestSpecificModelClasses:
    """Test cases for specific Claude Code model implementations."""

    def test_sonnet_latest_model(self):
        """Test ClaudeCodeSonnetLatest initialization."""
        model = ClaudeCodeSonnetLatest()
        assert model.model_id == "code/sonnet"
        assert model.claude_code_model_id == "sonnet"

    def test_opus_latest_model(self):
        """Test ClaudeCodeOpusLatest initialization."""
        model = ClaudeCodeOpusLatest()
        assert model.model_id == "code/opus"
        assert model.claude_code_model_id == "opus"

    def test_opus_4_model(self):
        """Test ClaudeCodeOpus4 initialization."""
        model = ClaudeCodeOpus4()
        assert model.model_id == "code/opus-4"
        assert model.claude_code_model_id == "claude-opus-4-20250514"

    def test_opus_4_point_1_model(self):
        """Test ClaudeCodeOpus4point1 initialization."""
        model = ClaudeCodeOpus4point1()
        assert model.model_id == "code/opus-4.1"
        assert model.claude_code_model_id == "claude-opus-4-1-20250805"

    def test_sonnet_4_model(self):
        """Test ClaudeCodeSonnet4 initialization."""
        model = ClaudeCodeSonnet4()
        assert model.model_id == "code/sonnet-4"
        assert model.claude_code_model_id == "claude-sonnet-4-20250514"

    def test_all_models_inherit_from_base(self):
        """Test that all model classes inherit from ClaudeCodeBase."""
        models = [
            ClaudeCodeSonnetLatest(),
            ClaudeCodeOpusLatest(),
            ClaudeCodeOpus4(),
            ClaudeCodeOpus4point1(),
            ClaudeCodeSonnet4(),
        ]

        for model in models:
            assert isinstance(model, ClaudeCodeBase)
            assert hasattr(model, "can_stream")
            assert hasattr(model, "attachment_types")
            assert hasattr(model, "execute")


class TestModelRegistration:
    """Test cases for model registration functionality."""

    def test_register_models(self):
        """Test that register_models registers all model classes."""
        mock_register = Mock()

        register_models(mock_register)

        assert mock_register.call_count == 5
        registered_classes = [
            call[0][0].__class__ for call in mock_register.call_args_list
        ]

        expected_classes = [
            ClaudeCodeSonnetLatest,
            ClaudeCodeOpusLatest,
            ClaudeCodeOpus4,
            ClaudeCodeOpus4point1,
            ClaudeCodeSonnet4,
        ]

        for expected_class in expected_classes:
            assert expected_class in registered_classes


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_prompt(self):
        """Test execution with empty prompt."""
        model = ClaudeCodeBase()
        mock_prompt = Mock()
        mock_prompt.prompt = ""
        mock_prompt.attachments = []

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "output"

            result = list(model.execute(mock_prompt, False, Mock(), Mock()))

        assert result == ["output"]

    def test_very_large_content_attachment(self):
        """Test execution with very large content attachment."""
        model = ClaudeCodeBase()
        mock_prompt = Mock()
        mock_prompt.prompt = "test"

        # Create large content (1MB of text)
        large_content = "x" * (1024 * 1024)
        attachment = Mock()
        attachment.content = large_content
        mock_prompt.attachments = [attachment]

        with patch("tempfile.mkstemp", return_value=(1, "/tmp/large_file")):
            with patch("os.fdopen", mock_open()) as mock_file:
                with patch("subprocess.run") as mock_subprocess:
                    mock_subprocess.return_value.returncode = 0
                    mock_subprocess.return_value.stdout = "output"

                    with patch("os.unlink"):
                        result = list(model.execute(mock_prompt, False, Mock(), Mock()))

        assert result == ["output"]
        mock_file.assert_called_once()

    def test_unicode_content_attachment(self):
        """Test execution with unicode content in attachment."""
        model = ClaudeCodeBase()
        mock_prompt = Mock()
        mock_prompt.prompt = "test"

        # Unicode content with various characters
        unicode_content = "Hello 🌍 世界 🚀 测试"
        attachment = Mock()
        attachment.content = unicode_content
        mock_prompt.attachments = [attachment]

        with patch("tempfile.mkstemp", return_value=(1, "/tmp/unicode_file")):
            with patch("os.fdopen", mock_open()):
                with patch("subprocess.run") as mock_subprocess:
                    mock_subprocess.return_value.returncode = 0
                    mock_subprocess.return_value.stdout = "output"

                    with patch("os.unlink"):
                        result = list(model.execute(mock_prompt, False, Mock(), Mock()))

        assert result == ["output"]

    def test_malformed_attachment(self):
        """Test execution with malformed attachment object."""
        model = ClaudeCodeBase()
        mock_prompt = Mock()
        mock_prompt.prompt = "test"

        # Attachment missing expected attributes
        attachment = Mock(spec=[])  # Empty spec means no attributes
        mock_prompt.attachments = [attachment]

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "output"

            result = list(model.execute(mock_prompt, False, Mock(), Mock()))

        assert result == ["output"]

    def test_options_object_variations(self):
        """Test different variations of options object."""
        model = ClaudeCodeBase()

        # Test with options object that has no claude_path attribute
        options = Mock(spec=[])
        model.options = options
        default_path = os.path.join(os.path.expanduser("~"), ".claude/local/claude")
        assert model.get_claude_command() == default_path

        # Test with options object that has claude_path but it's None
        options = Mock()
        options.claude_path = None
        model.options = options
        assert model.get_claude_command() == default_path

        # Test with options object that has empty claude_path
        options.claude_path = ""
        model.options = options
        assert model.get_claude_command() == default_path

    def test_prompt_without_attachments_attribute(self):
        """Test execution with prompt object missing attachments attribute."""
        model = ClaudeCodeBase()
        mock_prompt = Mock(spec=["prompt"])  # Only has prompt attribute
        mock_prompt.prompt = "test prompt"

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "output"

            result = list(model.execute(mock_prompt, False, Mock(), Mock()))

        assert result == ["output"]

    @patch("subprocess.run")
    def test_subprocess_timeout_with_cleanup(self, mock_subprocess):
        """Test that timeout properly cleans up temp files."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("cmd", 300)

        model = ClaudeCodeBase()
        mock_prompt = Mock()
        mock_prompt.prompt = "test"

        attachment = Mock()
        attachment.content = "test content"
        mock_prompt.attachments = [attachment]

        with patch("tempfile.mkstemp", return_value=(1, "/tmp/timeout_file")):
            with patch("os.fdopen", mock_open()):
                with patch("os.unlink") as mock_unlink:
                    result = list(model.execute(mock_prompt, False, Mock(), Mock()))

        assert result == ["Claude Code execution timed out (5 minutes)"]
        mock_unlink.assert_called_once_with("/tmp/timeout_file")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
