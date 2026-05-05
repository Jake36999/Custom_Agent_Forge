import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
INGESTOR_SCRIPT = REPO_ROOT / "Autonomous_Ingestor.ps1"
RESULT_MARKER = "RESULT_JSON:"

# HD1: Environment-variable override for shell binary path.
# Standardise on pwsh (cross-platform) and fall back only when the env var
# is unset.  Validates that the resolved path actually exists on disk to
# prevent path-hijacking.
POWERSHELL_ENV_VAR = "ALETHEIA_PWSH_PATH"


def _powershell_executable() -> str:
    env_override = os.environ.get(POWERSHELL_ENV_VAR)
    if env_override:
        resolved = Path(env_override)
        if resolved.is_file():
            return str(resolved)
        raise AssertionError(
            f"{POWERSHELL_ENV_VAR} is set to '{env_override}' but the file does not exist."
        )
    # Prefer pwsh (cross-platform) over legacy powershell
    for candidate in ("pwsh", "powershell"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise AssertionError(
        "PowerShell executable not found. Set the "
        f"{POWERSHELL_ENV_VAR} environment variable to the absolute path of pwsh."
    )


def _powershell_args(executable: str, command: str) -> list[str]:
    args = [executable, "-NoProfile"]
    if Path(executable).name.lower().startswith("powershell"):
        args.extend(["-ExecutionPolicy", "Bypass"])
    args.extend(["-Command", command])
    return args


def _ps_literal(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _run_powershell(body: str) -> dict:
    executable = _powershell_executable()
    command = textwrap.dedent(
        f"""
        $ErrorActionPreference = 'Stop'
        $env:ALETHEIA_SKIP_INGESTOR_AUTORUN = '1'
        . '{_ps_literal(INGESTOR_SCRIPT)}'

        function Set-TestGlobals {{
            param([string]$Root)
            $Global:StagingDir = Join-Path $Root 'staging'
            $Global:OutputDir = Join-Path $Root 'output'
            $Global:LogCsv = Join-Path $Root 'github_ingestion_log.csv'
            $Global:LandingPadDir = Join-Path $Root 'landing'
            $Global:FailedWorkspaceDir = Join-Path $Global:StagingDir 'failed_workspaces'
            Ensure-IngestionDirectories
        }}

        {body}
        """
    )
    env = os.environ.copy()
    env["ALETHEIA_SKIP_INGESTOR_AUTORUN"] = "1"
    result = subprocess.run(
        _powershell_args(executable, command),
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise AssertionError(
            "PowerShell scenario failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    for line in reversed(result.stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return json.loads(line[len(RESULT_MARKER):])

    raise AssertionError(f"PowerShell scenario did not emit a result payload.\nSTDOUT:\n{result.stdout}")


def test_manifest_backlog_detection() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        body = textwrap.dedent(
            f"""
            $Root = '{_ps_literal(root)}'
            Set-TestGlobals -Root $Root

            $LandingFile = Join-Path $Global:LandingPadDir 'repos.txt'
            @(
                'https://github.com/example/completed',
                'https://github.com/example/pending',
                'https://github.com/example/malformed'
            ) | Set-Content -LiteralPath $LandingFile -Encoding UTF8

            $ValidManifest = [ordered]@{{
                project_name = 'completed'
                repo_url = 'https://github.com/example/completed'
                ingestion_status = 'Completed'
                schema_version = 'aletheia_skill_dossier_v1'
                chunks_processed = 2
                timestamp = '2026-03-25T17:38:00Z'
            }}
            $ValidManifest | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Get-IngestionManifestPath -ProjectName 'completed') -Encoding UTF8
            Set-Content -LiteralPath (Get-IngestionManifestPath -ProjectName 'malformed') -Value '{{bad json' -Encoding UTF8

            Build-IngestionBacklog

            $Records = @(Import-Csv -Path $Global:LogCsv | Sort-Object ProjectName)
            $Statuses = [ordered]@{{}}
            foreach ($Record in $Records) {{
                $Statuses[$Record.ProjectName] = $Record.Status
            }}

            Write-Output ("{RESULT_MARKER}" + ($Statuses | ConvertTo-Json -Compress))
            """
        )
        statuses = _run_powershell(body)

    assert statuses["completed"] == "Completed", "Expected valid manifest to mark repo as completed."
    assert statuses["pending"] == "Pending", "Expected missing manifest to leave repo pending."
    assert statuses["malformed"] == "Pending", "Expected malformed manifest to leave repo pending."


def test_zero_bundle_failure_retains_workspace() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        body = textwrap.dedent(
            f"""
            $Root = '{_ps_literal(root)}'
            Set-TestGlobals -Root $Root

            function Write-EnvironmentSensor {{ }}
            function Test-CompilerReadiness {{ }}
            function Invoke-CheckedExternal {{
                param ([string]$Executable, [string[]]$Arguments, [string]$ErrorMessage)
                if ($Executable -eq 'git') {{
                    New-Item -ItemType Directory -Force -Path $Arguments[2] | Out-Null
                    return
                }}
                $ScriptName = [IO.Path]::GetFileName($Arguments[0])
                if ($ScriptName -eq 'semantic_slicer_AG.py') {{
                    return
                }}
                if ($ScriptName -eq 'cognitive_processor.py') {{
                    throw 'Compiler should not run during zero-bundle scenario.'
                }}
                throw $ErrorMessage
            }}

            Set-Content -LiteralPath (Join-Path $Global:LandingPadDir 'repos.txt') -Value 'https://github.com/example/zerobundle' -Encoding UTF8

            Build-IngestionBacklog
            Start-AutonomousIngestion

            $TargetFolder = Join-Path $Global:StagingDir 'zerobundle'
            $Records = @(Import-Csv -Path $Global:LogCsv)
            $Record = $Records | Where-Object {{ $_.ProjectName -eq 'zerobundle' }} | Select-Object -First 1
            $Result = [ordered]@{{
                status = $Record.Status
                manifest_exists = Test-Path (Get-IngestionManifestPath -ProjectName 'zerobundle')
                retained_failed_workspace = (@(Get-ChildItem -Path $Global:FailedWorkspaceDir -Directory -Filter 'zerobundle_*' -ErrorAction SilentlyContinue).Count -gt 0)
                target_exists = Test-Path $TargetFolder
            }}

            Write-Output ("{RESULT_MARKER}" + ($Result | ConvertTo-Json -Compress -Depth 5))
            """
        )
        result = _run_powershell(body)

    assert result["status"] == "Failed", "Expected zero-bundle repo to be marked failed."
    assert result["manifest_exists"] is False, "Expected zero-bundle failure to skip manifest creation."
    assert result["retained_failed_workspace"] is True, "Expected zero-bundle failure to retain the workspace."
    assert result["target_exists"] is False, "Expected failed workspace to be moved out of staging."


def test_successful_compile_writes_manifest_and_purges_workspace() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        body = textwrap.dedent(
            f"""
            $Root = '{_ps_literal(root)}'
            Set-TestGlobals -Root $Root

            function Write-EnvironmentSensor {{ }}
            function Test-CompilerReadiness {{ }}
            function Invoke-CheckedExternal {{
                param ([string]$Executable, [string[]]$Arguments, [string]$ErrorMessage)
                if ($Executable -eq 'git') {{
                    New-Item -ItemType Directory -Force -Path $Arguments[2] | Out-Null
                    return
                }}
                $ScriptName = [IO.Path]::GetFileName($Arguments[0])
                if ($ScriptName -eq 'semantic_slicer_AG.py') {{
                    Set-Content -LiteralPath (Join-Path $Arguments[1] 'raw_ast_bundle.json') -Value '{{}}' -Encoding UTF8
                    return
                }}
                if ($ScriptName -eq 'cognitive_processor.py') {{
                    Set-Content -LiteralPath $Arguments[3] -Value "schema: aletheia_skill_dossier_v1`ncapability_injection:`n  compiled_skills: []" -Encoding UTF8
                    return
                }}
                throw $ErrorMessage
            }}

            Set-Content -LiteralPath (Join-Path $Global:LandingPadDir 'repos.txt') -Value 'https://github.com/example/successrepo' -Encoding UTF8

            Build-IngestionBacklog
            Start-AutonomousIngestion

            $TargetFolder = Join-Path $Global:StagingDir 'successrepo'
            $ManifestPath = Get-IngestionManifestPath -ProjectName 'successrepo'
            $Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
            $Records = @(Import-Csv -Path $Global:LogCsv)
            $Record = $Records | Where-Object {{ $_.ProjectName -eq 'successrepo' }} | Select-Object -First 1
            $Result = [ordered]@{{
                status = $Record.Status
                manifest = $Manifest
                manifest_exists = Test-Path $ManifestPath
                target_exists = Test-Path $TargetFolder
                output_exists = Test-Path (Join-Path $Global:OutputDir 'KNOWLEDGE_MATRIX_successrepo_UNIFIED.yaml')
            }}

            Write-Output ("{RESULT_MARKER}" + ($Result | ConvertTo-Json -Compress -Depth 6))
            """
        )
        result = _run_powershell(body)

    assert result["status"] == "Completed", "Expected successful repo to be marked completed."
    assert result["manifest_exists"] is True, "Expected successful compile to write a manifest."
    assert result["target_exists"] is False, "Expected successful compile to purge the workspace."
    assert result["output_exists"] is True, "Expected successful compile to emit a unified YAML."
    manifest = result["manifest"]
    assert manifest["project_name"] == "successrepo", "Expected manifest to record the project name."
    assert manifest["repo_url"] == "https://github.com/example/successrepo", "Expected manifest to record the repo URL."
    assert manifest["ingestion_status"] == "Completed", "Expected manifest to record completed ingestion."
    assert manifest["schema_version"] == "aletheia_skill_dossier_v1", "Expected manifest schema version to match the dossier schema."
    assert manifest["chunks_processed"] == 1, "Expected manifest chunk count to match the compiled bundle count."


def test_compiler_failure_retains_workspace_without_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        body = textwrap.dedent(
            f"""
            $Root = '{_ps_literal(root)}'
            Set-TestGlobals -Root $Root

            function Write-EnvironmentSensor {{ }}
            function Test-CompilerReadiness {{ }}
            function Invoke-CheckedExternal {{
                param ([string]$Executable, [string[]]$Arguments, [string]$ErrorMessage)
                if ($Executable -eq 'git') {{
                    New-Item -ItemType Directory -Force -Path $Arguments[2] | Out-Null
                    return
                }}
                $ScriptName = [IO.Path]::GetFileName($Arguments[0])
                if ($ScriptName -eq 'semantic_slicer_AG.py') {{
                    Set-Content -LiteralPath (Join-Path $Arguments[1] 'raw_ast_bundle.json') -Value '{{}}' -Encoding UTF8
                    return
                }}
                if ($ScriptName -eq 'cognitive_processor.py') {{
                    throw 'Synthetic compiler failure.'
                }}
                throw $ErrorMessage
            }}

            Set-Content -LiteralPath (Join-Path $Global:LandingPadDir 'repos.txt') -Value 'https://github.com/example/compilerfail' -Encoding UTF8

            Build-IngestionBacklog
            Start-AutonomousIngestion

            $TargetFolder = Join-Path $Global:StagingDir 'compilerfail'
            $Records = @(Import-Csv -Path $Global:LogCsv)
            $Record = $Records | Where-Object {{ $_.ProjectName -eq 'compilerfail' }} | Select-Object -First 1
            $Result = [ordered]@{{
                status = $Record.Status
                manifest_exists = Test-Path (Get-IngestionManifestPath -ProjectName 'compilerfail')
                retained_failed_workspace = (@(Get-ChildItem -Path $Global:FailedWorkspaceDir -Directory -Filter 'compilerfail_*' -ErrorAction SilentlyContinue).Count -gt 0)
                target_exists = Test-Path $TargetFolder
            }}

            Write-Output ("{RESULT_MARKER}" + ($Result | ConvertTo-Json -Compress -Depth 5))
            """
        )
        result = _run_powershell(body)

    assert result["status"] == "Failed", "Expected compiler failure to mark the repo failed."
    assert result["manifest_exists"] is False, "Expected compiler failure to skip manifest creation."
    assert result["retained_failed_workspace"] is True, "Expected compiler failure to retain the workspace."
    assert result["target_exists"] is False, "Expected failed workspace to be moved out of staging."


def main() -> None:
    tests = [
        test_manifest_backlog_detection,
        test_zero_bundle_failure_retains_workspace,
        test_successful_compile_writes_manifest_and_purges_workspace,
        test_compiler_failure_retains_workspace_without_manifest,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
