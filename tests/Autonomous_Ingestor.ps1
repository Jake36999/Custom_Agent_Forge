# ==============================================================================
# SOVEREIGN KERNEL: AUTONOMOUS ROLLING-BUFFER INGESTOR (PORTABLE + LANDING PAD)
# ==============================================================================

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Global:StagingDir = "D:\staging_point"
$Global:OutputDir = "D:\Aletheia_Knowledge_Complexes"
$Global:LogCsv = "$PSScriptRoot\github_ingestion_log.csv"
$Global:LandingPadDir = "$PSScriptRoot\Landing_Pad"
$Global:MaxCapacityBytes = 40GB # 80% of 50GB
$Global:FailedWorkspaceDir = Join-Path $Global:StagingDir "failed_workspaces"

function Ensure-IngestionDirectories {
    $Paths = @(
        $Global:StagingDir,
        $Global:OutputDir,
        $Global:LandingPadDir,
        $Global:FailedWorkspaceDir
    )
    foreach ($Path in $Paths) {
        if (-not (Test-Path $Path)) {
            New-Item -ItemType Directory -Force -Path $Path | Out-Null
        }
    }
}

function Get-IngestionManifestPath {
    param ([string]$ProjectName)
    return Join-Path $Global:OutputDir "${ProjectName}_INGESTION_MANIFEST.json"
}

function Get-IngestionManifestRecord {
    param ([string]$ProjectName)

    $ManifestPath = Get-IngestionManifestPath -ProjectName $ProjectName
    if (-not (Test-Path $ManifestPath)) {
        return $null
    }

    try {
        $Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop
    } catch {
        return $null
    }

    if ($null -eq $Manifest) {
        return $null
    }

    $RequiredProps = @("project_name", "repo_url", "ingestion_status", "schema_version", "chunks_processed", "timestamp")
    foreach ($Prop in $RequiredProps) {
        if ($Manifest.PSObject.Properties.Name -notcontains $Prop) {
            return $null
        }
    }

    try {
        $ChunksProcessed = [int]$Manifest.chunks_processed
    } catch {
        return $null
    }

    if ($Manifest.project_name -ne $ProjectName) {
        return $null
    }
    if ([string]::IsNullOrWhiteSpace([string]$Manifest.repo_url)) {
        return $null
    }
    if ($Manifest.ingestion_status -ne "Completed") {
        return $null
    }
    if ($Manifest.schema_version -ne "aletheia_skill_dossier_v1") {
        return $null
    }
    if ($ChunksProcessed -le 0) {
        return $null
    }

    return $Manifest
}

function Write-IngestionManifest {
    param (
        [string]$ProjectName,
        [string]$RepoUrl,
        [int]$ChunksProcessed
    )

    Ensure-IngestionDirectories

    $ManifestPath = Get-IngestionManifestPath -ProjectName $ProjectName
    $TempPath = Join-Path $Global:OutputDir ("{0}.tmp.{1}" -f $ProjectName, ([guid]::NewGuid().ToString("N")))
    $Manifest = [ordered]@{
        project_name = $ProjectName
        repo_url = $RepoUrl
        ingestion_status = "Completed"
        schema_version = "aletheia_skill_dossier_v1"
        chunks_processed = $ChunksProcessed
        timestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
    }

    $ManifestJson = $Manifest | ConvertTo-Json -Depth 3
    Set-Content -LiteralPath $TempPath -Value $ManifestJson -Encoding UTF8
    Move-Item -LiteralPath $TempPath -Destination $ManifestPath -Force
    return $ManifestPath
}

# ------------------------------------------------------------------------------
# STEP 1: Extract links from Landing Pad and build the Smart CSV backlog
# ------------------------------------------------------------------------------
function Build-IngestionBacklog {
    Ensure-IngestionDirectories

    $SourceFiles = @(
        Get-ChildItem -Path $Global:LandingPadDir -Filter "*.txt" |
            Sort-Object FullName |
            Select-Object -ExpandProperty FullName
    )
    
    if ($SourceFiles.Count -eq 0) {
        Write-Host "[!] No .txt files found in Landing Pad: $Global:LandingPadDir" -ForegroundColor Yellow
        Write-Host "    Please drop your text files with GitHub links into this folder." -ForegroundColor Gray
        return
    }

    $Backlog = @()
    
    foreach ($File in $SourceFiles) {
        Write-Host "[*] Scanning $File for GitHub links..." -ForegroundColor Cyan
        $Content = @(Get-Content $File)
        foreach ($Line in $Content) {
            if ($Line -match "(https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+(?:\.git)?)") {
                $Url = $matches[0]
                $ProjectName = ($Url -split "/")[-1] -replace "\.git$", ""
                
                # SELF-HEALING LOGIC: Check if a valid repo manifest already exists
                $CurrentStatus = "Pending"
                $Manifest = Get-IngestionManifestRecord -ProjectName $ProjectName
                if ($Manifest) {
                    $CurrentStatus = "Completed"
                }
                
                $Backlog += [PSCustomObject]@{
                    ProjectName = $ProjectName
                    RepoUrl = $Url
                    Status = $CurrentStatus
                }
            }
        }
    }
    
    # Deduplicate and save to local CSV
    if ($Backlog.Count -gt 0) {
        $Backlog = $Backlog | Sort-Object RepoUrl -Unique
        $Backlog | Export-Csv -Path $Global:LogCsv -NoTypeInformation
        
        $PendingCount = @($Backlog | Where-Object { $_.Status -eq "Pending" }).Count
        $CompletedCount = @($Backlog | Where-Object { $_.Status -eq "Completed" }).Count
        
        Write-Host "[OK] Backlog synced to $Global:LogCsv" -ForegroundColor Green
        Write-Host "  -> Pending: $PendingCount (Will be ingested)" -ForegroundColor Yellow
        Write-Host "  -> Completed: $CompletedCount (Already recorded by ingestion manifests)" -ForegroundColor DarkGray
    } else {
        Write-Host "[!] No valid GitHub links found in the Landing Pad files." -ForegroundColor Yellow
    }
}

# ------------------------------------------------------------------------------
# HELPER: Get Directory Size
# ------------------------------------------------------------------------------
function Get-DirSize {
    param ([string]$Path)
    if (Test-Path $Path) {
        $Files = @(Get-ChildItem -Path $Path -Recurse -File -ErrorAction SilentlyContinue)
        if ($Files.Count -eq 0) {
            return 0
        }
        $Size = ($Files | Measure-Object -Property Length -Sum).Sum
        if ($null -eq $Size) { return 0 } else { return $Size }
    }
    return 0
}

function Invoke-CheckedExternal {
    param (
        [string]$Executable,
        [string[]]$Arguments,
        [string]$ErrorMessage
    )

    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$ErrorMessage (exit code $exitCode)"
    }
}

function Move-ToFailedWorkspace {
    param (
        [string]$ProjectName,
        [string]$SourcePath
    )

    if (-not (Test-Path $SourcePath)) {
        return $null
    }

    $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $Destination = Join-Path $Global:FailedWorkspaceDir "${ProjectName}_$Timestamp"
    Move-Item -LiteralPath $SourcePath -Destination $Destination -Force
    return $Destination
}

# HD2: Automated 24-hour TTL for forensic failed workspace snapshots.
# Prevents disk saturation from stagnant failure artifacts.
$Global:FailedWorkspaceTTLHours = 24

function Invoke-FailedWorkspaceTTL {
    if (-not (Test-Path $Global:FailedWorkspaceDir)) {
        return
    }

    $Cutoff = (Get-Date).AddHours(-$Global:FailedWorkspaceTTLHours)
    $Expired = @(
        Get-ChildItem -Path $Global:FailedWorkspaceDir -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.CreationTime -lt $Cutoff }
    )

    foreach ($Dir in $Expired) {
        Write-Host "  [TTL] Purging expired forensic workspace: $($Dir.Name) (created $($Dir.CreationTime))" -ForegroundColor DarkYellow
        Remove-Item -LiteralPath $Dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($Expired.Count -gt 0) {
        Write-Host "[TTL] Purged $($Expired.Count) expired failed workspace(s)." -ForegroundColor Yellow
    }
}

function Test-CompilerReadiness {
    Write-Host "[*] Running compiler import/compile gate..." -ForegroundColor Cyan
    $SourceFiles = @(
        (Join-Path $PSScriptRoot "cognitive_processor.py"),
        (Join-Path $PSScriptRoot "dag_runtime.py"),
        (Join-Path $PSScriptRoot "acs_engine.py"),
        (Join-Path $PSScriptRoot "dataset_formatter.py")
    )
    $Arguments = @("-m", "py_compile") + $SourceFiles
    Invoke-CheckedExternal -Executable "python" -Arguments $Arguments -ErrorMessage "Compiler preflight failed"
    Write-Host "[OK] Compiler preflight passed." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# STEP 2 & 3: The Rolling Buffer Batch Processor
# ------------------------------------------------------------------------------
function Write-EnvironmentSensor {
    $envMeta = @{}
    
    # 1. Hardware Metrics
    $envMeta.Hardware = @{
        CPU_Count = [Environment]::ProcessorCount
        RAM_GB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)
        GPUs = @(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name)
    }
    $envMeta.Hardware.HasCUDA = ($envMeta.Hardware.GPUs | Where-Object { $_ -match 'NVIDIA' }) -ne $null
    
    # 2. Toolchains & Runtimes (Extracting Absolute Paths)
    $envMeta.Toolchains = @{}
    
    $pythonPath = Get-Command python -ErrorAction SilentlyContinue
    $envMeta.Toolchains.Python = if ($pythonPath) { $pythonPath.Source } else { $null }

    $nvccPath = Get-Command nvcc -ErrorAction SilentlyContinue
    $envMeta.Toolchains.NVCC = if ($nvccPath) { $nvccPath.Source } else { $null }

    $mpiPath = Get-Command mpiexec -ErrorAction SilentlyContinue
    $envMeta.Toolchains.MPI = if ($mpiPath) { $mpiPath.Source } else { $null }
    
    # 3. Sandbox Engine Status (For iterative patch validation)
    $dockerStatus = Get-Service docker -ErrorAction SilentlyContinue
    $envMeta.Sandbox = @{
        DockerRunning = if ($dockerStatus -and $dockerStatus.Status -eq 'Running') { $true } else { $false }
    }

    # 4. Connectivity (Safe to initiate rolling buffer?)
    $ping = Test-NetConnection -ComputerName "github.com" -Port 443 -WarningAction SilentlyContinue
    $envMeta.Connectivity = @{
        GitHubReachable = if ($ping) { $ping.TcpTestSucceeded } else { $false }
    }

    # OS & Timestamp
    $envMeta.OS = (Get-CimInstance Win32_OperatingSystem).Caption
    $envMeta.Timestamp = (Get-Date).ToString('s')
    
    # Write to JSON
    $envJson = $envMeta | ConvertTo-Json -Depth 4
    $envPath = Join-Path $PSScriptRoot 'environment.json'
    $envJson | Set-Content -Path $envPath -Encoding UTF8
    Write-Host "[ENV] Advanced Environment metadata written to $envPath" -ForegroundColor DarkGray
}

function Start-AutonomousIngestion {
    Ensure-IngestionDirectories
    Invoke-FailedWorkspaceTTL
    Write-EnvironmentSensor
    Test-CompilerReadiness

    if (-not (Test-Path $Global:LogCsv)) {
        Write-Host "[!] Log CSV not found. Please run Build-IngestionBacklog first." -ForegroundColor Red
        return
    }

    $Records = Import-Csv -Path $Global:LogCsv
    $Pending = @(
        $Records |
            Where-Object { $_.Status -eq "Pending" } |
            Sort-Object ProjectName, RepoUrl
    )
    
    if ($Pending.Count -eq 0) {
        Write-Host "[!] No pending repositories found in log." -ForegroundColor Yellow
        return
    }

    $BatchSize = 5
    $i = 0

    while ($i -lt $Pending.Count) {
        Write-Host "`n===========================================================" -ForegroundColor Magenta
        Write-Host "[->] INITIATING DOWNLOAD BATCH..." -ForegroundColor Magenta
        Write-Host "===========================================================" -ForegroundColor Magenta

        $CurrentBatch = @()
        $BatchAttemptCount = 0
        
        for ($b = 0; $b -lt $BatchSize -and ($i + $b) -lt $Pending.Count; $b++) {
            $Repo = $Pending[$i + $b]
            $TargetFolder = Join-Path $Global:StagingDir $Repo.ProjectName
            $ManifestPath = Get-IngestionManifestPath -ProjectName $Repo.ProjectName
            $BatchAttemptCount += 1
            
            $CurrentSize = Get-DirSize -Path $Global:StagingDir
            if ($CurrentSize -ge $Global:MaxCapacityBytes) {
                $SizeGB = [math]::Round($CurrentSize / 1GB, 2)
                Write-Host "[!] Capacity threshold reached ($SizeGB GB / 40 GB). Halting downloads to process current batch." -ForegroundColor Yellow
                $BatchAttemptCount -= 1
                break
            }

            try {
                if (Test-Path $ManifestPath) {
                    Remove-Item -LiteralPath $ManifestPath -Force -ErrorAction SilentlyContinue
                }

                if (Test-Path $TargetFolder) {
                    $RetainedExisting = Move-ToFailedWorkspace -ProjectName $Repo.ProjectName -SourcePath $TargetFolder
                    Write-Host "  [!] Existing workspace moved to $RetainedExisting before fresh clone." -ForegroundColor DarkYellow
                }

                Write-Host " -> Cloning $($Repo.ProjectName)..." -ForegroundColor Gray
                Invoke-CheckedExternal -Executable "git" -Arguments @("clone", $Repo.RepoUrl, $TargetFolder) -ErrorMessage "Git clone failed for $($Repo.ProjectName)"
                $CurrentBatch += $Repo
            } catch {
                Write-Host "  [ERROR] Clone failed for $($Repo.ProjectName): $($_.Exception.Message)" -ForegroundColor Red
                if (Test-Path $TargetFolder) {
                    $RetainedClone = Move-ToFailedWorkspace -ProjectName $Repo.ProjectName -SourcePath $TargetFolder
                    Write-Host "  [!] Retained failed clone workspace at $RetainedClone" -ForegroundColor DarkYellow
                }
                $Repo.Status = "Failed"
            }
        }

        if ($CurrentBatch.Count -eq 0 -and $BatchAttemptCount -eq 0) {
            Write-Host "[!] No repositories could be added to the current batch. Resolve capacity or retained workspace issues before retrying." -ForegroundColor Yellow
            break
        }

        Write-Host "`n===========================================================" -ForegroundColor Cyan
        Write-Host "[*] COMPILING KNOWLEDGE MATRICES..." -ForegroundColor Cyan
        Write-Host "===========================================================" -ForegroundColor Cyan

        foreach ($Repo in $CurrentBatch) {
            $TargetFolder = Join-Path $Global:StagingDir $Repo.ProjectName
            $ManifestPath = Get-IngestionManifestPath -ProjectName $Repo.ProjectName

            Write-Host "[*] Processing: $($Repo.ProjectName)" -ForegroundColor Cyan

            try {
                # 1. Slice raw code into AST JSON
                Write-Host "  -> Slicing ASTs..." -ForegroundColor DarkCyan
                Invoke-CheckedExternal -Executable "python" -Arguments @(
                    "$PSScriptRoot\semantic_slicer_AG.py",
                    $TargetFolder,
                    "--base-dir", $TargetFolder,
                    "--format", "json",
                    "--agent-role", "architecture",
                    "--workers", "8",
                    "--deterministic",
                    "-o", "$TargetFolder\raw_ast_bundle.json"
                ) -ErrorMessage "AST slicing failed for $($Repo.ProjectName)"

                # 2. Compile directly to final Unified YAML Knowledge Matrix (Chunk-Aware)
                $BundleParts = @(
                    Get-ChildItem -Path $TargetFolder -Filter "raw_ast_bundle*.json" |
                        Sort-Object Name
                )
                if ($BundleParts.Count -eq 0) {
                    throw "No bundle parts were produced for $($Repo.ProjectName)."
                }

                $CompiledChunkCount = 0
                foreach ($Part in $BundleParts) {
                    Write-Host "  -> Compiling Matrix for $($Part.Name)..." -ForegroundColor DarkCyan
                    # Extract the part suffix (e.g., "_part1" from "raw_ast_bundle_part1.json")
                    $Suffix = $Part.BaseName -replace "raw_ast_bundle", ""
                    # Create a distinct UNIFIED file for each chunk
                    $FinalYaml = Join-Path $Global:OutputDir "KNOWLEDGE_MATRIX_$($Repo.ProjectName)$Suffix`_UNIFIED.yaml"
                    # Execute the compiler
                    Invoke-CheckedExternal -Executable "python" -Arguments @(
                        "$PSScriptRoot\cognitive_processor.py",
                        $Part.FullName,
                        "--output", $FinalYaml,
                        "--agent-prefix", "LogosAgent"
                    ) -ErrorMessage "Knowledge compilation failed for $($Repo.ProjectName) ($($Part.Name))"
                    $CompiledChunkCount += 1
                }

                $ManifestPath = Write-IngestionManifest -ProjectName $Repo.ProjectName -RepoUrl $Repo.RepoUrl -ChunksProcessed $CompiledChunkCount
                Write-Host "  [OK] Wrote ingestion manifest to $ManifestPath" -ForegroundColor DarkGreen
                Write-Host "  [OK] Successfully compiled pure Unified YAML chunks for $($Repo.ProjectName)" -ForegroundColor Green
                $Repo.Status = "Completed"

                Write-Host "  [X] Purging raw repository data for $($Repo.ProjectName)..." -ForegroundColor DarkGray
                Remove-Item -Path $TargetFolder -Recurse -Force -ErrorAction SilentlyContinue
            } catch {
                Write-Host "  [ERROR] Failed to compile $($Repo.ProjectName): $($_.Exception.Message)" -ForegroundColor Red
                $Repo.Status = "Failed"
                if (Test-Path $TargetFolder) {
                    $RetainedWorkspace = Move-ToFailedWorkspace -ProjectName $Repo.ProjectName -SourcePath $TargetFolder
                    Write-Host "  [!] Retained failed workspace at $RetainedWorkspace" -ForegroundColor DarkYellow
                }
            }
        }

        $Records | Export-Csv -Path $Global:LogCsv -NoTypeInformation
        $i += $BatchAttemptCount
    }
    
    Write-Host "`n[OK] ALL QUEUED REPOSITORIES PROCESSED!" -ForegroundColor Green
}

if ($env:ALETHEIA_SKIP_INGESTOR_AUTORUN -ne "1") {
    Build-IngestionBacklog
    Start-AutonomousIngestion
}
