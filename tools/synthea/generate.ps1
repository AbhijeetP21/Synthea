<#
.SYNOPSIS
  Generate a small synthetic FHIR cohort with Synthea, using Docker for the
  Java runtime (no local JDK required), then curate one patient bundle as the
  committed MVP sample.

.NOTES
  Synthetic data only — no real PHI ever touches this pipeline.
#>

$ErrorActionPreference = "Stop"

$root      = (Resolve-Path "$PSScriptRoot\..\..").Path
$jarDir    = Join-Path $root "tools\synthea"
$jarPath   = Join-Path $jarDir "synthea-with-dependencies.jar"
$outDir    = Join-Path $root "data\synthea"
$fhirDir   = Join-Path $outDir "fhir"
$sampleDir = Join-Path $root "data\sample"

# Tunables
$Patients = if ($env:SYNTHEA_PATIENTS) { $env:SYNTHEA_PATIENTS } else { 10 }
$Seed     = if ($env:SYNTHEA_SEED) { $env:SYNTHEA_SEED } else { 20240601 }
$State    = if ($env:SYNTHEA_STATE) { $env:SYNTHEA_STATE } else { "Massachusetts" }

New-Item -ItemType Directory -Force -Path $jarDir, $outDir, $sampleDir | Out-Null

if (-not (Test-Path $jarPath)) {
    Write-Host "Downloading Synthea (synthea-with-dependencies.jar)..."
    $url = "https://github.com/synthetichealth/synthea/releases/latest/download/synthea-with-dependencies.jar"
    Invoke-WebRequest -Uri $url -OutFile $jarPath
}

Write-Host "Generating $Patients patient(s) (seed=$Seed, state=$State) via Docker..."
docker run --rm `
    -v "${jarDir}:/synthea" `
    -v "${outDir}:/out" `
    eclipse-temurin:17-jre `
    java -jar /synthea/synthea-with-dependencies.jar `
    -p $Patients -s $Seed -a 30-75 `
    --exporter.baseDirectory /out `
    --exporter.fhir.export true `
    --exporter.hospital.fhir.export false `
    --exporter.practitioner.fhir.export false `
    $State

if ($LASTEXITCODE -ne 0) { throw "Synthea generation failed (exit $LASTEXITCODE)." }

# Curate: pick the richest patient bundle (largest file = most resources/history)
# and copy it to the committed sample location for a reproducible MVP.
$bundles = Get-ChildItem -Path $fhirDir -Filter "*.json" |
    Where-Object { $_.Name -notmatch "^(hospitalInformation|practitionerInformation)" } |
    Sort-Object Length -Descending

if (-not $bundles) { throw "No patient bundles produced in $fhirDir." }

$chosen = $bundles[0]
$dest   = Join-Path $sampleDir "patient_bundle.json"
Copy-Item -Path $chosen.FullName -Destination $dest -Force

Write-Host ""
Write-Host "Curated sample patient:" -ForegroundColor Green
Write-Host "  source: $($chosen.Name) ($([math]::Round($chosen.Length/1KB)) KB)"
Write-Host "  -> $dest"
Write-Host ""
Write-Host "Ingest it with:  mise run ingest data/sample/patient_bundle.json"
