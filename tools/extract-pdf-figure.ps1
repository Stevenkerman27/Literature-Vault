[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$InputPdf,

    [Parameter(Mandatory)]
    [ValidateRange(1, [int]::MaxValue)]
    [int[]]$Page,

    [Parameter(Mandatory)]
    [string]$OutputDirectory,

    [ValidateRange(72, 1200)]
    [int]$Dpi = 300,

    [int[]]$Crop,

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9._-]*$')]
    [string]$NamePrefix = 'figure',

    [ValidateSet('Png', 'Jpeg')]
    [string]$ImageFormat = 'Png',

    [string]$RendererPath,

    [switch]$KeepRenderedPage
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Crop -and ($Crop.Count -ne 4 -or $Crop[0] -lt 0 -or $Crop[1] -lt 0 -or $Crop[2] -le 0 -or $Crop[3] -le 0)) {
    throw 'Crop must contain exactly four values: x, y, width, height. x and y must be non-negative; width and height must be positive.'
}

function Get-PdfRenderer {
    param([string]$ExplicitPath)

    if ($ExplicitPath) {
        if (-not (Test-Path -LiteralPath $ExplicitPath -PathType Leaf)) {
            throw "RendererPath does not exist: $ExplicitPath"
        }
        return (Resolve-Path -LiteralPath $ExplicitPath).Path
    }

    $renderer = Get-Command pdftoppm -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $renderer) {
        throw 'pdftoppm was not found. Install Poppler or pass -RendererPath with the full path to pdftoppm.exe.'
    }
    return $renderer.Source
}

function Save-Image {
    param(
        [string]$SourcePath,
        [string]$DestinationPath,
        [int[]]$CropRectangle,
        [string]$Format
    )

    Add-Type -AssemblyName System.Drawing
    $source = [System.Drawing.Image]::FromFile($SourcePath)
    $bitmap = $null
    $graphics = $null
    try {
        if ($CropRectangle) {
            $rectangle = [System.Drawing.Rectangle]::new(
                $CropRectangle[0], $CropRectangle[1], $CropRectangle[2], $CropRectangle[3]
            )
            $bounds = [System.Drawing.Rectangle]::new(0, 0, $source.Width, $source.Height)
            if (-not $bounds.Contains($rectangle)) {
                throw "Crop rectangle ($($CropRectangle -join ', ')) exceeds the rendered page size ($($source.Width) x $($source.Height))."
            }
        }
        else {
            $rectangle = [System.Drawing.Rectangle]::new(0, 0, $source.Width, $source.Height)
        }

        $bitmap = [System.Drawing.Bitmap]::new($rectangle.Width, $rectangle.Height)
        $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
        $graphics.DrawImage($source, [System.Drawing.Rectangle]::new(0, 0, $bitmap.Width, $bitmap.Height), $rectangle, [System.Drawing.GraphicsUnit]::Pixel)
        $encoder = if ($Format -eq 'Jpeg') { [System.Drawing.Imaging.ImageFormat]::Jpeg } else { [System.Drawing.Imaging.ImageFormat]::Png }
        $bitmap.Save($DestinationPath, $encoder)
    }
    finally {
        if ($graphics) { $graphics.Dispose() }
        if ($bitmap) { $bitmap.Dispose() }
        $source.Dispose()
    }
}

$renderer = Get-PdfRenderer -ExplicitPath $RendererPath
$outputPath = New-Item -ItemType Directory -Force -Path $OutputDirectory
$extension = if ($ImageFormat -eq 'Jpeg') { 'jpg' } else { 'png' }
$tempDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("pdf-figure-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempDirectory | Out-Null

try {
    foreach ($pageNumber in ($Page | Sort-Object -Unique)) {
        $renderPrefix = Join-Path $tempDirectory "page-$pageNumber"
        & $renderer -f $pageNumber -l $pageNumber -r $Dpi -png $InputPdf $renderPrefix
        if ($LASTEXITCODE -ne 0) {
            throw "pdftoppm failed while rendering page $pageNumber (exit code $LASTEXITCODE)."
        }

        $renderedPage = Get-ChildItem -LiteralPath $tempDirectory -Filter "page-$pageNumber-*.png" |
            Select-Object -First 1
        if (-not $renderedPage) {
            throw "pdftoppm did not create an image for page $pageNumber."
        }

        $baseName = '{0}-p{1:d3}' -f $NamePrefix, $pageNumber
        $figurePath = Join-Path $outputPath "$baseName.$extension"
        Save-Image -SourcePath $renderedPage.FullName -DestinationPath $figurePath -CropRectangle $Crop -Format $ImageFormat

        if ($KeepRenderedPage) {
            Copy-Item -LiteralPath $renderedPage.FullName -Destination (Join-Path $outputPath "$baseName-page.png") -Force
        }
        Write-Output $figurePath
    }
}
finally {
    Remove-Item -LiteralPath $tempDirectory -Recurse -Force -ErrorAction SilentlyContinue
}
