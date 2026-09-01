<#
  Put the app one click away: a Desktop shortcut and a Start Menu entry, both
  pointing at run_app.bat.

      powershell -ExecutionPolicy Bypass -File scripts\install_shortcuts.ps1
      powershell -ExecutionPolicy Bypass -File scripts\install_shortcuts.ps1 -Remove

  Why a shortcut rather than the .bat itself: a shortcut can start minimised (so
  the console the server logs into is not in the way), can carry an icon, and can
  be pinned. Windows will not let a script pin anything to Start or the taskbar --
  that is deliberate on Microsoft's part -- so the last step is yours: right-click
  the Start Menu entry and choose "Pin to Start", or drag the Desktop one onto the
  taskbar.

  Nothing here writes into the repo. Both shortcuts are per-user, and -Remove takes
  them away again.
#>

[CmdletBinding()]
param(
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$target = Join-Path $repo 'run_app.bat'
if (-not (Test-Path $target)) { throw "run_app.bat not found beside the script: $target" }

$name = 'NFL Projections.lnk'
$desktop = Join-Path ([Environment]::GetFolderPath('Desktop')) $name
$startMenu = Join-Path ([Environment]::GetFolderPath('Programs')) $name

if ($Remove) {
    foreach ($path in @($desktop, $startMenu)) {
        if (Test-Path $path) { Remove-Item $path -Force; Write-Host "removed $path" }
        else { Write-Host "not there  $path" }
    }
    return
}

$shell = New-Object -ComObject WScript.Shell
foreach ($path in @($desktop, $startMenu)) {
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $target
    $link.WorkingDirectory = $repo
    $link.Description = 'NFL season projections - opens on http://localhost:8611'
    # 7 = minimised: the server's console is a log, not a thing to read
    $link.WindowStyle = 7
    # a chart-ish glyph out of the shell's own icon library, so the repo carries no binary
    $link.IconLocation = "$env:SystemRoot\System32\imageres.dll,174"
    $link.Save()
    Write-Host "created $path"
}

Write-Host ''
Write-Host 'One click starts the server and opens http://localhost:8611.'
Write-Host 'Clicking it again while it is running just opens the tab.'
Write-Host 'To pin it: right-click the Start Menu entry -> Pin to Start (Windows will not do this from a script).'
