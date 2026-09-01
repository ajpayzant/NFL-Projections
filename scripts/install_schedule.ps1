<#
  Keep the data current without anybody remembering to. Registers two Windows scheduled tasks that
  call scripts\weekly_update.bat.

      powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1
      powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -DailyAt 07:15 -WeeklyDay Tuesday
      powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -List
      powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Remove

  Two tasks because the data has two speeds:

    daily   light pass -- rosters, depth charts, schedules, injuries. Seconds. This is the one that
            matters in season: it is what stops a Sunday's projection being built on Wednesday's
            starters.
    weekly  full pass -- rebuilds the played-game tables from the engine repo first, then the light
            pass. Minutes. Wednesday early morning by default, which is after Monday night's game has
            been processed upstream and before anybody wants Thursday's numbers.

  Task Scheduler will show "Last Run Result: 1" for the daily task until the season starts, and that is
  the honest answer rather than a fault: 1 means a non-essential table was missed, and nflreadpy refuses
  the projection season's injuries outright until week 1. Anything that would actually cost you a
  projection -- rosters, depth charts, schedules -- comes back as 2.

  Both are registered under your own account, so no elevation is needed and nothing runs as SYSTEM.
  StartWhenAvailable is on: a machine that was asleep at 06:30 runs the task when it wakes rather than
  skipping the day. A console window may flash when a task fires -- that is the batch file, and every
  line it produced is in build\review\update.log.

  These tasks never touch a scenario, an override or anything under data\scenarios. They replace the
  tables the projection reads; your edits are keyed by player and re-apply to the new numbers on the
  next run. What they cannot do is reach into an already-running server: st.cache_data is in memory,
  so the app's sidebar reads build\review\last_update.json and says when a restart is owed.
#>

[CmdletBinding()]
param(
    [string]$DailyAt = '06:30',
    [string]$WeeklyAt = '04:00',
    [ValidateSet('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')]
    [string]$WeeklyDay = 'Wednesday',
    [switch]$List,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$bat = Join-Path $repo 'scripts\weekly_update.bat'
if (-not (Test-Path $bat)) { throw "weekly_update.bat not found: $bat" }

$daily = 'NFL Projections - daily data refresh'
$weekly = 'NFL Projections - weekly rebuild'

function Show-Tasks {
    foreach ($name in @($daily, $weekly)) {
        $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
        if (-not $task) { Write-Host "not registered   $name"; continue }
        $info = Get-ScheduledTaskInfo -TaskName $name
        Write-Host ("{0,-38} {1,-9} last {2} -> {3}  next {4}" -f `
            $name, $task.State, $info.LastRunTime, $info.LastTaskResult, $info.NextRunTime)
    }
}

if ($List) { Show-Tasks; return }

if ($Remove) {
    foreach ($name in @($daily, $weekly)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "removed $name"
        } else { Write-Host "not registered   $name" }
    }
    return
}

# StartWhenAvailable covers the asleep-at-06:30 case; IgnoreNew means a long weekly rebuild is never
# joined by a second copy of itself; the 4h limit stops a hung network fetch holding the slot forever.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4) `
    -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

function Register-Update {
    param([string]$Name, [string]$Description, $Trigger, [string]$Flags)

    $argument = if ($Flags) { "/c `"`"$bat`" $Flags`"" } else { "/c `"`"$bat`"`"" }
    $action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $argument -WorkingDirectory $repo
    Register-ScheduledTask -TaskName $Name -Description $Description -Action $action `
        -Trigger $Trigger -Settings $settings -Force | Out-Null
    Write-Host "registered $Name"
}

Register-Update -Name $daily -Description 'Rosters, depth charts, schedules and injuries for the projection season.' `
    -Trigger (New-ScheduledTaskTrigger -Daily -At $DailyAt) -Flags ''

Register-Update -Name $weekly -Description 'Rebuild the played-game lake, then refresh the light tables.' `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $WeeklyDay -At $WeeklyAt) -Flags '--full'

Write-Host ''
Show-Tasks
Write-Host ''
Write-Host "log: $(Join-Path $repo 'build\review\update.log')"
Write-Host ('Run one now without waiting:  Start-ScheduledTask -TaskName "{0}"' -f $daily)
Write-Host 'Undo:  powershell -ExecutionPolicy Bypass -File scripts\install_schedule.ps1 -Remove'
