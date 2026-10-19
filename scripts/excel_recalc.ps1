# Open a workbook in Excel, calculate every formula and save it with the results, so
# `python -m marketing_roi.workbook --check` can compare them with the analysis views.
# Windows with Excel installed only:
#   powershell -ExecutionPolicy Bypass -File scripts/excel_recalc.ps1 model.xlsx
param([Parameter(Mandatory = $true)][string]$Path)

$ErrorActionPreference = "Stop"
$full = (Resolve-Path $Path).Path
$excel = New-Object -ComObject Excel.Application
try {
    $excel.Visible = $false
    $excel.DisplayAlerts = $false
    $started = Get-Date
    $book = $excel.Workbooks.Open($full)
    $excel.CalculateFull()
    $book.Save()
    $book.Close($false)
    $seconds = ((Get-Date) - $started).TotalSeconds
    Write-Output ("calculated and saved {0} in {1:N1} s (Excel {2})" -f $full, $seconds, $excel.Version)
}
finally {
    $excel.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel)
}
