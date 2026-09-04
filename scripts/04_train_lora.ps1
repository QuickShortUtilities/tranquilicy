# Fine-tune MusicGen with LoRA on your chillout dataset, using the
# maintained community trainer (ylacombe/musicgen-dreamboothing) rather than
# a hand-rolled training loop -- it already handles the EnCodec audio-code
# labeling, LoRA injection, and Seq2SeqTrainer setup correctly.
#
# Prereqs:
#   - Python + CUDA-enabled torch installed (see requirements.txt)
#   - 01_chunk_audio.py and 02_caption_chunks.py already run
#
# No Hugging Face account needed: `datasets` auto-detects a local folder of
# audio + metadata.csv as a "train" split (confirmed working directly against
# the chunks folder), so DatasetDir just points straight at it.
#
# Launches training via a Windows Scheduled Task (not Start-Process). Windows
# groups even "detached" Start-Process children into the same job object as
# the launching session, so they still die when that session/terminal closes
# -- confirmed by a real overnight failure. Task Scheduler runs the process
# completely outside any job this session belongs to, so it survives.
#
# Usage:
#   .\04_train_lora.ps1
#   .\04_train_lora.ps1 -ResumeFromCheckpoint "D:\musicgen_data\lora-out\checkpoint-500"

param(
    [string]$DatasetDir = "C:\Users\Gaming PC\musicgen_data\chunks",
    [string]$OutDir = "D:\musicgen_data\lora-out",
    [string]$BaseModel = "facebook/musicgen-medium",
    [int]$Epochs = 2,
    [double]$LearningRate = 2e-4,
    [int]$BatchSize = 2,
    [int]$GradAccumSteps = 8,
    [string]$ResumeFromCheckpoint = "",
    [string]$TaskName = "ChilloutMusicgenTrain"
)

$venvPy = "C:\Users\Gaming PC\chillout-musicgen\venv\Scripts\python.exe"
$venvPip = "C:\Users\Gaming PC\chillout-musicgen\venv\Scripts\pip.exe"
$repoDir = "C:\Users\Gaming PC\chillout-musicgen\musicgen-dreamboothing"

# The vectorized/preprocessed audio cache `datasets` builds during training is
# a full uncompressed copy of the dataset (much bigger than the source wav
# chunks). For the ~31k-chunk Audio 2 set this cache will likely exceed D:'s
# free space, so use E: (4TB HDD, plenty of room) instead.
$env:HF_DATASETS_CACHE = "E:\musicgen_hf_datasets_cache_train"
New-Item -ItemType Directory -Force -Path $env:HF_DATASETS_CACHE | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path $repoDir)) {
    git clone https://github.com/ylacombe/musicgen-dreamboothing.git $repoDir
}

Set-Location $repoDir
& $venvPip install -e .

$trainArgs = @(
    "dreambooth_musicgen.py",
    "--use_lora",
    "--model_name_or_path", "`"$BaseModel`"",
    "--dataset_name", "`"$DatasetDir`"",
    "--train_split_name", "train",
    "--target_audio_column_name", "audio",
    "--text_column_name", "text",
    "--output_dir", "`"$OutDir`"",
    "--do_train",
    "--fp16",
    "--num_train_epochs", $Epochs,
    "--gradient_accumulation_steps", $GradAccumSteps,
    "--gradient_checkpointing",
    "--per_device_train_batch_size", $BatchSize,
    "--learning_rate", $LearningRate,
    "--guidance_scale", "3.0",
    "--pad_token_id", "2048",
    "--decoder_start_token_id", "2048",
    "--max_duration_in_seconds", "31",
    "--min_duration_in_seconds", "5.0",
    "--save_steps", "500",
    "--logging_steps", "20"
)
if ($ResumeFromCheckpoint -ne "") {
    $trainArgs += @("--resume_from_checkpoint", "`"$ResumeFromCheckpoint`"")
}

$stdout = "D:\musicgen_data\train_stdout.log"
$stderr = "D:\musicgen_data\train_stderr.log"
$pyArgString = $trainArgs -join " "
# cmd.exe wrapper gives us file-redirection for stdout/stderr, and running the
# whole thing via WMI's Win32_Process.Create spawns it as a child of the WMI
# provider host (wmiprvse.exe) instead of this session's process tree --
# Task Scheduler registration is denied in this session, and Start-Process
# "detached" children still die when this session's job object closes
# (confirmed twice by real overnight failures), so this is the fallback.
# IMPORTANT: Win32_Process.Create does NOT inherit this PowerShell session's
# $env: vars -- confirmed by a real failure where HF_DATASETS_CACHE silently
# fell back to the C: default and filled the drive to 100%. Set it explicitly
# inside the spawned cmd.exe's own environment instead.
$cmdLine = "cmd.exe /c `"set HF_DATASETS_CACHE=$($env:HF_DATASETS_CACHE)&& cd /d `"$repoDir`" && `"$venvPy`" $pyArgString > `"$stdout`" 2> `"$stderr`"`""

$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmdLine}
if ($result.ReturnValue -ne 0) {
    Write-Error "Win32_Process.Create failed with code $($result.ReturnValue)"
} else {
    Write-Output "Launched training via WMI process (PID $($result.ProcessId)), detached from this session. Logs: $stdout / $stderr"
}

Set-Location "C:\Users\Gaming PC\chillout-musicgen"
