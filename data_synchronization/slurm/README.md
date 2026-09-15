# SLURM scripts for LED sync pipeline

Two `sbatch` scripts to run the LED synchronization pipeline on the `kuma` cluster:

- `run_get_led_signal.sbatch` — runs `get_LED_signal.py`, extracting the LED signal from each video.
- `run_led_synchronizer.sbatch` — runs `LEDSynchronizer.py`, syncing the LED signals against ECoG data. Must be run **after** the LED signals exist, since it consumes their output.

Both request `h100` partition, `1 GPU` (unused, but required on this GPU-only cluster), `16 CPUs`, `50GB RAM`, `24h` walltime.

## Requirements

- **`mount_nas.sh` / `unmount_nas.sh`** must be on your `$PATH` — put them in `~/bin`. The jobs mount the NAS before running and unmount it afterward.
- **`ENCRYPTED_PASSWORD`** env var must be set in `~/.bashrc`. It's used by `mount_nas.sh` to authenticate the NAS mount. Create it with rclone:

  ```bash
  rclone obscure 'your_actual_password'
  ```

  Add the resulting string to `~/.bashrc`:

  ```bash
  export ENCRYPTED_PASSWORD="<output of rclone obscure>"
  ```

  Note this is obfuscation, not strong encryption — rclone can reverse it (`rclone reveal`). It just avoids storing/showing the plaintext password.

## Configuring

Before submitting, open each `.sbatch` file and check the variables under `# ---- User-configurable settings ----`: conda env name, repo path, and the input/output data folders. `OUTPUT_FOLDER` in `run_get_led_signal.sbatch` must match `LED_SIGNALS_FOLDER` in `run_led_synchronizer.sbatch`.

## Running

```bash
ssh kuma
cd /home/picek/repos/MultimodalBrainFeaturePipeline/data_synchronization/slurm

sbatch run_get_led_signal.sbatch
# wait for it to finish and populate OUTPUT_FOLDER, then:
sbatch run_led_synchronizer.sbatch
```

## Monitoring

```bash
squeue -u $USER                 # job status (PD=pending, R=running)
sinfo -p h100                   # partition/node availability if stuck pending
scancel <jobid>                 # cancel a job
```

Logs are written to `/scratch/picek/savedir/logs/sync/`, named `<timestamp>_<job-name>_<jobid>.out|.err` (built at runtime since SLURM's `--output`/`--error` don't support timestamp tokens). Tail the newest one live:

```bash
tail -f "$(ls -t /scratch/picek/savedir/logs/sync/*.out | head -1)"
```
