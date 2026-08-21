# io_uring on OpenShift / Kubernetes

This repository contains the configuration and tooling needed to enable `io_uring` for workloads running on OpenShift, with a focus on GPU-accelerated LLM inference services (vLLM + InstantTensor).

## Background

By default, OpenShift's `runtime/default` seccomp profile blocks the three `io_uring` syscalls (`io_uring_setup`, `io_uring_register`, `io_uring_enter`). The InstantTensor backend for vLLM uses `io_uring` for fast model loading, so enabling it requires three node-level changes on GPU nodes and a custom seccomp profile:

1. **`kernel.io_uring_disabled = 0`** — RHEL 9 CoreOS ships with io_uring disabled by default.
   2. **Unlimited memlock ulimit** — InstantTensor registers the host I/O buffer (several GiB) as a fixed io_uring buffer via `io_uring_register_buffers`, which locks those pages in RAM. The default container memlock limit (8 MiB) is far too low. Because the workload runs as a non-root UID with `allowPrivilegeEscalation: false`, `CAP_IPC_LOCK` cannot be used to bypass the limit (the capability ends up in `CapBnd` but never reaches `CapPrm`/`CapEff` for non-root processes). This repo uses a quick fix to raise the limit at the CRI-O level via `default_ulimits`. Further work might be worth to reduce the surface of this change.
3. **Custom seccomp profile** — adds `io_uring_setup`, `io_uring_register`, and `io_uring_enter` to the allow-list.

## Repository structure

```
openshift-configuration/   # Apply these manifests to the cluster (in order)
  src/
    butane.yaml            # Butane source for the MachineConfig (human-readable)
    io-uring.json          # Seccomp profile (runtime/default + io_uring syscalls)
  01-machine-config-pool.yaml   # MachineConfigPool targeting GPU nodes
  02-machine-config.yaml        # Sets kernel.io_uring_disabled=0 and deploys the seccomp profile
  03-scc-openshift-ai-inferenceservice-image-volume-io-uring.yaml  # SCC allowing the custom seccomp profile
  04-clusterrolebinding.yaml    # Binds the SCC to the default SA in experiment-02

io_uring_smoke-test/       # Minimal C test that exercises setup/register/enter
  main.c
  Dockerfile               # Ubuntu 24.04 image; built binary: /usr/local/bin/uring-test

vllm/                      # vLLM + InstantTensor image with io_uring smoke test baked in
  main.c                   # Same smoke test
  Dockerfile               # Built on top of RHAIOI vLLM CUDA RHEL9 image

test/                      # Kubernetes manifests for testing
  deployment.yaml                # Runs the smoke-test container with the io-uring seccomp profile
  llmisvc-instanttensors.yaml    # LLMInferenceService using InstantTensor + URING backend
  llmisvc-safetensors.yaml       # LLMInferenceService using standard safetensors (baseline)
  pvc.yaml                       # PersistentVolumeClaim backed by the Local Storage Operator
  task-download-model.yaml       # Tekton Task: download a model from HuggingFace to a PVC
  taskrun-download-model.yaml    # Tekton TaskRun: trigger the download task
```

## Setup

### 1. Enable io_uring on GPU nodes

The `MachineConfig` in `openshift-configuration/02-machine-config.yaml` does three things:

- Writes `/etc/sysctl.d/99-io-uring.conf` with `kernel.io_uring_disabled = 0` so the kernel permits `io_uring`.
- Writes `/etc/crio/crio.conf.d/99-memlock.conf` with `default_ulimits = ["memlock=-1:-1"]` so all containers on the node have an unlimited memlock limit, allowing `io_uring_register_buffers` to lock large I/O buffers.
- Deploys `io-uring.json` to `/var/lib/kubelet/seccomp/profiles/` — a copy of the `runtime/default` seccomp profile with `io_uring_setup`, `io_uring_register`, and `io_uring_enter` added to the allow-list.

The `MachineConfigPool` targets nodes labelled `nvidia.com/gpu.present=true`.

Apply in order:

```bash
oc apply -f openshift-configuration/01-machine-config-pool.yaml
oc apply -f openshift-configuration/02-machine-config.yaml
oc apply -f openshift-configuration/03-scc-openshift-ai-inferenceservice-image-volume-io-uring.yaml
oc apply -f openshift-configuration/04-clusterrolebinding.yaml
```

Wait for the `gpu` MachineConfigPool to finish rolling out before proceeding.

### 2. Regenerating the MachineConfig from source

The `MachineConfig` YAML is generated from the Butane source:

```bash
butane openshift-configuration/src/butane.yaml -o openshift-configuration/02-machine-config.yaml
```

### 3. Smoke-test the io_uring setup

Build and push the smoke-test image:

```bash
podman build -t <registry>/uring-test:latest io_uring_smoke-test/
podman push <registry>/uring-test:latest
```

Update the image reference in `test/deployment.yaml`, then:

```bash
oc apply -f test/deployment.yaml
oc logs -n experiment-02 deployment/test-io-uring
```

Expected output:

```
io_uring_setup: OK
io_uring_register (PROBE): OK
io_uring_register (FILES): OK
io_uring_register (BUFFERS): OK
io_uring_enter: OK
io_uring NOP completion: OK
io_uring Kubernetes test: PASS
```

The smoke test covers all checks that InstantTensor's `uring_available()` performs internally, including the probe opcode check and fixed buffer registration. Run it in the **main workload container** (not just the init container) to verify the memlock limit is in effect — the BUFFERS check specifically exercises `io_uring_register_buffers` and will fail with `Cannot allocate memory` if the memlock ulimit is still too low.

### 4. Provision model storage (Local Storage Operator)

Model weights are stored on a local NVMe/SSD volume managed by the **Local Storage Operator (LSO)**. The PVC in `test/pvc.yaml` references the `local-sc-gpu` StorageClass created by LSO and requests 80 Gi in `ReadWriteOnce` / `Filesystem` mode:

```bash
oc apply -f test/pvc.yaml
```

> **Prerequisites:** The Local Storage Operator must be installed and a `LocalVolume` (or `LocalVolumeSet`) must already expose the target disk on the GPU node under the `local-sc-gpu` StorageClass. The PVC will remain `Pending` until a matching `PersistentVolume` is available on that node.

The PVC name `model-storage-lso` is referenced by the `LLMInferenceService` manifests via `uri: pvc://model-storage-lso`.

### 6. Deploy vLLM with the InstantTensor URING backend

Download the model first (requires a HuggingFace token in the `llm-d-hf-token` secret):

```bash
oc apply -f test/taskrun-download-model.yaml
```

Deploy the inference service with InstantTensor and `io_uring`:

```bash
oc apply -f test/llmisvc-instanttensors.yaml
```

The key environment variables that activate the backend:

| Variable | Value |
|---|---|
| `INSTANTTENSOR_BACKEND` | `URING` |
| `INSTANTTENSOR_DEBUG` | `1` (optional, verbose logging) |

The pod's `securityContext` must reference the custom seccomp profile:

```yaml
securityContext:
  seccompProfile:
    type: Localhost
    localhostProfile: profiles/io-uring.json
```

## Notes

- The seccomp profile (`io-uring.json`) is the standard `runtime/default` profile from CRI-O with `io_uring_setup`, `io_uring_register`, and `io_uring_enter` added to the allow-list. All other syscall rules are unchanged.
- The `SCC` (`03-scc-openshift-ai-inferenceservice-image-volume-io-uring.yaml`) permits the `localhost/profiles/io-uring.json` seccomp profile while keeping privilege escalation and host access disabled.
- User namespaces (`hostUsers: false`) are used in the smoke-test deployment; the SCC allows `AllowHostLevel` for `userNamespaceLevel` to support this.
- **Why `CAP_IPC_LOCK` is not used for memlock:** adding `IPC_LOCK` to the container's `capabilities.add` puts it in `CapBnd` (the bounding set) but not in `CapPrm`/`CapEff`. For non-root containers with `allowPrivilegeEscalation: false`, Linux does not propagate bounding-set capabilities into the permitted/effective sets — file capabilities are blocked by `no_new_privs`, and ambient capabilities require `CapPrm` to already contain the capability (a Catch-22). The CRI-O `default_ulimits` approach sidesteps this entirely.
- **RHEL 9 / OCP kernel version:** RHEL 9 CoreOS ships kernel `5.14.x`. InstantTensor v0.1.9 includes a guard (`kernel_at_least_5_15()`) that returns false for kernels below 5.15, which would prevent the URING backend from being selected. Ensure the installed InstantTensor version does not have this check, or that it has been updated to account for RHEL 9's extensively backported 5.14 kernel.
