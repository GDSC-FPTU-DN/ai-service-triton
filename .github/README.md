<div style="width:100%;display:flex;flex-direction:column;justify-content:center;align-items:center">
<img style="width:80px"
src="https://seeklogo.com/images/G/google-developers-logo-F8BF3155AC-seeklogo.com.png" alt="logo"/>
</div>
<h3 style="text-align:center">Google Developer Student Clubs - FPT University Da Nang</h3>

# GDSC AI Services ~ Triton Server

This project is a children server of [GDSC AI Services](https://gdscfptu-ai-service-hf.hf.space). Using NVIDIA Triton Inference Server.

## ⚡ Use guides

### Build Triton model repository with `trsp`.

**Step 1**: Assume we have an `.onnx` model. Put this model to anywhere that you want in the working directory.

```
trsp/
models/ # Create this folder
model.onnx
config.yaml # Create this `.yaml` file
```

**Step 2**: Write a `config.yaml` for building model repository.

```yaml
model_repository: ./path/to/model/repository
models:
  [model-name]:
    engine: onnx # Use this with .onnx model
    max_batch_size: 4
    dynamic_batching: true
    max_queue_delay_microseconds: 100
    instance_group:
      - kind: gpu # or cpu
        count: 2
    versions:
      - version: 1
        path: ./path/to/model.onnx
```

**Step 3**: Build model repository.

```bash
python trsp/build.py -f /path/to/config.yaml
```

### Run Triton Inference Server with Docker.

You must have `Docker` in your computer.

```bash
docker run -it --gpus all --rm -p 8000:8000 -v ${pwd}/models:/models nvcr.io/nvidia/tritonserver:23.12-py3 tritonserver --model-repository=/models
```

Explain Docker command:

- `docker run`: Command for run Docker container.
- `-it`: Open terminal of container.
- `--gpus all`: Using GPUs for container.
- `--rm`: Remove container after exited.
- `-p 8000:8000`: Expose port of container at `8000`.
- `-v ${pwd}/models:/models`: Mount model repository to container files.
- `nvcr.io/nvidia/tritonserver:23.12-py3`: Triton server Image with version `23.12`.
- `tritonserver --model-repository=/models`: Start Triton server.

### Test Triton server.

We built an unit-test for testing functionality of triton server. Running test by the following command:

```bash
python test/run.py
```

## 😊 Contributors

- Đoàn Quang Minh - [Ming-doan](https://github.com/Ming-doan)
