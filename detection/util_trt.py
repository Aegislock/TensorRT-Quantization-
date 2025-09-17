import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from calibrator import Calibrator

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="",
               fp16_mode=False, int8_mode=False,
               calibration_stream=None, calibration_table_path="",
               save_engine=False):
    """
    Builds or loads a TensorRT engine with automatic handling of static vs dynamic batch dimensions.
    Includes verbose debugging and FP16/INT8 support checks.
    """

    def build_engine(max_batch_size, save_engine):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, \
             builder.create_network(network_flags) as network, \
             trt.OnnxParser(network, TRT_LOGGER) as parser, \
             builder.create_builder_config() as config, \
             trt.Runtime(TRT_LOGGER) as runtime:

            # Parse ONNX
            if not os.path.exists(onnx_file_path):
                raise FileNotFoundError(f"ONNX file {onnx_file_path} not found")
            print(f"Loading ONNX file from {onnx_file_path}...")
            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    print("ONNX parse errors:")
                    for i in range(parser.num_errors):
                        print(parser.get_error(i))
                    raise RuntimeError("Failed to parse ONNX model.")
            print("ONNX parsing completed successfully.")

            # Workspace memory
            workspace_size = 2 << 30  # 2GB
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
            print(f"Workspace size set to {workspace_size} bytes")

            # FP16/INT8 flags
            if fp16_mode:
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    print("FP16 mode enabled")
                else:
                    print("Warning: FP16 not supported on this GPU, skipping")

            if int8_mode:
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    if calibration_stream is None:
                        raise RuntimeError("Calibration stream required for INT8 mode")
                    config.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)
                    print("INT8 mode enabled")
                else:
                    print("Warning: INT8 not supported on this GPU, skipping")

            # Optimization profile
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            input_name = input_tensor.name
            input_shape = tuple(input_tensor.shape)  # e.g., (-1,3,640,640) or (1,3,640,640)

            # Detect static batch
            if input_shape[0] != -1:
                # Static batch: force all profile shapes to match ONNX input
                min_shape = opt_shape = max_shape = input_shape
                print(f"Detected static batch size: {input_shape[0]}")
            else:
                # Dynamic batch: use max_batch_size
                min_shape = (1,) + input_shape[1:]
                opt_shape = (max_batch_size,) + input_shape[1:]
                max_shape = (max_batch_size,) + input_shape[1:]
                print(f"Detected dynamic batch. Using max_batch_size={max_batch_size}")

            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)

            # Debug prints
            print("Building engine with settings:")
            print(f"  Input name: {input_name}")
            print(f"  Input shape: {input_shape}")
            print(f"  Optimization profile: min={min_shape}, opt={opt_shape}, max={max_shape}")
            print(f"  FP16 requested: {fp16_mode}, INT8 requested: {int8_mode}")
            print(f"  FP16 supported: {builder.platform_has_fast_fp16}")
            print(f"  INT8 supported: {builder.platform_has_fast_int8}")

            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError(
                    "Failed to build engine. Possible causes:\n"
                    "- Unsupported ONNX ops (check verbose logs)\n"
                    "- FP16/INT8 not supported or incorrectly configured\n"
                    "- Optimization profile shape mismatch\n"
                    "- Workspace memory too small"
                )

            engine = runtime.deserialize_cuda_engine(serialized_engine)
            if engine is None:
                raise RuntimeError("Failed to deserialize engine from serialized network")

            print("Engine creation completed successfully.")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                print(f"Engine saved to {engine_file_path}")

            return engine

    # Load prebuilt engine if available
    if os.path.exists(engine_file_path):
        print(f"Loading serialized engine from {engine_file_path}")
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError("Failed to deserialize existing engine")
            print("Engine loaded successfully from file.")
            return engine
    else:
        return build_engine(max_batch_size, save_engine)
