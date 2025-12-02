#!/usr/bin/env python3
"""
Convert Keras model to TensorFlow Lite format for mobile deployment.

Usage:
    python convert_to_tflite.py
    python convert_to_tflite.py --model model.keras --output squat_scorer.tflite
"""

import argparse
from pathlib import Path

import tensorflow as tf
import numpy as np


def convert_keras_to_tflite(
    keras_model_path: str,
    output_path: str,
    optimize: bool = True,
    quantize: bool = False
):
    """
    Convert a Keras model to TFLite format.
    
    Args:
        keras_model_path: Path to .keras model file
        output_path: Path to save .tflite file
        optimize: Whether to apply default optimizations
        quantize: Whether to apply quantization (INT8)
    """
    print(f"Loading Keras model from: {keras_model_path}")
    try:
        model = tf.keras.models.load_model(keras_model_path)
        print("Model loaded successfully!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Create a representative dataset for quantization if needed
    # This is a dummy dataset - in practice, use real data
    def representative_dataset():
        # Generate sample input matching model input shape
        # Shape: (batch, NUM_FRAMES, FEATURE_DIMS) = (1, 16, 133)
        for _ in range(10):
            yield [np.random.randn(1, 16, 133).astype(np.float32)]
    
    print("\nConverting to TFLite...")
    print("Step 1: Exporting to SavedModel format...")
    
    # Save as SavedModel first (works with both Keras 2 and 3)
    saved_model_dir = Path("checkpoints") / "squat_scorer_savedmodel"
    if saved_model_dir.exists():
        import shutil
        shutil.rmtree(saved_model_dir)
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try Keras 3 export method first
        try:
            model.export(str(saved_model_dir))
            print("✓ SavedModel created successfully (Keras 3 export)!")
        except (AttributeError, TypeError) as e1:
            print(f"Keras 3 export not available: {e1}")
            # Try alternative: create a concrete function and save
            try:
                @tf.function(input_signature=[tf.TensorSpec(shape=[None, 16, 133], dtype=tf.float32)])
                def model_func(inputs):
                    return model(inputs)
                
                # Save using the concrete function
                tf.saved_model.save(
                    model,
                    str(saved_model_dir),
                    signatures={'serving_default': model_func}
                )
                print("✓ SavedModel created successfully (with concrete function)!")
            except Exception as e2:
                print(f"Concrete function method failed: {e2}")
                # Last resort: try TensorFlow's SavedModel API
                try:
                    tf.saved_model.save(model, str(saved_model_dir))
                    print("✓ SavedModel created successfully (TF SavedModel)!")
                except Exception as e3:
                    print(f"✗ All SavedModel methods failed!")
                    print(f"  Keras 3 export: {e1}")
                    print(f"  Concrete function: {e2}")
                    print(f"  TF SavedModel: {e3}")
                    import traceback
                    traceback.print_exc()
                    return False
    except Exception as e:
        print(f"✗ Unexpected error creating SavedModel: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 2: Converting SavedModel to TFLite...")
    
    try:
        # Convert from SavedModel
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        
        # Apply optimizations
        if optimize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            print("Applied default optimizations")
        
        # Apply quantization if requested
        if quantize:
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            print("Applied INT8 quantization")
        else:
            # For LSTM/BiLSTM models, we may need SELECT_TF_OPS
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            print("Using TFLITE_BUILTINS + SELECT_TF_OPS")
        
        # Key flags for TensorList + LSTM compatibility
        converter.experimental_enable_resource_variables = True
        # Note: _experimental_lower_tensor_list_ops is deprecated in newer TF versions
        try:
            converter._experimental_lower_tensor_list_ops = False
        except AttributeError:
            # Attribute doesn't exist in newer TF versions, skip
            pass
            
            # Convert from SavedModel
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            
            if optimize:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if quantize:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            else:
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS,
                ]
            
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        output_path_obj.write_bytes(tflite_model)
        
        file_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"\n✓ Successfully converted to TFLite!")
        print(f"  Output: {output_path}")
        print(f"  Size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TFLite conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tflite_model(tflite_path: str):
    """Test the converted TFLite model with a dummy input."""
    print(f"\nTesting TFLite model: {tflite_path}")
    
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Input details: {input_details[0]}")
        print(f"Output details: {output_details[0]}")
        
        # Create dummy input
        input_shape = input_details[0]['shape']
        input_data = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n✓ TFLite model test successful!")
        print(f"  Input shape: {input_shape}")
        print(f"  Output shape: {output_data.shape}")
        print(f"  Sample output: {output_data[0][0]:.4f}")
        print(f"  (Remember: output is normalized 0-1, multiply by 100 for score)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TFLite model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite")
    parser.add_argument("--model", type=str, default="model.keras",
                       help="Path to input Keras model (default: model.keras)")
    parser.add_argument("--output", type=str, default="squat_scorer.tflite",
                       help="Path to output TFLite file (default: squat_scorer.tflite)")
    parser.add_argument("--no-optimize", action="store_true",
                       help="Disable optimizations")
    parser.add_argument("--quantize", action="store_true",
                       help="Apply INT8 quantization (smaller file, may reduce accuracy)")
    parser.add_argument("--test", action="store_true",
                       help="Test the converted TFLite model")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    # Convert
    success = convert_keras_to_tflite(
        str(model_path),
        args.output,
        optimize=not args.no_optimize,
        quantize=args.quantize
    )
    
    if not success:
        return 1
    
    # Test if requested
    if args.test:
        test_tflite_model(args.output)
    
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print("\nNote: For Android deployment, you may need to:")
    print("  1. Include SELECT_TF_OPS in your app's build.gradle")
    print("  2. Test the model with real input data")
    print("  3. Consider quantization for smaller file size")
    print("\nFor more info, see: https://www.tensorflow.org/lite/android/ops_select")
    
    return 0


if __name__ == "__main__":
    exit(main())

