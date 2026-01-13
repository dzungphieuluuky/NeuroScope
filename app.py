import streamlit as st
import torch
import torch.nn as nn
import torch.onnx
import os
import tempfile
import json
from io import BytesIO
import pandas as pd
from safetensors.torch import load_file
from streamlit_agraph import agraph, Node, Edge, Config
from torchview import draw_graph
import gc

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NeuroScope Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN CSS STYLING ---
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .code-block {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 20px;
        border-radius: 10px;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.6;
    }
    
    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: #f8f9ff;
        transition: all 0.3s ease;
    }
    
    .upload-zone:hover {
        border-color: #764ba2;
        background: #f0f2ff;
    }
    
    .stat-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .progress-text {
        font-weight: 500;
        color: #667eea;
    }
    
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 15px;
        border-radius: 8px;
        color: #15803d;
    }
    
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        color: #991b1b;
    }
    
    .tab-content {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
if 'uploaded_model' not in st.session_state:
    st.session_state.uploaded_model = None
if 'model_info' not in st.session_state:
    st.session_state.model_info = None
if 'onnx_bytes' not in st.session_state:
    st.session_state.onnx_bytes = None

# --- UTILITY FUNCTIONS ---

def load_model_file(uploaded_file):
    """Load model from .pt, .pth, or .safetensors file"""
    try:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_ext in ['.pt', '.pth']:
            model = torch.load(temp_path, map_location='cpu', weights_only=False)
        elif file_ext == '.safetensors':
            state_dict = load_file(temp_path)
            # Create a simple container for state dict
            model = type('Model', (), {})()
            for key, value in state_dict.items():
                setattr(model, key.split('.')[0], value)
        else:
            return None, "Unsupported file format"
        
        os.remove(temp_path)
        return model, None
    except Exception as e:
        return None, str(e)

def get_model_architecture(model):
    """Extract model architecture information"""
    try:
        info = {
            'type': type(model).__name__,
            'module': type(model).__module__,
            'parameters': sum(p.numel() for p in model.parameters()) if hasattr(model, 'parameters') else 0,
            'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad) if hasattr(model, 'parameters') else 0,
            'layers': len(list(model.modules())) if hasattr(model, 'modules') else 0,
        }
        return info
    except Exception as e:
        return {'error': str(e)}

def infer_input_shape(model, default_shape="1, 3, 224, 224"):
    """Try to infer input shape from model"""
    try:
        first_layer = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                first_layer = module
                break
        
        if first_layer:
            in_channels = first_layer.in_channels
            return f"1, {in_channels}, 224, 224"
    except:
        pass
    
    return default_shape

def trace_model(model, input_shape):
    """Trace model and generate computational graph"""
    try:
        shape = [int(x.strip()) for x in input_shape.split(',')]
        if len(shape) < 1 or any(s <= 0 for s in shape):
            return None, "Invalid shape dimensions"
        
        dummy_input = torch.randn(*shape)
        
        # Try to get output shape
        with torch.no_grad():
            if isinstance(model, nn.Module):
                output = model(dummy_input)
            else:
                output = model
        
        graph = draw_graph(model, input_size=tuple(shape), expand_nested=True)
        return graph.visual_graph, None
    except Exception as e:
        return None, str(e)

def export_to_onnx(model, input_shape, opset_version=18, dynamic_batch=False):
    """Export model to ONNX format"""
    try:
        shape = [int(x.strip()) for x in input_shape.split(',')]
        dummy_input = torch.randn(*shape)
        
        onnx_buffer = BytesIO()
        
        export_kwargs = {
            'opset_version': int(max(opset_version, 18)),
            'do_constant_folding': True,
            'input_names': ['input'],
            'output_names': ['output'],
        }
        
        if dynamic_batch:
            export_kwargs['dynamic_axes'] = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (dummy_input,),
                onnx_buffer,
                **export_kwargs
            )
        
        onnx_buffer.seek(0)
        return onnx_buffer.getvalue(), None
    except Exception as e:
        return None, str(e)

# --- MAIN UI ---

# Header
with st.container():
    st.markdown("""
    <div class="header-card">
        <h1 style="margin: 0; font-size: 2.5em;">NeuroScope Pro</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.95;">
            Convert & Visualize PyTorch Models in One Click
        </p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Upload Model")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "Select a model file",
        type=['pt', 'pth', 'safetensors'],
        help="Upload a PyTorch model file (.pt, .pth) or SafeTensors model (.safetensors)"
    )

# Main content
if uploaded_file is not None:
    # Load model
    if st.session_state.uploaded_model is None or st.session_state.uploaded_model.name != uploaded_file.name:
        with st.spinner("Loading model..."):
            model, error = load_model_file(uploaded_file)
            if error:
                st.error(f"Failed to load model: {error}")
                st.stop()
            
            st.session_state.uploaded_model = uploaded_file
            st.session_state.model = model
            st.session_state.model_info = get_model_architecture(model)
    
    model = st.session_state.model
    model_info = st.session_state.model_info
    
    # Model Info Section
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Parameters",
                f"{model_info.get('parameters', 0):,}",
                help="Total number of model parameters"
            )
        
        with col2:
            st.metric(
                "Trainable Params",
                f"{model_info.get('trainable', 0):,}",
                help="Number of trainable parameters"
            )
        
        with col3:
            st.metric(
                "Model Size",
                f"{model_info.get('parameters', 0) * 4 / (1024 * 1024):.2f} MB",
                help="Estimated model size in float32"
            )
        
        with col4:
            st.metric(
                "Layers",
                f"{model_info.get('layers', 0)}",
                help="Total number of layers"
            )
    
    st.markdown("---")
    
    # Configuration Section
    st.markdown("### Configuration")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        default_shape = infer_input_shape(model)
        input_shape = st.text_input(
            "Input Shape (N, C, H, W)",
            value=default_shape,
            help="Batch size, channels, height, width"
        )
    
    with col2:
        opset_version = st.select_slider(
            "ONNX Opset",
            options=list(range(9, 19)),
            value=18,
            help="ONNX operator set version"
        )
    
    with col3:
        dynamic_batch = st.checkbox(
            "Dynamic Batch",
            value=False,
            help="Enable dynamic batch size in ONNX"
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Graph Visualization", "Model Details", "Export"])
    
    # TAB 1: Graph Visualization
    with tab1:
        st.markdown("### Computational Graph")
        
        if st.button("Generate Graph", key="gen_graph", use_container_width=True):
            with st.spinner("Generating computational graph..."):
                graph, error = trace_model(model, input_shape)
                if error:
                    st.error(f"Graph generation failed: {error}")
                else:
                    st.session_state.graph = graph
                    st.success("Graph generated successfully!")
        
        if 'graph' in st.session_state and st.session_state.graph:
            st.graphviz_chart(st.session_state.graph)
        else:
            st.info("Click 'Generate Graph' to visualize the model's computational flow")
    
    # TAB 2: Model Details
    with tab2:
        st.markdown("### Model Architecture Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Information")
            
            info_data = {
                'Type': model_info.get('type', 'Unknown'),
                'Module': model_info.get('module', 'Unknown'),
                'Total Layers': f"{model_info.get('layers', 0)}",
                'Total Parameters': f"{model_info.get('parameters', 0):,}",
                'Trainable Parameters': f"{model_info.get('trainable', 0):,}",
                'Model Size (MB)': f"{model_info.get('parameters', 0) * 4 / (1024 * 1024):.2f}",
            }
            
            info_df = pd.DataFrame(list(info_data.items()), columns=['Metric', 'Value'])
            st.dataframe(info_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Layer Breakdown")
            
            layer_counts = {}
            for module in model.modules():
                layer_type = type(module).__name__
                if layer_type != 'Sequential':
                    layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            
            if layer_counts:
                layers_df = pd.DataFrame(
                    list(layer_counts.items()),
                    columns=['Layer Type', 'Count']
                ).sort_values('Count', ascending=False)
                st.dataframe(layers_df, hide_index=True, use_container_width=True)
    
    # TAB 3: Export
    with tab3:
        st.markdown("### Export to ONNX")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Conversion Settings")
            
            export_info = f"""
            - **Input Shape:** {input_shape}
            - **Opset Version:** {opset_version}
            - **Dynamic Batch:** {'Yes' if dynamic_batch else 'No'}
            """
            st.markdown(export_info)
            
            if st.button("Convert to ONNX", key="convert_onnx", use_container_width=True):
                with st.spinner("Converting to ONNX..."):
                    onnx_bytes, error = export_to_onnx(
                        model,
                        input_shape,
                        opset_version,
                        dynamic_batch
                    )
                    
                    if error:
                        st.error(f"Conversion failed: {error}")
                    else:
                        st.session_state.onnx_bytes = onnx_bytes
                        st.success("Conversion successful!")
        
        with col2:
            st.markdown("#### Download")
            
            if st.session_state.onnx_bytes:
                file_size = len(st.session_state.onnx_bytes) / (1024 * 1024)
                st.info(f"ONNX file ready • {file_size:.2f} MB")
                
                st.download_button(
                    label="Download ONNX Model",
                    data=st.session_state.onnx_bytes,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}.onnx",
                    mime="application/octet-stream",
                    use_container_width=True
                )
                
                st.markdown("""
                #### Next Steps
                - Open in [Netron.app](https://netron.app)
                - Deploy with TensorRT, OpenVINO, or CoreML
                - Use for inference optimization
                """)
            else:
                st.warning("Click 'Convert to ONNX' to generate the ONNX file")

else:
    # Welcome Screen
    st.markdown("""
    <div class="info-card">
        <h3>Get Started</h3>
        <p>Upload a PyTorch model file (.pt, .pth) or SafeTensors model (.safetensors) to begin.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Features")
        features = [
            "🔄 Convert to ONNX format",
            "📊 Visualize computational graphs",
            "📈 Analyze model architecture",
            "💾 Download optimized models",
            "⚡ Support for multiple formats",
        ]
        for feature in features:
            st.markdown(f"- {feature}")
    
    with col2:
        st.markdown("#### Supported Formats")
        formats = [
            ".pt - PyTorch tensor format",
            ".pth - PyTorch model checkpoint",
            ".safetensors - SafeTensors format",
        ]
        for fmt in formats:
            st.markdown(f"✓ {fmt}")
    
    st.markdown("""
    ---
    
    #### How to Use
    
    1. **Upload** your model file using the sidebar
    2. **Configure** input shape and export settings
    3. **Generate** computational graphs to visualize architecture
    4. **Convert** to ONNX for deployment
    5. **Download** and use in your applications
    
    ---
    
    #### Why ONNX?
    
    ONNX (Open Neural Network Exchange) enables:
    - Cross-framework model compatibility
    - Hardware acceleration (TensorRT, OpenVINO, etc.)
    - Production deployment optimization
    - Model standardization
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>NeuroScope Pro</strong> | Advanced PyTorch Model Converter & Visualizer</p>
    <p style="font-size: 0.9em;">Powered by PyTorch, Torchview, and ONNX Runtime</p>
</div>
""", unsafe_allow_html=True)