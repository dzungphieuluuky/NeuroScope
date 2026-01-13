import streamlit as st
import torch
import torch.nn as nn
import importlib.util
import sys
import os
import shutil
import zipfile
import tempfile
import ast
import json
import inspect
import pandas as pd
import gc
from collections import OrderedDict
from streamlit_agraph import agraph, Node, Edge, Config
from torchview import draw_graph
from pathlib import Path
import textwrap
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="NeuroScope Pro: Industry Standard Debugger")

# --- CUSTOM CSS FOR BETTER CODE DISPLAY ---
st.markdown("""
<style>
    .code-container {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        overflow-x: auto;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.5;
        max-height: 500px;
        overflow-y: auto;
    }
    .module-detail-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-box {
        background: rgba(255,255,255,0.1);
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }
    .tensor-info {
        background: #2d3748;
        color: #e2e8f0;
        padding: 12px;
        border-radius: 6px;
        margin: 6px 0;
        font-family: monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: #f7fafc;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITY: PROJECT LOADING & AST ANALYSIS ---

class ProjectLoader:
    """Handles File/Zip uploads and sys.path setup with complete module resolution."""
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.temp_dir = tempfile.mkdtemp()
        self.is_zip = uploaded_file.name.endswith('.zip')
        self.module_cache = {}

    def __enter__(self):
        if self.is_zip:
            zip_path = os.path.join(self.temp_dir, "project.zip")
            with open(zip_path, "wb") as f:
                f.write(self.uploaded_file.getbuffer())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            # Cleanup MacOS junk
            if os.path.exists(os.path.join(self.temp_dir, "__MACOSX")):
                shutil.rmtree(os.path.join(self.temp_dir, "__MACOSX"))
        else:
            file_path = os.path.join(self.temp_dir, self.uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(self.uploaded_file.getbuffer())

        if self.temp_dir not in sys.path:
            sys.path.insert(0, self.temp_dir)
        
        # Pre-import all Python modules in the project for cross-file dependencies
        self._preload_modules()
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir in sys.path:
            sys.path.remove(self.temp_dir)
        # Clear imported modules
        for mod_name in list(self.module_cache.keys()):
            if mod_name in sys.modules:
                del sys.modules[mod_name]
        shutil.rmtree(self.temp_dir)

    def _preload_modules(self):
        """Pre-import all Python modules to resolve dependencies."""
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.temp_dir)
                    module_name = rel_path.replace(os.sep, '.').replace('.py', '')
                    
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, full_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                            self.module_cache[module_name] = module
                    except Exception as e:
                        # Skip modules that can't be imported independently
                        pass

    def get_py_files(self):
        py_files = []
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    py_files.append(os.path.relpath(full_path, self.temp_dir))
        return py_files

class ArchitectureScanner:
    """Scans .py files for classes inheriting from nn.Module with enhanced metadata."""
    
    @staticmethod
    def scan(root_dir, file_rel_paths):
        found_classes = [] # List of tuples (class_name, file_path, imports, dependencies)
        
        for f_path in file_rel_paths:
            full_path = os.path.join(root_dir, f_path)
            try:
                with open(full_path, "r", encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                # Extract imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(f"{node.module}")
                
                # Look for classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        is_module = False
                        for base in node.bases:
                            if isinstance(base, ast.Name) and base.id == 'Module':
                                is_module = True
                            elif isinstance(base, ast.Attribute) and base.attr == 'Module':
                                is_module = True
                        
                        if is_module:
                            found_classes.append({
                                'name': node.name,
                                'file': f_path,
                                'imports': imports,
                                'line': node.lineno
                            })
            except Exception as e:
                continue
        return found_classes

def load_dynamic_class(root_dir, file_rel_path, class_name):
    file_path = os.path.join(root_dir, file_rel_path)
    module_name = file_rel_path.replace(os.sep, '.').replace('.py', '')
    
    try:
        # Check if already loaded
        if module_name in sys.modules:
            return getattr(sys.modules[module_name], class_name)
        
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    except Exception as e:
        raise e

# --- UTILITY: INTERACTIVE ARGUMENT PARSING ---

def generate_init_form(model_class):
    """Introspects __init__, generates Streamlit widgets, returns args dict."""
    sig = inspect.signature(model_class.__init__)
    params = sig.parameters
    
    args = {}
    st.markdown("##### 🛠 Model Configuration")
    
    cols = st.columns(2)
    i = 0
    
    for name, param in params.items():
        if name == 'self': continue
        
        col = cols[i % 2]
        i += 1
        
        default = param.default if param.default is not inspect.Parameter.empty else None
        label = f"`{name}`"
        
        # Type inference and widget generation
        if param.annotation is int or isinstance(default, int):
            val = default if isinstance(default, int) else 0
            args[name] = col.number_input(label, value=val, step=1, help=f"Type: int")
            
        elif param.annotation is float or isinstance(default, float):
            val = default if isinstance(default, float) else 0.0
            args[name] = col.number_input(label, value=val, format="%.4f", help=f"Type: float")
            
        elif param.annotation is bool or isinstance(default, bool):
            val = default if isinstance(default, bool) else False
            args[name] = col.checkbox(label, value=val)
            
        else:
            val = str(default) if default is not None else ""
            input_val = col.text_input(label, value=val, help="Type: String/JSON. Enter lists/dicts as JSON.")
            
            try:
                if (input_val.startswith('[') and input_val.endswith(']')) or \
                   (input_val.startswith('{') and input_val.endswith('}')):
                    args[name] = json.loads(input_val)
                elif input_val.lower() == 'none':
                    args[name] = None
                else:
                    args[name] = input_val
            except:
                args[name] = input_val
                
    return args

# --- ENHANCED TRACER WITH DETAILED INFORMATION ---

class DetailedTracer:
    """Enhanced tracer with Netron-style detailed information."""
    def __init__(self):
        self.data = OrderedDict()
        self.hooks = []
        self.module_source_map = {}

    def register(self, model):
        def hook(name, module):
            def fn(mod, inp, out):
                # Input processing
                in_shapes = []
                in_dtypes = []
                for x in inp:
                    if isinstance(x, torch.Tensor):
                        in_shapes.append(list(x.shape))
                        in_dtypes.append(str(x.dtype))
                
                # Output processing
                if isinstance(out, tuple):
                    out_shapes = [list(x.shape) for x in out if isinstance(x, torch.Tensor)]
                    out_dtypes = [str(x.dtype) for x in out if isinstance(x, torch.Tensor)]
                elif isinstance(out, torch.Tensor):
                    out_shapes = [list(out.shape)]
                    out_dtypes = [str(out.dtype)]
                else:
                    out_shapes = []
                    out_dtypes = []
                
                # Weight and bias statistics
                params_info = {}
                if hasattr(mod, 'weight') and mod.weight is not None:
                    w = mod.weight
                    params_info['weight'] = {
                        'shape': list(w.shape),
                        'dtype': str(w.dtype),
                        'mean': f"{w.mean().item():.6f}",
                        'std': f"{w.std().item():.6f}",
                        'min': f"{w.min().item():.6f}",
                        'max': f"{w.max().item():.6f}",
                        'numel': w.numel()
                    }
                
                if hasattr(mod, 'bias') and mod.bias is not None:
                    b = mod.bias
                    params_info['bias'] = {
                        'shape': list(b.shape),
                        'dtype': str(b.dtype),
                        'mean': f"{b.mean().item():.6f}",
                        'std': f"{b.std().item():.6f}",
                        'numel': b.numel()
                    }
                
                # Module attributes (like kernel_size, stride, etc.)
                module_attrs = {}
                for attr in ['kernel_size', 'stride', 'padding', 'dilation', 'groups', 
                             'in_features', 'out_features', 'in_channels', 'out_channels',
                             'num_heads', 'embed_dim', 'dropout', 'eps', 'momentum']:
                    if hasattr(mod, attr):
                        module_attrs[attr] = getattr(mod, attr)
                
                # Source code extraction
                try:
                    source = inspect.getsource(mod.__class__)
                    source_file = inspect.getsourcefile(mod.__class__)
                except:
                    source = "Source not available"
                    source_file = "Unknown"

                self.data[name] = {
                    'type': mod.__class__.__name__,
                    'module': mod.__class__.__module__,
                    'in_shapes': in_shapes,
                    'in_dtypes': in_dtypes,
                    'out_shapes': out_shapes,
                    'out_dtypes': out_dtypes,
                    'params_total': sum(p.numel() for p in mod.parameters()),
                    'params_trainable': sum(p.numel() for p in mod.parameters() if p.requires_grad),
                    'params_info': params_info,
                    'module_attrs': module_attrs,
                    'source': source,
                    'source_file': source_file
                }
            return fn

        for n, m in model.named_modules():
            if n == "": continue 
            self.hooks.append(m.register_forward_hook(hook(n, m)))

    def clear(self):
        for h in self.hooks: h.remove()
        self.hooks = []
        self.data = OrderedDict()

# --- CODE FORMATTING UTILITIES ---

def format_source_code(source_code, highlight_lines=None):
    """Format source code with syntax highlighting and optional line highlighting."""
    try:
        # Clean up the source code
        lines = source_code.split('\n')
        # Remove common leading whitespace
        min_indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
        cleaned_lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Syntax highlight
        formatter = HtmlFormatter(style='monokai', noclasses=True)
        highlighted = highlight(cleaned_code, PythonLexer(), formatter)
        
        return f'<div class="code-container">{highlighted}</div>'
    except:
        return f'<div class="code-container"><pre>{source_code}</pre></div>'

def display_module_details(module_name, module_info):
    """Display detailed module information in Netron-style."""
    st.markdown(f'<div class="module-detail-card">', unsafe_allow_html=True)
    st.markdown(f"### 🔍 {module_name}")
    st.markdown(f"**Type:** `{module_info['type']}`")
    st.markdown(f"**Module:** `{module_info['module']}`")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create columns for organized display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Tensor Information")
        
        # Input tensors
        st.markdown('<div class="tensor-info">', unsafe_allow_html=True)
        st.markdown("**Inputs:**")
        for i, (shape, dtype) in enumerate(zip(module_info['in_shapes'], module_info['in_dtypes'])):
            st.markdown(f"- Input {i}: `{shape}` | `{dtype}`")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Output tensors
        st.markdown('<div class="tensor-info">', unsafe_allow_html=True)
        st.markdown("**Outputs:**")
        for i, (shape, dtype) in enumerate(zip(module_info['out_shapes'], module_info['out_dtypes'])):
            st.markdown(f"- Output {i}: `{shape}` | `{dtype}`")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### ⚙️ Module Attributes")
        if module_info['module_attrs']:
            attrs_df = pd.DataFrame([module_info['module_attrs']]).T
            attrs_df.columns = ['Value']
            st.dataframe(attrs_df, use_container_width=True)
        else:
            st.info("No special attributes")
    
    # Parameters section
    if module_info['params_info']:
        st.markdown("#### 🎯 Parameters")
        st.markdown(f"**Total Parameters:** `{module_info['params_total']:,}`")
        st.markdown(f"**Trainable Parameters:** `{module_info['params_trainable']:,}`")
        
        for param_name, param_stats in module_info['params_info'].items():
            with st.expander(f"📊 {param_name.upper()} Statistics"):
                stats_df = pd.DataFrame([param_stats]).T
                stats_df.columns = ['Value']
                st.dataframe(stats_df, use_container_width=True)
    
    # Source code
    st.markdown("#### 💻 Source Code")
    st.caption(f"File: `{module_info.get('source_file', 'Unknown')}`")
    formatted_code = format_source_code(module_info['source'])
    st.markdown(formatted_code, unsafe_allow_html=True)

# --- MAIN UI ---

def main():
    st.sidebar.title("🧠 NeuroScope Pro")
    st.sidebar.markdown("**Advanced PyTorch Model Debugger**")
    st.sidebar.markdown("Generate **Torchview** graphs & **ONNX** exports with full module resolution.")
    st.sidebar.divider()

    # 1. UPLOAD
    uploaded_file = st.sidebar.file_uploader("📂 Upload Project (.py / .zip)", type=["py", "zip"])

    if uploaded_file:
        # Use session state to persist results
        if 'scan_results' not in st.session_state or st.session_state.last_upload != uploaded_file.name:
            with ProjectLoader(uploaded_file) as loader:
                files = loader.get_py_files()
                classes = ArchitectureScanner.scan(loader.temp_dir, files)
                st.session_state.scan_results = classes
                st.session_state.last_upload = uploaded_file.name
                st.session_state.temp_dir = loader.temp_dir
        
        found_classes = st.session_state.scan_results
        
        if not found_classes:
            st.error("No `nn.Module` classes found in the uploaded files.")
            st.stop()

        # 2. SELECTION
        cls_options = [f"{c['name']} ({c['file']})" for c in found_classes]
        selected_option = st.sidebar.selectbox("Select Model Class", cls_options)
        selected_cls_name = selected_option.split(" (")[0]
        selected_cls_file = selected_option.split(" (")[1][:-1]

        # Display model info
        selected_class = next(c for c in found_classes if c['name'] == selected_cls_name)
        with st.sidebar.expander("📋 Model Info"):
            st.write(f"**File:** {selected_class['file']}")
            st.write(f"**Line:** {selected_class['line']}")
            st.write(f"**Dependencies:** {len(selected_class['imports'])}")

        # 3. DYNAMIC CONFIGURATION
        init_args = {}
        with ProjectLoader(uploaded_file) as loader:
            try:
                ModelClass = load_dynamic_class(loader.temp_dir, selected_cls_file, selected_cls_name)
                
                with st.sidebar.form("config_form"):
                    init_args = generate_init_form(ModelClass)
                    
                    st.markdown("##### 🔌 Input Tensor Config")
                    shape_str = st.text_input("Input Shape (N, C, H, W)", "1, 3, 224, 224")
                    
                    # Advanced options
                    with st.expander("⚙️ Advanced Options"):
                        export_onnx = st.checkbox("Export ONNX", value=True)
                        opset_version = st.number_input("ONNX Opset Version", value=14, min_value=9, max_value=18)
                        dynamic_axes = st.checkbox("Enable Dynamic Axes", value=False)
                    
                    load_btn = st.form_submit_button("🚀 Compile & Trace", use_container_width=True)
            except Exception as e:
                st.sidebar.error(f"Error inspecting class: {e}")
                st.stop()

        # 4. EXECUTION
        if load_btn:
            with st.spinner("🔄 Compiling Model & Tracing Graph..."):
                with ProjectLoader(uploaded_file) as loader:
                    try:
                        # Instantiate
                        ModelClass = load_dynamic_class(loader.temp_dir, selected_cls_file, selected_cls_name)
                        model = ModelClass(**init_args)
                        model.eval()
                        
                        # Parse Shape
                        shape = [int(x.strip()) for x in shape_str.split(',')]
                        dummy_input = torch.randn(*shape)
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # A. Interactive Trace (Custom Hooks)
                        status_text.text("Registering hooks...")
                        progress_bar.progress(25)
                        
                        tracer = DetailedTracer()
                        tracer.register(model)
                        
                        try:
                            status_text.text("Running forward pass...")
                            progress_bar.progress(50)
                            
                            with torch.no_grad():
                                output = model(dummy_input)
                            st.session_state.trace_data = tracer.data
                            st.session_state.trace_status = "success"
                            st.session_state.model_output = output
                        except Exception as e:
                            st.error(f"Forward Pass Failed: {e}")
                            st.session_state.trace_data = tracer.data
                            st.session_state.trace_status = "failed"

                        # B. Torchview Trace
                        status_text.text("Generating Torchview graph...")
                        progress_bar.progress(75)
                        
                        try:
                            graph_view = draw_graph(model, input_size=tuple(shape), expand_nested=True)
                            st.session_state.graph_dot = graph_view.visual_graph
                        except Exception as e:
                            st.warning(f"Torchview tracing failed: {e}")
                            st.session_state.graph_dot = None

                        # C. ONNX Export
                        if export_onnx:
                            status_text.text("Exporting to ONNX...")
                            progress_bar.progress(90)
                            
                            st.session_state.onnx_bytes = None
                            try:
                                onnx_path = os.path.join(tempfile.gettempdir(), "model.onnx")
                                
                                export_params = {
                                    'opset_version': opset_version,
                                    'do_constant_folding': True,
                                    'input_names': ['input'],
                                    'output_names': ['output']
                                }
                                
                                if dynamic_axes:
                                    export_params['dynamic_axes'] = {
                                        'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}
                                    }
                                
                                torch.onnx.export(model, dummy_input, onnx_path, **export_params)
                                
                                with open(onnx_path, "rb") as f:
                                    st.session_state.onnx_bytes = f.read()
                            except ImportError:
                                st.error("ONNX Export failed: Missing 'onnxscript'. Run `pip install onnxscript`.")
                            except Exception as e:
                                st.warning(f"ONNX Export failed: {e}")
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Complete!")
                        
                        # Model summary
                        total_params = sum(p.numel() for p in model.parameters())
                        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        st.session_state.model_summary = {
                            'total_params': total_params,
                            'trainable_params': trainable_params,
                            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
                        }

                    except Exception as e:
                        st.error(f"Critical Error: {e}")
                        st.exception(e)

    # --- VISUALIZATION TABS ---
    
    if 'trace_data' in st.session_state:
        # Display model summary at top
        if 'model_summary' in st.session_state:
            summary = st.session_state.model_summary
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Parameters", f"{summary['total_params']:,}")
            col2.metric("Trainable Parameters", f"{summary['trainable_params']:,}")
            col3.metric("Model Size", f"{summary['model_size_mb']:.2f} MB")
            col4.metric("Trace Status", "✅ Success" if st.session_state.trace_status == "success" else "⚠️ Failed")
        
        st.divider()
        
        t1, t2, t3, t4 = st.tabs(["🔎 Interactive Debugger", "🕸️ Torchview Graph", "📊 Module Analysis", "📤 Export"])
        
        # TAB 1: INTERACTIVE DEBUGGER
        with t1:
            data = st.session_state.trace_data
            
            if not data:
                st.warning("No trace data available. Model may have failed during forward pass.")
            else:
                col_graph, col_info = st.columns([2, 1])
                
                with col_graph:
                    st.markdown("### 🗺️ Computational Graph")
                    
                    # Build Node Graph
                    nodes, edges = [], []
                    keys = list(data.keys())
                    
                    for i, (k, v) in enumerate(data.items()):
                        color = "#ffcccb" if st.session_state.trace_status == "failed" and i == len(data)-1 else "#90EE90"
                        
                        label = f"{k.split('.')[-1]}\n{v['type']}\n{v['out_shapes'][0] if v['out_shapes'] else 'N/A'}"
                        nodes.append(Node(id=k, label=label, shape="box", color=color, size=25))
                        if i > 0:
                            edges.append(Edge(source=keys[i-1], target=k))
                    
                    config = Config(
                        width="100%", 
                        height=700, 
                        directed=True, 
                        nodeHighlightBehavior=True,
                        highlightColor="#667eea"
                    )
                    sel_node = agraph(nodes, edges, config)

                with col_info:
                    st.markdown("### 📌 Quick Info")
                    st.caption("Click on a node in the graph to see details")
                    
                    if sel_node and sel_node in data:
                        info = data[sel_node]
                        st.markdown(f"**Selected:** `{sel_node}`")
                        st.markdown(f"**Type:** `{info['type']}`")
                        st.markdown(f"**Parameters:** `{info['params_total']:,}`")
                        
                        if info['out_shapes']:
                            st.markdown(f"**Output Shape:** `{info['out_shapes'][0]}`")
                        
                        if st.button("📋 View Full Details", use_container_width=True):
                            st.session_state.selected_module = sel_node
                    else:
                        st.info("Select a module from the graph")

        # TAB 2: TORCHVIEW
        with t2:
            st.markdown("### 🌐 High-Fidelity Structural Graph")
            st.caption("Powered by `torchview`. Shows strict data dependencies and nested module structure.")
            
            if 'graph_dot' in st.session_state and st.session_state.graph_dot:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.graphviz_chart(st.session_state.graph_dot)
                
                with col2:
                    st.markdown("#### 🎯 Graph Legend")
                    st.markdown("""
                    - **Rectangles**: Operations/Layers
                    - **Arrows**: Data flow
                    - **Labels**: Tensor shapes
                    - **Colors**: Operation types
                    """)
                    
                    if st.button("🔍 Inspect Module", use_container_width=True):
                        st.info("Click on nodes in the Interactive Debugger tab for detailed inspection")
            else:
                st.warning("Graph not available. Model may be incompatible with Torchview.")

# TAB 3: MODULE ANALYSIS
        with t3:
            st.markdown("### 🔬 Detailed Module Analysis")
            
            if 'selected_module' in st.session_state:
                module_name = st.session_state.selected_module
                if module_name in data:
                    display_module_details(module_name, data[module_name])
            else:
                st.info("👈 Select a module from the Interactive Debugger tab to view detailed analysis")
                
                # Show all modules list
                st.markdown("#### 📚 All Modules")
                module_list = list(data.keys())
                
                # Create searchable/filterable module list
                col1, col2 = st.columns([2, 1])
                with col1:
                    search_term = st.text_input("🔍 Search modules", placeholder="e.g., conv, linear, attention")
                with col2:
                    filter_type = st.selectbox("Filter by type", ["All"] + list(set([data[m]['type'] for m in module_list])))
                
                # Filter modules
                filtered_modules = module_list
                if search_term:
                    filtered_modules = [m for m in filtered_modules if search_term.lower() in m.lower()]
                if filter_type != "All":
                    filtered_modules = [m for m in filtered_modules if data[m]['type'] == filter_type]
                
                st.caption(f"Showing {len(filtered_modules)} of {len(module_list)} modules")
                
                # Display modules in a grid
                for i in range(0, len(filtered_modules), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(filtered_modules):
                            mod_name = filtered_modules[i + j]
                            mod_info = data[mod_name]
                            
                            with col:
                                with st.container(border=True):
                                    st.markdown(f"**{mod_name}**")
                                    st.caption(f"Type: `{mod_info['type']}`")
                                    
                                    # Mini stats
                                    subcol1, subcol2 = st.columns(2)
                                    subcol1.metric("Params", f"{mod_info['params_total']:,}", label_visibility="collapsed")
                                    if mod_info['out_shapes']:
                                        subcol2.metric("Output", str(mod_info['out_shapes'][0]), label_visibility="collapsed")
                                    
                                    if st.button(f"🔍 Inspect", key=f"inspect_{mod_name}", use_container_width=True):
                                        st.session_state.selected_module = mod_name
                                        st.rerun()

        # TAB 4: EXPORT
        with t4:
            st.markdown("### 📦 Model Export & Documentation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 ONNX Export")
                if st.session_state.get('onnx_bytes'):
                    st.success("✅ ONNX model generated successfully!")
                    st.markdown("""
                    **Next Steps:**
                    1. Download the `.onnx` file below
                    2. Open it in [Netron.app](https://netron.app)
                    3. Explore the computational graph interactively
                    
                    **Benefits:**
                    - Cross-platform model visualization
                    - Detailed operator-level inspection
                    - Compatible with deployment frameworks
                    """)
                    
                    # ONNX file info
                    onnx_size = len(st.session_state.onnx_bytes) / (1024 * 1024)
                    st.info(f"📊 ONNX File Size: {onnx_size:.2f} MB")
                    
                    st.download_button(
                        label="⬇️ Download ONNX Model",
                        data=st.session_state.onnx_bytes,
                        file_name="model_neuroscope.onnx",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                    
                    # Additional info
                    with st.expander("ℹ️ About ONNX Format"):
                        st.markdown("""
                        **ONNX (Open Neural Network Exchange)** is an open format for representing machine learning models.
                        
                        - Supports model deployment across frameworks
                        - Enables hardware acceleration optimization
                        - Industry-standard for model interchange
                        - Compatible with TensorRT, OpenVINO, CoreML, and more
                        """)
                else:
                    st.error("❌ ONNX export failed or was disabled")
                    st.markdown("""
                    **Possible reasons:**
                    - ONNX export was disabled in configuration
                    - Model contains unsupported operations
                    - Dynamic control flow detected
                    
                    **Solutions:**
                    - Enable ONNX export in the sidebar
                    - Simplify model architecture
                    - Check error logs above
                    """)
            
            with col2:
                st.markdown("#### 📄 Model Report")
                
                # Generate comprehensive report
                if st.button("📝 Generate Full Report", use_container_width=True):
                    report_lines = []
                    report_lines.append("# Model Architecture Report\n")
                    report_lines.append(f"**Generated:** {pd.Timestamp.now()}\n")
                    report_lines.append(f"**Model Class:** {selected_cls_name}\n")
                    report_lines.append(f"**Source File:** {selected_cls_file}\n\n")
                    
                    if 'model_summary' in st.session_state:
                        summary = st.session_state.model_summary
                        report_lines.append("## Model Summary\n\n")
                        report_lines.append(f"| Metric | Value |\n")
                        report_lines.append(f"|--------|-------|\n")
                        report_lines.append(f"| Total Parameters | {summary['total_params']:,} |\n")
                        report_lines.append(f"| Trainable Parameters | {summary['trainable_params']:,} |\n")
                        report_lines.append(f"| Model Size | {summary['model_size_mb']:.2f} MB |\n\n")
                    
                    report_lines.append("## Layer-by-Layer Analysis\n\n")
                    
                    for idx, (mod_name, mod_info) in enumerate(data.items(), 1):
                        report_lines.append(f"### {idx}. {mod_name}\n\n")
                        report_lines.append(f"**Type:** `{mod_info['type']}`  \n")
                        report_lines.append(f"**Module Path:** `{mod_info['module']}`  \n")
                        report_lines.append(f"**Total Parameters:** {mod_info['params_total']:,}  \n")
                        report_lines.append(f"**Trainable Parameters:** {mod_info['params_trainable']:,}  \n\n")
                        
                        if mod_info['in_shapes']:
                            report_lines.append(f"**Input Shapes:**\n")
                            for i, (shape, dtype) in enumerate(zip(mod_info['in_shapes'], mod_info['in_dtypes'])):
                                report_lines.append(f"- Input {i}: `{shape}` ({dtype})\n")
                            report_lines.append("\n")
                        
                        if mod_info['out_shapes']:
                            report_lines.append(f"**Output Shapes:**\n")
                            for i, (shape, dtype) in enumerate(zip(mod_info['out_shapes'], mod_info['out_dtypes'])):
                                report_lines.append(f"- Output {i}: `{shape}` ({dtype})\n")
                            report_lines.append("\n")
                        
                        if mod_info['module_attrs']:
                            report_lines.append(f"**Attributes:**\n")
                            for attr, val in mod_info['module_attrs'].items():
                                report_lines.append(f"- {attr}: `{val}`\n")
                            report_lines.append("\n")
                        
                        if mod_info['params_info']:
                            report_lines.append(f"**Parameter Details:**\n\n")
                            for param_name, param_stats in mod_info['params_info'].items():
                                report_lines.append(f"*{param_name.capitalize()}:*\n")
                                report_lines.append(f"```\n")
                                for stat_key, stat_val in param_stats.items():
                                    report_lines.append(f"{stat_key}: {stat_val}\n")
                                report_lines.append(f"```\n\n")
                        
                        report_lines.append("---\n\n")
                    
                    report = "".join(report_lines)
                    
                    st.download_button(
                        label="⬇️ Download Report (Markdown)",
                        data=report,
                        file_name=f"model_report_{selected_cls_name}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                    
                    st.success("✅ Report generated successfully!")
                
                # Quick stats preview
                st.markdown("#### 📊 Quick Statistics")
                
                if data:
                    # Calculate statistics
                    total_layers = len(data)
                    param_layers = sum(1 for m in data.values() if m['params_total'] > 0)
                    total_params = sum(m['params_total'] for m in data.values())
                    
                    stats_df = pd.DataFrame({
                        'Metric': ['Total Layers', 'Parametric Layers', 'Total Parameters'],
                        'Value': [total_layers, param_layers, f"{total_params:,}"]
                    })
                    
                    st.dataframe(stats_df, hide_index=True, use_container_width=True)
            
            # Additional export options
            st.divider()
            st.markdown("#### 🔧 Additional Export Options")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                st.markdown("**📊 Parameters CSV**")
                if st.button("Generate CSV", key="export_csv", use_container_width=True):
                    params_data = []
                    for mod_name, mod_info in data.items():
                        params_data.append({
                            'Module': mod_name,
                            'Type': mod_info['type'],
                            'Total_Parameters': mod_info['params_total'],
                            'Trainable_Parameters': mod_info['params_trainable'],
                            'Output_Shape': str(mod_info['out_shapes'][0]) if mod_info['out_shapes'] else 'N/A'
                        })
                    
                    df = pd.DataFrame(params_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="⬇️ Download CSV",
                        data=csv,
                        file_name=f"model_parameters_{selected_cls_name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col4:
                st.markdown("**📋 Layer Summary**")
                if st.button("Generate Summary", key="layer_summary", use_container_width=True):
                    # Create a summary of layer types
                    layer_types = {}
                    for mod_info in data.values():
                        layer_type = mod_info['type']
                        if layer_type not in layer_types:
                            layer_types[layer_type] = {'count': 0, 'params': 0}
                        layer_types[layer_type]['count'] += 1
                        layer_types[layer_type]['params'] += mod_info['params_total']
                    
                    summary_lines = ["# Layer Type Summary\n\n"]
                    summary_lines.append("| Layer Type | Count | Total Parameters |\n")
                    summary_lines.append("|------------|-------|------------------|\n")
                    
                    for layer_type, stats in sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True):
                        summary_lines.append(f"| {layer_type} | {stats['count']} | {stats['params']:,} |\n")
                    
                    summary = "".join(summary_lines)
                    
                    st.download_button(
                        label="⬇️ Download Summary",
                        data=summary,
                        file_name=f"layer_summary_{selected_cls_name}.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            with col5:
                st.markdown("**🔄 Session**")
                if st.button("Reset Session", key="reset_session", use_container_width=True):
                    # Clear all session state
                    keys_to_delete = list(st.session_state.keys())
                    for key in keys_to_delete:
                        del st.session_state[key]
                    st.success("Session reset! Refresh page to start over.")
                    st.rerun()
    
    else:
        # Welcome screen
        st.markdown("""
        # 🧠 Welcome to NeuroScope Pro
        
        ### The Ultimate PyTorch Model Debugger
        
        **Features:**
        - 📂 **Multi-file Project Support** - Upload entire projects as ZIP files with automatic dependency resolution
        - 🔍 **Interactive Graph Visualization** - Click and explore your model's computational graph in real-time
        - 📊 **Detailed Module Analysis** - Netron-style detailed information for every layer including tensor shapes and statistics
        - 💻 **Source Code Inspection** - View and analyze module implementations with syntax highlighting
        - 📤 **ONNX Export** - Export your models for deployment and external visualization tools
        - 📈 **Performance Metrics** - Track parameters, shapes, and data flow through your network
        - 🎯 **Smart Module Detection** - Automatically discovers all nn.Module classes in your project
        
        ### Getting Started
        
        1. **Upload** your PyTorch project (`.py` file or `.zip` archive) using the sidebar
        2. **Select** the model class you want to analyze from the detected classes
        3. **Configure** initialization parameters and input tensor shapes interactively
        4. **Click** "Compile & Trace" to analyze your model architecture
        5. **Explore** the interactive visualizations and detailed module information across tabs
        
        ### Supported Features
        
        - ✅ Custom `nn.Module` classes with complex architectures
        - ✅ Multi-file projects with cross-module dependencies and imports
        - ✅ Dynamic argument configuration with type inference
        - ✅ Comprehensive parameter statistics (mean, std, min, max)
        - ✅ Interactive computational graph with node selection
        - ✅ Syntax-highlighted source code viewing with smart formatting
        - ✅ ONNX export with configurable opset versions (9-18)
        - ✅ Detailed model reports and documentation generation
        - ✅ CSV export for parameters and layer analysis
        
        ### Advanced Capabilities
        
        - **Cross-file dependency resolution** - Automatically handles imports between project files
        - **Weight statistics** - Detailed analysis of weights and biases for each layer
        - **Tensor flow tracking** - Monitor shapes and dtypes through the entire network
        - **Module attributes** - Inspect kernel sizes, strides, padding, and other configuration
        - **Searchable module list** - Filter and find specific layers quickly
        - **Export flexibility** - Multiple output formats for different use cases
        
        ---
        
        **👈 Get started by uploading a file in the sidebar**
        """)
        
        # Example code snippet
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📝 Example: Simple CNN"):
                st.code('''
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128 * 56 * 56, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
                ''', language='python')
        
        with col2:
            with st.expander("📝 Example: Multi-file Project"):
                st.code('''
# File: modules.py
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# File: model.py
from modules import ConvBlock

class CustomNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = ConvBlock(3, 64)
        self.block2 = ConvBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
                ''', language='python')
        
        # Tips section
        st.markdown("---")
        st.markdown("### 💡 Pro Tips")
        
        tip_col1, tip_col2, tip_col3 = st.columns(3)
        
        with tip_col1:
            st.info("""
            **📦 ZIP Projects**
            
            Upload entire project folders as ZIP files. NeuroScope will automatically:
            - Detect all Python files
            - Resolve imports
            - Find all nn.Module classes
            """)
        
        with tip_col2:
            st.info("""
            **🔍 Module Inspection**
            
            Click on any node in the Interactive Debugger to:
            - View detailed statistics
            - Inspect source code
            - Analyze tensor shapes
            """)
        
        with tip_col3:
            st.info("""
            **📊 Export Options**
            
            Multiple export formats:
            - ONNX for deployment
            - Markdown reports
            - CSV for analysis
            - Layer summaries
            """)

if __name__ == "__main__":
    main()